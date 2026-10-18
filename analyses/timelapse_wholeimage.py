#!/usr/bin/env python3
"""
Flimera / EzTime FLIM analysis -- Timelapse (whole image)

Same time-window splitting as analyses/timelapse.py, but instead of fitting
every masked super-pixel separately per window, this pools the WHOLE masked
region's photons into one curve per window and fits that once -- trading
spatial resolution for a much higher photon count per fit.

The mask itself is chosen ONCE, from the whole acquisition (all windows
combined), so every window sums photons from the same fixed super-pixels.

REQUIRES a RAW photon-stream file.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from flim_core.config import MIN_NEIGHBORS, BIN_SIZE, DEFAULT_THRESHOLD, TAIL_START_OFFSET
from flim_core.masking import bin_cube, report_dead_pixels, clean_mask, prompt_roi, load_mask
from flim_core.fitting import _tail_axis
from flim_core.plotting import plot_overview, plot_overview_and_mask
from flim_core.fitting import AnalysisCancelled
from flim_core.irf import (
    load_irf, estimate_shift, shift_array,
    global_reconv_fit, global_reconv_fit_mono,
    plot_irf_diagnostic,
)
from analyses._common import (
    window_boundaries,
    apply_spatial_and_time_binning,
    load_raw_with_time, build_group_cube, crop_photon_data,
    fit_window_sum, fit_window_sum_reconv, fit_window_sum_mean_arrival,
    fit_window_sum_mean_arrival_reconv,
    plot_window_fit, plot_window_fit_reconv, plot_pooled_mean_arrival,
    nice_tick_interval, save_trend_npz,
)
from analyses.timelapse import assign_time_windows

NAME = 'Timelapse (whole image)'
DESCRIPTION = 'Split into time windows, pool the masked region per window, fit one tau per window.'
REQUIRES_RAW = True
IRF_CAPABLE = True
EXTRA_ARGS = ['step']


def preview_slice(loaded_data, settings):
    """Same as analyses/timelapse.py's preview_slice -- first time window."""
    step_s = settings["extra"]["step"]
    if step_s <= 0:
        raise ValueError("Step size must be greater than 0.")
    window_idx, n_win = assign_time_windows(loaded_data["time_ms"], step_s * 1000.0,
                                            loaded_data["duration_ms"])
    b = window_boundaries(window_idx, n_win)
    sel = slice(b[0], b[1])   # first window as a slice (view), not a full-length boolean mask
    img, cube = build_group_cube(loaded_data, sel)
    return img, cube, "first time window"


def plot_trend(results, labels, x, symbol, prefix, show):
    """Final window: one fitted lifetime per time window, plus a straight-line
    trend fit through the points. X-axis ticks use a round time interval
    (e.g. every 10 or 20 s) rather than one tick per window, so this stays
    readable regardless of how many windows there are."""
    vals = np.array([r["val"] if r is not None else np.nan for r in results])
    finite = np.isfinite(vals)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(x, vals, "o-", color="#3477c4", lw=1.5, markersize=6,
           markeredgecolor="white", markeredgewidth=0.8, zorder=2, label="measured")

    if finite.sum() >= 2:
        slope, intercept = np.polyfit(x[finite], vals[finite], 1)
        xf = np.array([x.min(), x.max()])
        ax.plot(xf, slope * xf + intercept, "--", color="#c44", lw=1.5, zorder=1,
               label=f"linear trend: {slope:+.3f} ns/s")
        ax.legend(fontsize=9)

    step = nice_tick_interval(x.max() - x.min() if len(x) > 1 else 1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(step))
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(f"{symbol} (ns)")
    ax.set_title(f"{symbol} vs. time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = f"{prefix}_trend_summed.png"
    fig.savefig(out_png, dpi=140)
    print(f"\nSaved trend: {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Time-lapse FLIM analysis: one pooled-ROI fit per time window")
    ap.add_argument("h5file", help="RAW Flimera .H5 photon-stream file (not an export/.ds3d project)")
    ap.add_argument("--step", type=float, default=None,
                    help="time window duration, in seconds (prompts if omitted)")
    ap.add_argument("--threshold", type=int, default=None,
                    help="min photons/super-pixel (measured over the WHOLE acquisition) to include "
                         "in the mask (skips the interactive prompt)")
    ap.add_argument("--tail-start-offset", type=int, default=None,
                    help=f"bins after the decay peak where the tail fit starts "
                         f"(default {TAIL_START_OFFSET}). No effect with --fit-method irf.")
    ap.add_argument("--fit-method", choices=["tail", "irf", "mean", "mean_irf"], default="tail",
                    help="'tail' (default): fit the decay tail only. 'irf': fit the FULL curve "
                         "against a measured instrument response (reconvolution) -- requires --irf. "
                         "'mean': no fit at all -- compute the photon-weighted mean arrival time per "
                         "window (fast, no bounds; --mono/--fix-tau-short have no effect).")
    ap.add_argument("--irf", default=None,
                    help="IRF measurement file (any supported format), required with --fit-method irf. "
                         "Should be measured under the same timing/detector settings as this data.")
    ap.add_argument("--no-shift", action="store_true",
                    help="fix the IRF/data timing offset at 0 instead of fitting it. Only use this "
                         "when the IRF and data were acquired at matched laser settings in the same "
                         "session. No effect without --fit-method irf or mean_irf.")
    ap.add_argument("--irf-full", action="store_true",
                    help="use the IRF's full background-subtracted trace (out to --irf-window-end-ns "
                         "after the main peak) instead of windowing to just the main peak -- for when "
                         "secondary IRF peaks are a real, reproducible part of the instrument response.")
    ap.add_argument("--irf-window-end-ns", type=float, default=15.0,
                    help="with --irf-full, how far past the IRF's main peak (ns) to keep (default 15.0).")
    ap.add_argument("--fix-tau-short", action="store_true",
                    help="fix tau_short to a reference global value; float only tau_long "
                         "(no effect with --mono)")
    ap.add_argument("--tau-short-value", type=float, default=None,
                    help="use with --fix-tau-short to supply an exact tau_short value (ns) "
                         "yourself, instead of deriving it automatically.")
    ap.add_argument("--tau-short-bounds", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="optimizer search range (ns) for the short lifetime in a biexponential fit "
                         "(default from flim_core/config.py). Use to match a specific dye.")
    ap.add_argument("--tau-long-bounds", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="optimizer search range (ns) for the long lifetime in a biexponential fit "
                         "(default from flim_core/config.py).")
    ap.add_argument("--tau-mono-bounds", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="optimizer search range (ns) for a mono-exponential fit (--mono) "
                         "(default from flim_core/config.py).")
    ap.add_argument("--mono", action="store_true",
                    help="fit a mono-exponential model instead of the default biexponential")
    ap.add_argument("--min-neighbors", type=int, default=None,
                    help=f"neighbour filter for the mask (default {MIN_NEIGHBORS}); 0 disables it")
    ap.add_argument("--bin", type=int, default=BIN_SIZE,
                    help=f"spatial binning factor NxN (default {BIN_SIZE})")
    ap.add_argument("--moving-average", type=int, default=None,
                    help="accepted for consistency with the pixel-wise plugins but IGNORED here: "
                         "this plugin pools pixels into one curve, and moving-averaged pixels share "
                         "photons, so pooling them would count each photon up to N**2 times.")
    ap.add_argument("--time-bin", type=int, default=1,
                    help="group every N adjacent microtime bins into one (default 1 = none); "
                         "--tail-start-offset is rescaled automatically so the real ns margin "
                         "past the peak is unchanged.")
    ap.add_argument("--roi", type=lambda s: tuple(int(p) for p in s.split(",")), default=None,
                    help="crop to a rectangular region of interest before anything else: "
                         "x0,y0,x1,y1 in pixels (x1/y1 exclusive)")
    ap.add_argument("--roi-prompt", action="store_true",
                    help="interactively ask for an ROI first (blank input keeps the full image)")

    ap.add_argument("--external-mask", default=None,
                    help="path to a previously-saved mask (.npy, see flim_core.masking.save_mask) -- "
                         "if given, use this mask directly and skip threshold/min-neighbours entirely. "
                         "Its shape must match the current image AFTER --bin/--roi are applied. Useful "
                         "for analyzing the exact same pixels across acquisitions that need different "
                         "thresholds (e.g. the same sample at different laser powers).")
    ap.add_argument("--no-overview-plot", dest="save_overview", action="store_false", default=True,
                    help="skip saving the overview/mask-preview plot")
    ap.add_argument("--no-results-plot", dest="save_results", action="store_false", default=True,
                    help="skip saving the per-window pooled-fit plot")
    ap.add_argument("--no-irf-check-plot", dest="save_irf_check", action="store_false", default=True,
                    help="skip saving the IRF diagnostic plot (only relevant with --fit-method irf)")
    ap.add_argument("--no-show", action="store_true", help="save figures without displaying")
    ap.add_argument("--prefix", default=None, help="optional filename prefix")
    args = ap.parse_args()
    run(args)


def run(args):
    """Everything main() does after parsing arguments -- see periodic.py's
    run() docstring for why this split exists (direct GUI invocation)."""
    show = not args.no_show
    fit_method = getattr(args, "fit_method", "tail")
    irf = None
    no_shift = getattr(args, "no_shift", False)
    if fit_method in ("irf", "mean_irf"):
        irf_path = getattr(args, "irf", None)
        if not irf_path:
            print(f"Error: --fit-method {fit_method} requires --irf <file> (an IRF measurement file).")
            return
        irf_full = getattr(args, "irf_full", False)
        print(f"Loading IRF from {irf_path} "
              f"({'full trace (echoes kept)' if irf_full else 'windowed to the main peak'}) ...")
        irf = load_irf(irf_path, window=not irf_full,
                       window_end_ns=getattr(args, "irf_window_end_ns", 15.0),
                       time_bin=(getattr(args, "time_bin", None) or 1),
                       pixel_offset_path=getattr(args, "pixel_offset", None))
        if no_shift:
            print("  --no-shift: IRF/data timing offset fixed at 0 (not fitted) -- "
                  "only appropriate if IRF and data were acquired at matched laser settings.")
        print(f"  IRF: {irf['source']} | {irf['n_photons']} raw photons | "
              f"FWHM {irf['fwhm_bins']} bins ({irf['fwhm_bins']*irf['tpb']*1e9:.2f} ns) | "
              f"{irf['bg_fraction']*100:.0f}% of raw counts were flat background (removed)")

    stem = Path(args.h5file).stem
    prefix = f"{args.prefix}_{stem}" if args.prefix else stem

    print(f"Loading raw photon stream from {args.h5file} ...")
    try:
        d = load_raw_with_time(args.h5file)
    except ValueError as e:
        print(f"Error: {e}")
        return
    print(f"  {d['n_photons']} photons | sensor {d['W']}x{d['H']} | {d['n_frames']} frames | "
          f"frame rate {d['frame_rate_hz']:.1f} Hz | acquisition duration {d['duration_ms']/1000:.3f} s")

    if irf is not None and abs(d["tpb"] * (getattr(args, "time_bin", None) or 1) - irf["tpb"]) > 1e-15:
        print(f"  WARNING: data TimePerBin ({d['tpb'] * (getattr(args, 'time_bin', None) or 1):.3e} s) differs from IRF's "
              f"({irf['tpb']:.3e} s) -- these were likely acquired under different "
              f"timing settings and this IRF may not be valid for this data.")

    # --- optional ROI crop ---
    roi = args.roi
    if roi is None and args.roi_prompt:
        preview_img = np.bincount(d["y"] * d["W"] + d["x"], minlength=d["H"] * d["W"]) \
                        .reshape(d["H"], d["W"]).astype(float)
        plot_overview(preview_img, f"{prefix}_roi_preview_overview.png", show,
                      title_suffix="  (whole acquisition, for ROI selection)")
        roi = prompt_roi(d["W"], d["H"])
    if roi is not None:
        d, roi = crop_photon_data(d, roi)

    step_s = args.step
    if step_s is None:
        resp = input("Time window step, in seconds (e.g. 5 for 0-5s, 5-10s, ...) [5.0]: ").strip()
        step_s = float(resp) if resp else 5.0
    step_ms = step_s * 1000.0

    window_idx, n_windows = assign_time_windows(d["time_ms"], step_ms, d["duration_ms"])
    boundaries = window_boundaries(window_idx, n_windows)   # slices, not per-window boolean masks
    print(f"\nStep = {step_s} s over {d['duration_ms']/1000:.3f} s total  ->  {n_windows} windows")
    window_descs, window_t0, window_t1 = [], [], []
    for w in range(n_windows):
        t0 = w * step_s
        t1 = min((w + 1) * step_s, d["duration_ms"] / 1000.0)
        window_descs.append(f"{t0:.2f}-{t1:.2f}s")
        window_t0.append(t0); window_t1.append(t1)

    # --- mask is chosen ONCE, from the WHOLE acquisition (all photons, all
    # windows combined) -- not from a single window -- so every window sums
    # photons from exactly the same fixed set of super-pixels. ---
    full_img, full_cube = build_group_cube(d, slice(None))
    full_img_orig = full_img   # unbinned, for the overview panel
    report_dead_pixels(full_img)
    # Spatial binning + time binning via the shared helper (moving average is
    # deliberately not offered in whole-image plugins -- see
    # periodic_wholeimage.py / normal_wholeimage.py for why). d_native /
    # args_native are kept for the per-window calls below, which must start
    # from the UN-binned d; d and args are rebound to the post-binning copies.
    d_native, args_native = d, args
    full_img, full_cube, d, args, binning_used = apply_spatial_and_time_binning(
        full_img, full_cube, d, args, allow_moving_average=False)

    # --threshold is meant to carry the same "photons in about one window"
    # meaning it has in the pixel-wise plugin (analyses/timelapse.py), even
    # though the MASK itself is deliberately chosen from photons pooled
    # across the whole acquisition (a more stable decision than any single,
    # noisier window could give -- see the comment above). So the threshold
    # comparison and the overview display both use the per-window-EQUIVALENT
    # average (full_img / n_windows), not the raw cumulative total.
    # Comparing the raw cumulative total against the same --threshold value
    # used for a single window silently makes the effective threshold
    # n_windows times more lenient here -- with 5 windows, this produced a
    # mask roughly 3x too large in practice (thresholds meant for ~1s of
    # data being applied to ~5s of pooled counts instead).
    full_img_orig = full_img_orig / n_windows
    full_img = full_img / n_windows

    unit = "super-pixel" if binning_used else "pixel"

    external_mask_path = getattr(args, "external_mask", None)
    if external_mask_path:
        print(f"\nLoading external mask from {external_mask_path} ...")
        mask = load_mask(external_mask_path)
        if mask.shape != full_img.shape:
            print(f"Error: external mask shape {mask.shape} does not match the current "
                  f"image shape {full_img.shape} -- check that --bin/--roi match how the "
                  f"mask was saved (masks are resolution-specific).")
            return
        threshold = None
        min_neighbors = None
        removed = np.zeros_like(mask)
        print(f"  external mask: {int(mask.sum())} {unit}s selected")
    else:
        if args.threshold is not None:
            threshold = args.threshold
        else:
            nz = full_img[full_img > 0]
            default = int(np.percentile(nz, 75)) if nz.size else DEFAULT_THRESHOLD
            print(f"\nChoose the mask threshold using the WHOLE acquisition ({d['duration_ms']/1000:.2f}s, "
                  f"all windows combined) as reference -- this fixed mask is reused for every window.")
            print(f"Photon counts below are averaged per window ({n_windows} windows total), so this "
                  f"threshold means the same thing as --threshold in the pixel-wise plugin.")
            print(f"(per-{unit}-per-window photons: median {int(np.median(nz))}, "
                  f"90th pct {int(np.percentile(nz, 90))}; suggested default {default})")
            resp = input(f"Minimum photons/{unit}/window [{default}]: ").strip()
            threshold = int(resp) if resp else default

        min_neighbors = args.min_neighbors if args.min_neighbors is not None else MIN_NEIGHBORS
        raw_mask = full_img > threshold
        mask = clean_mask(raw_mask, min_neighbors)
        removed = raw_mask & ~mask
        print(f"\nMask threshold = {threshold}  ->  {int(raw_mask.sum())} {unit}s above threshold, "
              f"{int(mask.sum())} kept after neighbour filter")
    if mask.sum() == 0:
        print(f"No {unit}s remain; lower the threshold or the neighbour requirement.")
        return

    # --- overview + mask preview, in one file (matches the GUI's display) ---
    if getattr(args, "save_overview", True):
        plot_overview_and_mask(full_img_orig, full_img, mask, removed, threshold, min_neighbors,
                               args.bin, f"{prefix}_overview.png", show)

    # --- IRF alignment: done ONCE for the whole file (not per window), since
    # it reflects the optical/detector system's timing, not something that
    # should vary within one measurement -- and re-estimating it per window
    # would be noisier, especially for short windows with fewer photons.
    irf_aligned = None
    if fit_method in ("irf", "mean_irf"):
        whole_signal_sum = full_cube[mask].sum(axis=0).astype(float)
        if no_shift:
            # Skip the external pre-alignment entirely -- the fit models below
            # also fix shift at 0, so pre-shifting the kernel here would just
            # reintroduce an offset through the back door.
            shift = 0.0
            irf_aligned = irf["irf_norm"]
        else:
            shift = estimate_shift(whole_signal_sum, irf["irf_norm"])
            irf_aligned = shift_array(irf["irf_norm"], shift)
        print(f"\n  IRF-to-data alignment (whole acquisition, masked region): "
              f"shift = {shift} bins ({shift*d['tpb']*1e9:.3f} ns)"
              f"{' (fixed, not fitted)' if no_shift else ''}")
        if fit_method == "irf":
            # mean_irf has no reference fit / diagnostic plot -- only the alignment above.
            if args.mono:
                g_ref = global_reconv_fit_mono(whole_signal_sum, irf_aligned, d["tpb"] * 1e9, no_shift=no_shift)
                print(f"  reference reconvolution fit (whole acquisition): tau = {g_ref['tau']:.2f} ns")
            else:
                g_ref = global_reconv_fit(whole_signal_sum, irf_aligned, d["tpb"] * 1e9, no_shift=no_shift)
                print(f"  reference reconvolution fit (whole acquisition): "
                      f"tau_short = {g_ref['tau_short']:.2f} ns, tau_long = {g_ref['tau_long']:.2f} ns")
            if getattr(args, "save_irf_check", True):
                plot_irf_diagnostic(irf, whole_signal_sum, shift, g_ref, d["tpb"],
                                    f"{prefix}_irf_check.png", show,
                                    n_signal=int(mask.sum()), unit=unit)

    # --- for each window, sum the SAME masked super-pixels' photons into one
    # curve and fit once ---
    s_like = dict(B0=d["B0"], B1=d["B1"], tpb=d["tpb"])
    results = []
    manual_tau_short = getattr(args, "tau_short_value", None)
    if fit_method == "irf" and args.fix_tau_short and not args.mono and manual_tau_short is None:
        # IRF mode already computed a robust whole-acquisition reference fit
        # above -- use its tau_short directly rather than deriving one from
        # window 1 (which the tail-fit path does, since it has no such
        # reference fit available before the loop starts).
        tau_short_fixed = g_ref["tau_short"]
        print(f"\n  (using the whole-acquisition reference tau_short = {tau_short_fixed:.2f} ns "
              f"for every window)")
    else:
        tau_short_fixed = manual_tau_short if (args.fix_tau_short and manual_tau_short is not None) else None

    cancel_check = getattr(args, "cancel_check", None)
    for w in range(n_windows):
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelled(f"Cancelled after {w}/{n_windows} windows.")
        sel = slice(boundaries[w], boundaries[w + 1])
        img_w, cube_w = build_group_cube(d, sel)
        img_w, cube_w, _, _, _ = apply_spatial_and_time_binning(
            img_w, cube_w, d_native, args_native, allow_moving_average=False)

        print(f"\n--- window {w+1}/{n_windows}  ({window_descs[w]}) ---")

        if fit_method in ("mean", "mean_irf"):
            if args.mono:
                print(f"  --mono has no effect with fit_method={fit_method}; ignoring.")
            if fit_method == "mean_irf":
                tau_mean, tau_mean_std, n_photons = fit_window_sum_mean_arrival_reconv(
                    cube_w, mask, irf_aligned, d["tpb"] * 1e9, no_shift=no_shift)
            else:
                tau_mean, tau_mean_std, n_photons = fit_window_sum_mean_arrival(
                    cube_w, mask, s_like, offset=getattr(args, "tail_start_offset", None))
            if tau_mean is None:
                print("  no photons in the masked region for this window; skipping.")
                results.append(None)
                continue
            print(f"  {n_photons} photons pooled from {int(mask.sum())} masked {unit}s")
            print(f"  tau_mean = {tau_mean:.2f} ns  (std = {tau_mean_std:.3f} ns)")

            summed = cube_w[mask].sum(axis=0).astype(float)
            if getattr(args, "save_results", True):
                plot_pooled_mean_arrival(summed, d["tpb"], tau_mean, tau_mean_std,
                                         f"Window {w+1}/{n_windows} ({window_descs[w]}): "
                                         f"pooled mean arrival time",
                                         f"{prefix}_window{w+1}of{n_windows}_fit.png", show)

            wprefix = f"{prefix}_window{w+1}of{n_windows}"
            np.savez(f"{wprefix}_maps.npz",
                     photons_per_pixel=img_w, mask=mask, n_photons=n_photons,
                     tau_mean_ns=tau_mean, tau_mean_std_ns=tau_mean_std,
                     window=w, n_windows=n_windows, window_desc=window_descs[w],
                     threshold=threshold, min_neighbors=min_neighbors, bin_size=args.bin,
                     fit_method=fit_method)
            print(f"  saved {wprefix}_maps.npz")
            results.append(dict(val=tau_mean, n_photons=n_photons, tau_mean_std=tau_mean_std))
            continue

        if fit_method == "irf":
            g, n_photons = fit_window_sum_reconv(cube_w, mask, irf_aligned, d["tpb"] * 1e9,
                                                 args.mono, tau_short_fixed,
                                                 tau_short_bounds=getattr(args, "tau_short_bounds", None),
                                                 tau_long_bounds=getattr(args, "tau_long_bounds", None),
                                                 tau_mono_bounds=getattr(args, "tau_mono_bounds", None),
                                                 no_shift=no_shift)
        else:
            g, n_photons = fit_window_sum(cube_w, mask, s_like, args.mono, tau_short_fixed,
                                          offset=getattr(args, "tail_start_offset", None),
                                          tau_short_bounds=getattr(args, "tau_short_bounds", None),
                                          tau_long_bounds=getattr(args, "tau_long_bounds", None),
                                          tau_mono_bounds=getattr(args, "tau_mono_bounds", None))
        if g is None:
            print("  no photons in the masked region for this window; skipping.")
            results.append(None)
            continue
        print(f"  {n_photons} photons pooled from {int(mask.sum())} masked {unit}s")

        if args.mono:
            print(f"  fit: tau = {g['tau']:.2f} ns")
            val = g["tau"]
        else:
            if fit_method == "tail" and args.fix_tau_short and tau_short_fixed is None:
                # First window, no manual value given, tail-fit mode: do the
                # free fit (already just performed above) and pin its own
                # tau_short for every subsequent window.
                tau_short_fixed = g["tau_short"]
                print(f"  (this window's tau_short = {tau_short_fixed:.2f} ns will be fixed for all "
                      f"subsequent windows)")
            elif args.fix_tau_short:
                source = "manually set" if manual_tau_short is not None else (
                    "from the whole-acquisition reference fit" if fit_method == "irf" else "from window 1")
                print(f"  (tau_short fixed to {tau_short_fixed:.2f} ns, {source})")
            print(f"  fit: tau_short = {g['tau_short']:.2f} ns, tau_long = {g['tau_long']:.2f} ns")
            val = g["tau_long"]

        summed = cube_w[mask].sum(axis=0).astype(float)
        symbol = "\u03c4" if args.mono else "\u03c4_long"
        if getattr(args, "save_results", True):
            if fit_method == "irf":
                plot_window_fit_reconv(summed, d["tpb"], g, args.mono,
                                       f"Window {w+1}/{n_windows} ({window_descs[w]}): pooled fit (IRF)",
                                       f"{prefix}_window{w+1}of{n_windows}_fit.png", show)
            else:
                _, _, s_idx, tn = _tail_axis(s_like, summed[None, None, :], offset=getattr(args, "tail_start_offset", None))
                plot_window_fit(summed, s_idx, tn, g, args.mono,
                                f"Window {w+1}/{n_windows} ({window_descs[w]}): pooled fit",
                                f"{prefix}_window{w+1}of{n_windows}_fit.png", show)

        wprefix = f"{prefix}_window{w+1}of{n_windows}"
        if args.mono:
            print(f"  tau_std = {g.get('tau_std', float('nan')):.3f} ns")
            np.savez(f"{wprefix}_maps.npz",
                     photons_per_pixel=img_w, mask=mask, n_photons=n_photons,
                     tau_ns=g["tau"], tau_std_ns=g.get("tau_std", np.nan), model="mono",
                     window=w, n_windows=n_windows,
                     window_desc=window_descs[w], threshold=threshold,
                     min_neighbors=min_neighbors, bin_size=args.bin, fit_method=fit_method)
        else:
            print(f"  tau_short_std = {g.get('tau_short_std', float('nan')):.3f} ns, "
                  f"tau_long_std = {g.get('tau_long_std', float('nan')):.3f} ns")
            np.savez(f"{wprefix}_maps.npz",
                     photons_per_pixel=img_w, mask=mask, n_photons=n_photons,
                     tau_short_ns=g["tau_short"], tau_long_ns=g["tau_long"],
                     tau_short_std_ns=g.get("tau_short_std", np.nan), tau_long_std_ns=g.get("tau_long_std", np.nan),
                     model="bi", window=w, n_windows=n_windows, window_desc=window_descs[w],
                     threshold=threshold, min_neighbors=min_neighbors, bin_size=args.bin,
                     fit_method=fit_method)
        print(f"  saved {wprefix}_maps.npz")

        results.append(dict(val=val, n_photons=n_photons, g=g))

    x = np.array([(w + 0.5) * step_s for w in range(n_windows)])
    symbol = "\u27e8t\u27e9" if fit_method in ("mean", "mean_irf") else ("\u03c4" if args.mono else "\u03c4_long")
    plot_trend(results, window_descs, x, symbol, prefix, show)
    save_trend_npz(results, window_t0, window_t1, x, args.mono, f"{prefix}_trend_summed.npz",
                   fit_method=fit_method)

    n_ok = sum(r is not None for r in results)
    print(f"\nDone. Fitted {n_ok}/{n_windows} windows.")


if __name__ == "__main__":
    main()
