#!/usr/bin/env python3
"""
Flimera / EzTime FLIM analysis -- Timelapse (pixel-wise)

Plugin metadata (NAME/DESCRIPTION/REQUIRES_RAW/IRF_CAPABLE/EXTRA_ARGS) below is
what analyses/_registry.py and the GUI use to list and drive this plugin --
they never import this module by a hardcoded name.

Splits the WHOLE acquisition into consecutive, non-overlapping time windows
of a fixed duration and fits each window's pixel map separately -- for a
slower, non-repeating change (photobleaching, a one-off drug response, drift),
as opposed to periodic.py's phase-pooling for a genuinely repeating signal.

REQUIRES a RAW photon-stream file (per-photon timing is needed to split by time).

Usage
-----
  python -m analyses.timelapse  data.H5 --step 5
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from flim_core.config import MIN_NEIGHBORS, BIN_SIZE, DEFAULT_THRESHOLD, TAIL_START_OFFSET
from flim_core.masking import bin_cube, prompt_roi, report_dead_pixels, clean_mask, load_mask
from flim_core.plotting import plot_overview
from flim_core.fitting import AnalysisCancelled
from flim_core.irf import (
    load_irf, estimate_shift, shift_array,
    global_reconv_fit, global_reconv_fit_mono,
    plot_irf_diagnostic,
)
from analyses._common import window_boundaries, apply_spatial_and_time_binning, load_raw_with_time, build_group_cube, crop_photon_data, process_group

NAME = 'Timelapse (pixel-wise)'
DESCRIPTION = 'Split the acquisition into fixed-duration time windows and fit a tau map per window.'
REQUIRES_RAW = True
IRF_CAPABLE = True
# Extra CLI/GUI settings this plugin needs beyond the common set (used by the
# GUI to build its settings panel and metadata log generically).
EXTRA_ARGS = ['step']


def preview_slice(loaded_data, settings):
    """First time window, for the GUI's mask-preview panel."""
    step_s = settings["extra"]["step"]
    if step_s <= 0:
        raise ValueError("Step size must be greater than 0.")
    window_idx, n_win = assign_time_windows(loaded_data["time_ms"], step_s * 1000.0,
                                            loaded_data["duration_ms"])
    b = window_boundaries(window_idx, n_win)
    sel = slice(b[0], b[1])   # first window as a slice (view), not a full-length boolean mask
    img, cube = build_group_cube(loaded_data, sel)
    return img, cube, "first time window"


def assign_time_windows(time_ms, step_ms, duration_ms):
    """Which window (0, 1, 2, ...) does each photon's elapsed time fall
    into? Plain sequential chunking -- no wraparound/modulo, unlike the
    periodic script's phase grouping."""
    n_windows = int(np.ceil(duration_ms / step_ms))
    window_idx = np.clip((time_ms // step_ms).astype(int), 0, n_windows - 1)
    return window_idx, n_windows


def plot_timelapse_summary(results, step_s, prefix, show, mono):
    """Final window: median lifetime vs. actual elapsed time, one point per
    time window -- the time-lapse trace itself."""
    centers = [(r["group"] + 0.5) * step_s for r in results if r is not None]
    medians = [r["median"] for r in results if r is not None]
    lo = [r["median"] - r["iqr"][0] for r in results if r is not None]
    hi = [r["iqr"][1] - r["median"] for r in results if r is not None]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.errorbar(centers, medians, yerr=[lo, hi], fmt="o-", color="#3477c4",
               capsize=4, lw=1.5, markersize=6)
    ax.set_xlabel(f"Elapsed time (s)  [window = {step_s} s]")
    label = "\u03c4 (ns)" if mono else "\u03c4_long (ns)"
    ax.set_ylabel(f"median {label} (error bars = IQR)")
    ax.set_title("Lifetime vs. time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = f"{prefix}_timelapse_summary.png"
    fig.savefig(out_png, dpi=130)
    print(f"\nSaved time-lapse summary: {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Simple time-lapse FLIM analysis: fixed-duration sequential windows")
    ap.add_argument("h5file", help="RAW Flimera .H5 photon-stream file (not an export/.ds3d project)")
    ap.add_argument("--step", type=float, default=None,
                    help="time window duration, in seconds (prompts if omitted)")
    ap.add_argument("--threshold", type=int, default=None,
                    help="min photons/pixel for the mask, applied to EVERY window "
                         "(skips the interactive prompt after window 1's overview)")
    ap.add_argument("--tail-start-offset", type=int, default=None,
                    help="bins after the decay peak where the tail fit starts (default 50). "
                         "No effect with --fit-method irf.")
    ap.add_argument("--fit-method", choices=["tail", "irf", "mean", "mean_irf"], default="tail",
                    help="'tail' (default): fit the decay tail only. 'irf': fit the FULL curve "
                         "against a measured instrument response (reconvolution) -- requires --irf. "
                         "'mean': no fit at all -- compute the photon-weighted mean arrival time "
                         "(fast, no bounds; --mono/--fix-tau-short have no effect).")
    ap.add_argument("--irf", default=None,
                    help="IRF measurement file (any supported format), required with --fit-method irf.")
    ap.add_argument("--no-shift", action="store_true",
                    help="fix the IRF/data timing offset at 0 instead of fitting it. Only use this "
                         "when the IRF and data were acquired at matched laser settings in the same "
                         "session -- otherwise a real timing offset (e.g. laser drive-current timing "
                         "walk) can go uncorrected. No effect without --fit-method irf or mean_irf.")
    ap.add_argument("--irf-full", action="store_true",
                    help="use the IRF's full background-subtracted trace (out to --irf-window-end-ns "
                         "after the main peak) instead of windowing to just the main peak. Use this "
                         "when secondary peaks in the IRF trace are believed to be a real part of the "
                         "instrument response (e.g. reproducible laser/SPAD echoes) rather than a "
                         "measurement artifact specific to how the IRF shot was set up. No effect "
                         "without --fit-method irf or mean_irf.")
    ap.add_argument("--irf-window-end-ns", type=float, default=15.0,
                    help="with --irf-full, how far past the IRF's main peak (in ns) to keep -- should "
                         "clear every real secondary peak but stay well before the trace flattens into "
                         "pure background (default 15.0; inspect the raw IRF trace to choose your own).")
    ap.add_argument("--fix-tau-short", action="store_true",
                    help="fix tau_short to each window's own reference value; float only tau_long")
    ap.add_argument("--tau-short-value", type=float, default=None,
                    help="use with --fix-tau-short to supply an exact tau_short value (ns) "
                         "yourself, instead of deriving it per window.")
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
                    help=f"spatial binning factor NxN, applied within each window (default {BIN_SIZE})")
    ap.add_argument("--moving-average", type=int, default=None,
                    help="spatial smoothing by a centered NxN moving average instead of block "
                         "binning (odd N: 3, 5, 7, ...); overrides --bin if both are given. Output "
                         "stays full-resolution but neighbouring pixels share photons (NOT "
                         "independent) -- see flim_core.masking.moving_average_cube.")
    ap.add_argument("--time-bin", type=int, default=1,
                    help="group every N adjacent microtime bins into one (default 1 = none); "
                         "--tail-start-offset is rescaled automatically so the real ns margin "
                         "past the peak is unchanged.")
    ap.add_argument("--roi", type=lambda s: tuple(int(p) for p in s.split(",")), default=None,
                    help="crop to a rectangular region of interest before splitting into "
                         "time windows: x0,y0,x1,y1 in pixels (x1/y1 exclusive). Speeds up "
                         "every window's bincount/mask/fit steps by skipping pixels outside the box.")
    ap.add_argument("--roi-prompt", action="store_true",
                    help="interactively ask for an ROI (shown against the whole-acquisition "
                         "photon count, ignoring time) before anything else. Ignored if --roi is given.")

    ap.add_argument("--external-mask", default=None,
                    help="path to a previously-saved mask (.npy, see flim_core.masking.save_mask) -- "
                         "if given, use this mask directly and skip threshold/min-neighbours entirely. "
                         "Its shape must match the current image AFTER --bin/--roi are applied. Useful "
                         "for analyzing the exact same pixels across acquisitions that need different "
                         "thresholds (e.g. the same sample at different laser powers).")
    ap.add_argument("--no-quality-plots", dest="save_quality_plots", action="store_false", default=True,
                    help="skip saving the fit-quality diagnostic plot (uncertainty map/histogram/noise floor)")
    ap.add_argument("--no-overview-plot", dest="save_overview", action="store_false", default=True,
                    help="skip saving the overview/mask-preview plot")
    ap.add_argument("--no-results-plot", dest="save_results", action="store_false", default=True,
                    help="skip saving the results plot (lifetime map + distribution)")
    ap.add_argument("--no-irf-check-plot", dest="save_irf_check", action="store_false", default=True,
                    help="skip saving the IRF diagnostic plot (only relevant with --fit-method irf)")
    ap.add_argument("--no-show", action="store_true", help="save figures without displaying")
    ap.add_argument("--prefix", default=None, help="optional filename prefix")
    args = ap.parse_args()
    run(args)


def run(args):
    """Everything main() does after parsing arguments -- see periodic.py's
    run() docstring for why this split exists (direct GUI invocation)."""
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
        print(f"  IRF: {irf['source']} | {irf['n_photons']} raw photons | "
              f"FWHM {irf['fwhm_bins']} bins ({irf['fwhm_bins']*irf['tpb']*1e9:.2f} ns) | "
              f"{irf['bg_fraction']*100:.0f}% of raw counts were flat background (removed)")
        if no_shift:
            print("  --no-shift: IRF/data timing offset fixed at 0 (not fitted) -- "
                  "only appropriate if IRF and data were acquired at matched laser settings.")

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

    # --- optional ROI crop, using a quick whole-acquisition preview (all
    # photons, ignoring time) just to pick the box from. Cropping here,
    # before any window splitting, means every window built afterward is
    # automatically restricted to the ROI -- faster throughout the rest.
    roi = args.roi
    if roi is None and args.roi_prompt:
        preview_img = np.bincount(d["y"] * d["W"] + d["x"], minlength=d["H"] * d["W"]) \
                        .reshape(d["H"], d["W"]).astype(float)
        if getattr(args, "save_overview", True):
            plot_overview(preview_img, f"{prefix}_roi_preview_overview.png", not args.no_show,
                          title_suffix="  (whole acquisition, for ROI selection)")
        roi = prompt_roi(d["W"], d["H"])
    if roi is not None:
        d, roi = crop_photon_data(d, roi)

    step_s = args.step
    if step_s is None:
        resp = input(f"Time window step, in seconds (e.g. 5 for 0-5s, 5-10s, ...) [5.0]: ").strip()
        step_s = float(resp) if resp else 5.0
    step_ms = step_s * 1000.0

    window_idx, n_windows = assign_time_windows(d["time_ms"], step_ms, d["duration_ms"])
    print(f"\nStep = {step_s} s over {d['duration_ms']/1000:.3f} s total  ->  {n_windows} windows")

    window_descs = []
    for w in range(n_windows):
        t0 = w * step_s
        t1 = min((w + 1) * step_s, d["duration_ms"] / 1000.0)
        window_descs.append(f"{t0:.3f}-{t1:.3f} s")

    # Build each window's image/cube ON DEMAND, one at a time -- NOT all
    # n_windows of them held in memory simultaneously. That's the real
    # memory cost here, independent of fit_method: each full-resolution
    # cube is H*W*nb float32 (~112 MB for a Flimera sensor's 126x192x1215),
    # so pre-building and storing all of them (the previous approach) needs
    # O(n_windows) memory -- 600 windows -> ~67 GB. The searchsorted fix
    # already applied avoids a large temporary boolean array per window,
    # but that dict of ALL windows' cubes was the bigger cost, and it was
    # already there for every fit_method (not just irf/mean_irf) -- "mean"
    # mode working while mean_irf failed just meant plain "mean" was
    # already close to the memory ceiling, and mean_irf's extra
    # whole-acquisition accumulator (needed for the IRF alignment step
    # below) was enough to push it over. Building window cubes just-in-time
    # instead brings memory down to O(1) -- one or two cubes at a time --
    # at the cost of rebuilding window 0's cube a second time when its turn
    # comes up in the main loop (cheap: build_group_cube is just a couple
    # of bincount calls, not a large allocation on its own).
    boundaries = window_boundaries(window_idx, n_windows)

    def window_cube(w):
        sel = slice(boundaries[w], boundaries[w + 1])
        return build_group_cube(d, sel)

    img0, cube0 = window_cube(0)
    print(f"  window 1/{n_windows} ({window_descs[0]}): {boundaries[1]-boundaries[0]} photons")

    report_dead_pixels(img0)

    # --- show window 1's overview, then choose the threshold (reused for every window) ---
    # (uses the same "group1ofN" filename that process_group will independently
    # produce for window 1 in the loop below, so this is a preview, not a duplicate file)
    if getattr(args, "save_overview", True):
        plot_overview(img0, f"{prefix}_group1of{n_windows}_overview.png", not args.no_show,
                      title_suffix=f"  (window 1/{n_windows}: {window_descs[0]})")

    external_mask_path = getattr(args, "external_mask", None)
    external_mask = None
    if external_mask_path:
        print(f"\nLoading external mask from {external_mask_path} (used for every window) ...")
        external_mask = load_mask(external_mask_path)
        threshold = None
        min_neighbors = None
    else:
        if args.threshold is not None:
            threshold = args.threshold
        else:
            nz = img0[img0 > 0]
            default = DEFAULT_THRESHOLD
            print(f"\nChoose the mask threshold using window 1 ({window_descs[0]}) as reference.")
            print(f"This SAME threshold will be applied to every other window automatically.")
            print(f"(window 1 per-pixel photons: median {int(np.median(nz))}, "
                  f"90th pct {int(np.percentile(nz, 90))}; suggested default {default})")
            resp = input(f"Minimum photons/pixel [{default}]: ").strip()
            threshold = int(resp) if resp else default

        min_neighbors = args.min_neighbors if args.min_neighbors is not None else MIN_NEIGHBORS

    # --- IRF alignment: done ONCE for the whole file (not per window), since
    # it reflects the optical/detector system's timing, not something that
    # should vary within one measurement. Uses a combined whole-acquisition
    # image/cube (sum of every window, accumulated incrementally -- see the
    # note above window_cube() -- rather than built as one big sum() over
    # every window's already-stored cube) purely for a clean, high-SNR
    # alignment reference -- this does NOT replace process_group's own
    # per-window mask.
    irf_aligned = None
    if fit_method in ("irf", "mean_irf"):
        whole_img = img0.astype(np.float64)
        whole_cube = cube0.astype(np.float64)  # float64 accumulator: avoids
                                                # precision loss summing
                                                # hundreds of float32 cubes
        for w in range(1, n_windows):
            img_w, cube_w = window_cube(w)
            whole_img += img_w
            whole_cube += cube_w
        # Same spatial smoothing + time binning every group gets (so the IRF
        # alignment/reference fit see the same grid the per-group fits will);
        # d_ref / args_ref carry the post-binning tpb and bin for this block.
        whole_img, whole_cube, d_ref, args_ref, _ = apply_spatial_and_time_binning(
            whole_img, whole_cube, d, args)
        # threshold has "per single window" meaning everywhere else in this
        # script (it's applied per-window in process_group) -- comparing it
        # directly against whole_img (summed over all n_windows windows)
        # would make it silently n_windows times more lenient here, giving
        # a reference mask far larger than any individual window's own mask
        # (observed in practice: ~11000 pixels here vs ~500-850 per window).
        # Divide by n_windows first so the same threshold value selects a
        if external_mask is not None:
            if external_mask.shape != whole_img.shape:
                print(f"Error: external mask shape {external_mask.shape} does not match "
                      f"the whole-acquisition image shape {whole_img.shape} -- check that "
                      f"--bin/--roi match how the mask was saved.")
                return
            whole_mask = external_mask
        else:
            # threshold has "per single window" meaning everywhere else in this
            # script (it's applied per-window in process_group) -- comparing it
            # directly against whole_img (summed over all n_windows windows)
            # would make it silently n_windows times more lenient here, giving
            # a reference mask far larger than any individual window's own mask
            # (observed in practice: ~11000 pixels here vs ~500-850 per window).
            # Divide by n_windows first so the same threshold value selects a
            # comparably-sized, comparable region.
            whole_mask = clean_mask((whole_img / n_windows) > threshold, min_neighbors)
        whole_signal_sum = whole_cube[whole_mask].sum(axis=0).astype(float)
        if no_shift:
            # Skip the external pre-alignment entirely -- with the internal
            # fit model also fixing shift at 0 (passed to process_group
            # below), pre-shifting the kernel here would just reintroduce
            # an offset through the back door.
            shift = 0.0
            irf_aligned = irf["irf_norm"]
        else:
            shift = estimate_shift(whole_signal_sum, irf["irf_norm"])
            irf_aligned = shift_array(irf["irf_norm"], shift)
        print(f"\n  IRF-to-data alignment (whole acquisition, masked region): "
              f"shift = {shift} bins ({shift*d_ref['tpb']*1e9:.3f} ns)"
              f"{' (fixed, not fitted)' if no_shift else ''}")
        if fit_method == "irf":
            # mean_irf doesn't need a reference biexponential fit (or its
            # diagnostic plot) -- only the alignment computed above.
            if args.mono:
                g_ref = global_reconv_fit_mono(whole_signal_sum, irf_aligned, d_ref["tpb"] * 1e9, no_shift=no_shift)
                print(f"  reference reconvolution fit (whole acquisition): tau = {g_ref['tau']:.2f} ns")
            else:
                g_ref = global_reconv_fit(whole_signal_sum, irf_aligned, d_ref["tpb"] * 1e9, no_shift=no_shift)
                print(f"  reference reconvolution fit (whole acquisition): "
                      f"tau_short = {g_ref['tau_short']:.2f} ns, tau_long = {g_ref['tau_long']:.2f} ns")
            if getattr(args, "save_irf_check", True):
                plot_irf_diagnostic(irf, whole_signal_sum, shift, g_ref, d_ref["tpb"],
                                    f"{prefix}_irf_check.png", not args.no_show,
                                    n_signal=int(whole_mask.sum()),
                                    unit="super-pixels" if (args_ref.bin and args_ref.bin > 1) else "pixels")
        del whole_img, whole_cube   # done with the accumulator; free it before the main loop

    # --- process every window with the same threshold/min_neighbors/bin/model ---
    # (reuses process_group from the periodic script -- it's generic over
    # "a group of photons and a label", it doesn't care whether that group
    # came from phase-wrapping or plain sequential chunking)
    results = []
    cancel_check = getattr(args, "cancel_check", None)
    for w in range(n_windows):
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelled(f"Cancelled after {w}/{n_windows} windows.")
        img_w, cube_w = (img0, cube0) if w == 0 else window_cube(w)
        r = process_group(w, n_windows, img_w, cube_w, d,
                          prefix, args, threshold, min_neighbors, window_descs[w],
                          irf_aligned=irf_aligned, external_mask=external_mask)
        results.append(r)
        del img_w, cube_w   # let GC reclaim this window's cube before the next one is built

    if any(r is not None for r in results):
        plot_timelapse_summary(results, step_s, prefix, not args.no_show, args.mono)

    print(f"\nDone. Processed {sum(r is not None for r in results)}/{n_windows} windows.")


if __name__ == "__main__":
    main()
