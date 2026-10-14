#!/usr/bin/env python3
"""
Flimera / EzTime FLIM analysis -- Periodic (pixel-wise)

Plugin metadata (NAME/DESCRIPTION/REQUIRES_RAW/IRF_CAPABLE/EXTRA_ARGS) below is
what analyses/_registry.py and the GUI use to list and drive this plugin --
they never import this module by a hardcoded name.

For a sample that changes periodically during the acquisition, splits the
acquisition into N phase groups covering one period and pools photons from
the SAME phase across every repeated cycle, then fits each phase group's
pooled cube pixel-by-pixel (producing a tau map per phase group).

REQUIRES a RAW photon-stream file (per-photon timing is needed to split by phase).

Usage
-----
  python -m analyses.periodic  data.H5 --period 1.0 --groups 5
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
from analyses._common import (
    apply_spatial_and_time_binning,
    load_raw_with_time, assign_phase_groups, build_group_cube, crop_photon_data, process_group,
)

NAME = 'Periodic (pixel-wise)'
DESCRIPTION = 'Split the acquisition into repeating phase groups and fit a tau map per group.'
REQUIRES_RAW = True
IRF_CAPABLE = True
# Extra CLI/GUI settings this plugin needs beyond the common set (used by the
# GUI to build its settings panel and metadata log generically).
EXTRA_ARGS = ['period', 'groups']


def preview_slice(loaded_data, settings):
    """First phase group, for the GUI's mask-preview panel. Reads period/
    groups out of settings["extra"], the same shape every plugin receives
    from the GUI (or from a script building an equivalent dict)."""
    period_s = settings["extra"]["period"]
    n_groups = settings["extra"]["groups"]
    if period_s <= 0 or n_groups <= 0:
        raise ValueError("Period and number of groups must be greater than 0.")
    group_idx = assign_phase_groups(loaded_data["time_ms"], period_s * 1000.0, n_groups)
    sel = group_idx == 0
    img, cube = build_group_cube(loaded_data, sel)
    return img, cube, "first phase group"


def plot_phase_summary(results, n_groups, period_s, prefix, show, mono):
    """Final window: median lifetime vs phase group, across the whole period --
    this is the actual periodic-change trace the whole exercise is for."""
    centers = [(r["group"] + 0.5) * period_s / n_groups for r in results if r is not None]
    medians = [r["median"] for r in results if r is not None]
    lo = [r["median"] - r["iqr"][0] for r in results if r is not None]
    hi = [r["iqr"][1] - r["median"] for r in results if r is not None]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.errorbar(centers, medians, yerr=[lo, hi], fmt="o-", color="#c44",
               capsize=4, lw=1.5, markersize=6)
    ax.set_xlabel(f"Phase within period (s)  [period = {period_s} s]")
    label = "\u03c4 (ns)" if mono else "\u03c4_long (ns)"
    ax.set_ylabel(f"median {label} (error bars = IQR)")
    ax.set_title(f"Lifetime vs. phase  ({n_groups} groups/period)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = f"{prefix}_phase_summary.png"
    fig.savefig(out_png, dpi=130)
    print(f"\nSaved phase summary: {out_png}")
    if show:
        plt.show()
    plt.close(fig)

    out_npz = f"{prefix}_phase_summary.npz"
    n_pixels = [r["n_pixels"] if r is not None else 0 for r in results]
    pixel_offset_path = getattr(args, "pixel_offset", None)
    np.savez(out_npz, phase_center_s=np.array(centers), median=np.array(medians),
             iqr_low=np.array(lo), iqr_high=np.array(hi), n_pixels=np.array(n_pixels),
             period_s=period_s, n_groups=n_groups, mono=mono,
             pixel_offset_calibration=str(Path(pixel_offset_path).resolve())
             if pixel_offset_path else "none")
    print(f"Saved phase summary data: {out_npz}")


def main():
    ap = argparse.ArgumentParser(description="Phase-resolved FLIM analysis for periodic samples")
    ap.add_argument("h5file", help="RAW Flimera .H5 photon-stream file (not an export/.ds3d project)")
    ap.add_argument("--period", type=float, default=None,
                    help="period of the repeating change, in seconds (prompts if omitted)")
    ap.add_argument("--groups", type=int, default=None,
                    help="number of phase groups to split each period into (prompts if omitted)")
    ap.add_argument("--threshold", type=int, default=None,
                    help="min photons/pixel for the mask, applied to EVERY group "
                         "(skips the interactive prompt after group 1's overview)")
    ap.add_argument("--tail-start-offset", type=int, default=None,
                    help=f"bins after the decay peak where the tail fit starts "
                         f"(default {TAIL_START_OFFSET}). No effect with --fit-method irf.")
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
                    help="fix tau_short to each group's own reference value; float only tau_long")
    ap.add_argument("--tau-short-value", type=float, default=None,
                    help="use with --fix-tau-short to supply an exact tau_short value (ns) "
                         "yourself, instead of deriving it per group.")
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
                    help=f"spatial binning factor NxN, applied within each phase group (default {BIN_SIZE})")
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
                         "phase groups: x0,y0,x1,y1 in pixels (x1/y1 exclusive). Speeds up "
                         "every group's bincount/mask/fit steps by skipping pixels outside the box.")
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
    """Everything main() does after parsing arguments -- pulled into its own
    function so a GUI (or any other caller) can build an argparse.Namespace-
    like object directly and call this without going through the command
    line at all. As long as every attribute that would otherwise trigger an
    interactive input() prompt (period, groups, threshold) is set to a
    concrete (non-None) value, this runs straight through with no prompts."""
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
    # photons, ignoring time entirely) just to pick the box from. Cropping
    # here, before any phase-group splitting, means every group built
    # afterward is automatically restricted to the ROI -- faster bincount
    # calls, masking, and fitting throughout the rest of the script.
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

    period_s = args.period
    if period_s is None:
        resp = input("Period of the repeating change, in seconds [1.0]: ").strip()
        period_s = float(resp) if resp else 1.0
    n_groups = args.groups
    if n_groups is None:
        resp = input("Number of phase groups per period [5]: ").strip()
        n_groups = int(resp) if resp else 5

    period_ms = period_s * 1000.0
    n_cycles = d["duration_ms"] / period_ms
    group_width_ms = period_ms / n_groups
    frame_duration_ms = 1000.0 / d["frame_rate_hz"]
    print(f"\nPeriod = {period_s} s over {n_groups} groups "
          f"({group_width_ms:.1f} ms/group) -- ~{n_cycles:.1f} repeated cycles pooled per group")
    if group_width_ms < frame_duration_ms * 3:
        print(f"  NOTE: group width ({group_width_ms:.2f} ms) is only a few frame durations "
              f"({frame_duration_ms:.3f} ms/frame) -- phase timing resolution is limited by "
              f"the camera's frame rate, so very fine phase splitting won't add real precision "
              f"beyond a certain point.")

    group_idx = assign_phase_groups(d["time_ms"], period_ms, n_groups)
    phase_descs = []
    for g in range(n_groups):
        t0, t1 = g * group_width_ms / 1000.0, (g + 1) * group_width_ms / 1000.0
        phase_descs.append(f"{t0:.3f}-{t1:.3f} s (of each {period_s} s cycle)")

    # Build every group's pooled image/cube up front (cheap: just bincount calls).
    group_imgs, group_cubes = {}, {}
    for g in range(n_groups):
        sel = group_idx == g
        img_g, cube_g = build_group_cube(d, sel)
        group_imgs[g], group_cubes[g] = img_g, cube_g
        print(f"  group {g+1}/{n_groups} ({phase_descs[g]}): {int(sel.sum())} photons pooled")

    report_dead_pixels(group_imgs[0])

    # --- show group 1's overview, then choose the threshold (reused for every group) ---
    if getattr(args, "save_overview", True):
        plot_overview(group_imgs[0], f"{prefix}_group1of{n_groups}_overview.png", not args.no_show,
                      title_suffix=f"  (group 1/{n_groups}: {phase_descs[0]})")

    external_mask_path = getattr(args, "external_mask", None)
    external_mask = None
    if external_mask_path:
        print(f"\nLoading external mask from {external_mask_path} (used for every group) ...")
        external_mask = load_mask(external_mask_path)
        threshold = None
        min_neighbors = None
    else:
        if args.threshold is not None:
            threshold = args.threshold
        else:
            nz = group_imgs[0][group_imgs[0] > 0]
            default = DEFAULT_THRESHOLD
            print(f"\nChoose the mask threshold using group 1 ({phase_descs[0]}) as reference.")
            print(f"This SAME threshold will be applied to every other group automatically.")
            print(f"(group 1 per-pixel photons: median {int(np.median(nz))}, "
                  f"90th pct {int(np.percentile(nz, 90))}; suggested default {default})")
            resp = input(f"Minimum photons/pixel [{default}]: ").strip()
            threshold = int(resp) if resp else default

        min_neighbors = args.min_neighbors if args.min_neighbors is not None else MIN_NEIGHBORS

    # --- IRF alignment: done ONCE for the whole file (not per group), since
    # it reflects the optical/detector system's timing, not something that
    # should vary within one measurement. Uses a combined whole-acquisition
    # image/cube (sum of every group) purely for a clean, high-SNR alignment
    # reference -- this does NOT replace process_group's own per-group mask.
    irf_aligned = None
    if fit_method in ("irf", "mean_irf"):
        whole_img = sum(group_imgs.values())
        whole_cube = sum(group_cubes.values())
        # Same spatial smoothing + time binning every group gets (so the IRF
        # alignment/reference fit see the same grid the per-group fits will);
        # d_ref / args_ref carry the post-binning tpb and bin for this block.
        whole_img, whole_cube, d_ref, args_ref, _ = apply_spatial_and_time_binning(
            whole_img, whole_cube, d, args)
        if external_mask is not None:
            if external_mask.shape != whole_img.shape:
                print(f"Error: external mask shape {external_mask.shape} does not match "
                      f"the whole-acquisition image shape {whole_img.shape} -- check that "
                      f"--bin/--roi match how the mask was saved.")
                return
            whole_mask = external_mask
        else:
            # threshold has "per single group" meaning everywhere else in this
            # script (it's applied per-group in process_group) -- comparing it
            # directly against whole_img (summed over all n_groups groups) would
            # make it silently n_groups times more lenient here, giving a
            # reference mask far larger than any individual group's own mask.
            # Divide by n_groups first so the same threshold value selects a
            # comparably-sized, comparable region.
            whole_mask = clean_mask((whole_img / n_groups) > threshold, min_neighbors)
        whole_signal_sum = whole_cube[whole_mask].sum(axis=0).astype(float)
        if no_shift:
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
            # diagnostic plot) -- the moment-based method has no reference
            # tau to seed or fix, only the alignment computed above.
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

    # --- process every group with the same threshold/min_neighbors/bin/model ---
    results = []
    cancel_check = getattr(args, "cancel_check", None)
    for g in range(n_groups):
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelled(f"Cancelled after {g}/{n_groups} groups.")
        r = process_group(g, n_groups, group_imgs[g], group_cubes[g], d,
                          prefix, args, threshold, min_neighbors, phase_descs[g],
                          irf_aligned=irf_aligned, external_mask=external_mask)
        results.append(r)

    if any(r is not None for r in results):
        plot_phase_summary(results, n_groups, period_s, prefix, not args.no_show, args.mono)

    print(f"\nDone. Processed {sum(r is not None for r in results)}/{n_groups} groups.")


if __name__ == "__main__":
    main()
