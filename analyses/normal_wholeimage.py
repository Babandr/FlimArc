#!/usr/bin/env python3
"""
Flimera / EzTime FLIM analysis -- Normal (whole image)

Same masking workflow as analyses/normal.py -- binning, threshold, mono/bi
model choice -- but pools every masked pixel's decay histogram into ONE
combined curve and fits that curve ONCE, giving a single tau (or
tau_short/tau_long) value for the whole measurement instead of a per-pixel
map.

Works on BOTH raw photon-stream files AND exported/.ds3d histogram files,
same as analyses/normal.py -- there is no time/phase splitting here.

IRF reconvolution fitting (fit_whole_image_reconv, below) mirrors the same
pattern as analyses/normal.py's per-pixel version: load the IRF, align it
to the pooled masked curve once via estimate_shift, then fit (or compute
the IRF-corrected mean arrival time of) that one pooled curve.
"""

import argparse
import copy
from pathlib import Path

import numpy as np

from flim_core.config import MIN_NEIGHBORS, BIN_SIZE, DEFAULT_THRESHOLD, TAIL_START_OFFSET
from flim_core.io import load_flim_data, photon_count_map, build_decay_cube
from flim_core.masking import (
    clean_mask, bin_cube, bin_time_cube, moving_average_cube, crop_to_roi, prompt_roi,
    report_dead_pixels, load_mask,
)
from flim_core.fitting import _tail_axis, global_fit
from flim_core.irf import load_irf, estimate_shift, shift_array, plot_irf_diagnostic
from flim_core.plotting import plot_overview_and_mask
from analyses._common import (
    global_fit_fixed_short, plot_window_fit, build_group_cube,
    fit_window_sum_mean_arrival, plot_pooled_mean_arrival,
    fit_window_sum_reconv, fit_window_sum_mean_arrival_reconv, plot_window_fit_reconv,
)

NAME = "Normal (whole image)"
DESCRIPTION = "Pool every masked pixel into one curve and fit a single tau for the whole image."
REQUIRES_RAW = False   # works on raw, exported, and .ds3d project files alike
IRF_CAPABLE = True
EXTRA_ARGS = []


def preview_slice(loaded_data, settings):
    """Same rationale as analyses/normal.py's preview_slice: no windowing
    here, so the preview is always the whole acquisition."""
    if "cube" in loaded_data:
        return loaded_data["img"], loaded_data["cube"], "whole acquisition"
    d = loaded_data
    img, cube = build_group_cube(d, slice(None))
    return img, cube, "whole acquisition"


def fit_whole_image(cube, mask, s, mono, tau_short_fixed=None, offset=None,
                    tau_short_bounds=None, tau_long_bounds=None, tau_mono_bounds=None):
    """Pool every masked pixel's decay histogram into one combined curve,
    then fit that single curve (mono, free biexponential, or biexponential
    with tau_short pinned). Returns a result dict (from global_fit /
    global_fit_fixed_short) plus the photon count used, or (None, 0) if the
    mask is empty."""
    if mask.sum() == 0:
        return None, 0
    summed = cube[mask].sum(axis=0).astype(float)
    n_photons = int(summed.sum())
    if n_photons == 0:
        return None, 0
    _, peak_bin, s_idx, tn = _tail_axis(s, cube[mask].sum(axis=0)[None, None, :], offset=offset)
    if mono:
        g = global_fit(summed, s, s_idx, tn, mono=True, tau_mono_bounds=tau_mono_bounds)
    elif tau_short_fixed is not None:
        g = global_fit_fixed_short(summed, s, s_idx, tn, tau_short_fixed, tau_long_bounds=tau_long_bounds)
    else:
        g = global_fit(summed, s, s_idx, tn, mono=False,
                       tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds)
    return g, n_photons


def fit_whole_image_reconv(cube, mask, irf_aligned, tpb_ns, mono, tau_short_fixed=None,
                           tau_short_bounds=None, tau_long_bounds=None, tau_mono_bounds=None, no_shift=False):
    """Reconvolution equivalent of fit_whole_image: thin wrapper around
    _common.fit_window_sum_reconv, kept here (rather than calling that
    directly from process_file) so this module's own call sites read the
    same way for both the tail and reconvolution paths."""
    return fit_window_sum_reconv(cube, mask, irf_aligned, tpb_ns, mono, tau_short_fixed,
                                 tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                                 tau_mono_bounds=tau_mono_bounds, no_shift=no_shift)


def process_file(h5file, args):
    stem = Path(h5file).stem
    prefix = f"{args.prefix}_{stem}" if args.prefix else stem
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
                       time_bin=(getattr(args, "time_bin", None) or 1))
        print(f"  IRF: {irf['source']} | {irf['n_photons']} raw photons | "
              f"FWHM {irf['fwhm_bins']} bins ({irf['fwhm_bins']*irf['tpb']*1e9:.2f} ns) | "
              f"{irf['bg_fraction']*100:.0f}% of raw counts were flat background (removed)")
        if no_shift:
            print("  --no-shift: IRF/data timing offset fixed at 0 (not fitted) -- "
                  "only appropriate if IRF and data were acquired at matched laser settings.")

    print(f"\n=== {h5file} ===")
    print("Loading data ...")
    s = load_flim_data(h5file)
    print(f"  format: {s['source']} | {s['n_photons']} photons | sensor {s['W']}x{s['H']} | "
          f"bins {s['B0']}-{s['B1']} | {s['tpb']*1e12:.1f} ps/bin")

    img = photon_count_map(s)
    img_orig = img
    nz = img[img > 0]
    print(f"  per-pixel photons: median {int(np.median(nz))}, mean {nz.mean():.1f}, "
          f"90th pct {int(np.percentile(nz, 90))}, max {int(nz.max())}")
    report_dead_pixels(img)

    cube = build_decay_cube(s)
    roi = args.roi
    if roi is None and args.roi_prompt:
        roi = prompt_roi(s["W"], s["H"])
    if roi is not None:
        img, cube, roi = crop_to_roi(img, cube, roi)
        img_orig = img
        nz = img[img > 0]
        print(f"  after ROI crop: median {int(np.median(nz))}, "
              f"90th pct {int(np.percentile(nz, 90))} photons/pixel")

    # Spatial smoothing: block binning and moving-average are mutually
    # exclusive alternatives -- see analyses/normal.py's process_file for
    # the full reasoning (mirrored here identically).
    # Moving average is deliberately NOT applied in whole-image mode: this
    # plugin pools every masked pixel into one curve, and moving-averaged
    # pixels share photons (each real photon appears in up to window**2
    # output pixels), so pooling them counts every photon many times over --
    # verified on real data: a 5x5 window turned 258k real photons into
    # 30M "pooled" ones and shrank the reported fit std by ~35%, a false
    # precision gain. Pooling everything already does the job smoothing is
    # meant for, so it's skipped here rather than silently inflating counts.
    moving_avg = getattr(args, "moving_average", None)
    if moving_avg and moving_avg > 1:
        print(f"  WARNING: --moving-average {moving_avg} is ignored in whole-image mode "
              f"(pooled pixels would count each photon up to {moving_avg**2}x, overstating precision); "
              f"using --bin {args.bin} instead.")
        moving_avg = None
    binning_used = bool(args.bin and args.bin > 1)
    if binning_used:
        img, cube = bin_cube(img, cube, args.bin)
        nz_b = img[img > 0]
        print(f"  after {args.bin}x{args.bin} binning: median {int(np.median(nz_b))}, "
              f"90th pct {int(np.percentile(nz_b, 90))} photons/super-pixel")
    plot_bin_size = args.bin

    # Time binning -- see analyses/normal.py's process_file for the full
    # reasoning (mirrored here identically), including the tail-offset
    # rescale right below.
    time_bin = getattr(args, "time_bin", None) or 1
    if time_bin > 1:
        cube, tpb_binned = bin_time_cube(cube, time_bin, s["tpb"])
        s = dict(s)
        s["tpb"] = tpb_binned
        s["B1"] = s["B0"] + cube.shape[2] - 1
    raw_offset = args.tail_start_offset if getattr(args, "tail_start_offset", None) is not None \
        else TAIL_START_OFFSET
    # Work on a copy so rescaling doesn't compound when one args object is reused
    # across several files (e.g. a shell glob on the command line).
    args = copy.copy(args)
    args.tail_start_offset = max(1, round(raw_offset / time_bin)) if time_bin > 1 else raw_offset

    external_mask_path = getattr(args, "external_mask", None)
    if external_mask_path:
        print(f"\nLoading external mask from {external_mask_path} ...")
        mask = load_mask(external_mask_path)
        if mask.shape != img.shape:
            print(f"Error: external mask shape {mask.shape} does not match the current "
                  f"image shape {img.shape} -- check that --bin/--roi match how the mask "
                  f"was saved (masks are resolution-specific).")
            return
        threshold = None
        min_neighbors = None
        removed = np.zeros_like(mask)
        unit = "super-pixels" if binning_used else "pixels"
        print(f"  external mask: {int(mask.sum())} {unit} selected")
    else:
        if args.threshold is not None:
            threshold = args.threshold
        else:
            unit = "super-pixel" if binning_used else "pixel"
            if binning_used:
                default = int(np.percentile(nz_b, 90))
            else:
                default = DEFAULT_THRESHOLD
            print(f"\nChoose the mask: minimum photons per {unit} to include.")
            print(f"(higher = fewer but better-fit {unit}s; suggested {default})")
            resp = input(f"Minimum photons/{unit} [{default}]: ").strip()
            threshold = int(resp) if resp else default

        min_neighbors = args.min_neighbors if args.min_neighbors is not None else MIN_NEIGHBORS
        raw_mask = img > threshold
        mask = clean_mask(raw_mask, min_neighbors)
        removed = raw_mask & ~mask
        unit = "super-pixels" if binning_used else "pixels"
        print(f"\nMask threshold = {threshold}  ->  {int(raw_mask.sum())} {unit} above threshold")
        print(f"Spatial filter (need \u2265 {min_neighbors} selected neighbours) removed "
              f"{int(removed.sum())} isolated {unit[:-1]}(s)  ->  {int(mask.sum())} kept")

    if mask.sum() == 0:
        print(f"No {unit} remain; lower the threshold or the neighbour requirement.")
        return

    # --- overview + mask preview, in one file (matches the GUI's display) ---
    if getattr(args, "save_overview", True):
        plot_overview_and_mask(img_orig, img, mask, removed, threshold, min_neighbors,
                               plot_bin_size, f"{prefix}_overview.png", show)

    # --- pool the masked region into one curve, fit (or compute mean arrival time) once ---
    s_like = dict(B0=s["B0"], B1=s["B1"], tpb=s["tpb"])
    manual_tau_short = getattr(args, "tau_short_value", None)
    tau_short_fixed = manual_tau_short if (args.fix_tau_short and manual_tau_short is not None) else None
    offset = getattr(args, "tail_start_offset", None)

    irf_aligned = None
    if fit_method in ("irf", "mean_irf"):
        signal_sum = cube[mask].sum(axis=0).astype(float)
        if no_shift:
            # Skip the external pre-alignment entirely -- with the internal
            # fit model also fixing shift at 0 (passed below), pre-shifting
            # the kernel here would just reintroduce an offset through the
            # back door.
            shift = 0.0
            irf_aligned = irf["irf_norm"]
        else:
            shift = estimate_shift(signal_sum, irf["irf_norm"])
            irf_aligned = shift_array(irf["irf_norm"], shift)
        print(f"\n  IRF-to-data alignment: shift = {shift} bins ({shift*s['tpb']*1e9:.3f} ns)"
              f"{' (fixed, not fitted)' if no_shift else ''}")

    if fit_method == "mean_irf":
        if args.mono:
            print("  --mono has no effect with fit_method=mean_irf (there's only one lifetime to compute); ignoring.")
        if args.fix_tau_short:
            print("  --fix-tau-short has no effect with fit_method=mean_irf; ignoring.")
        tau_mean, tau_mean_std, n_photons = fit_window_sum_mean_arrival_reconv(
            cube, mask, irf_aligned, s["tpb"] * 1e9, no_shift=no_shift)
        if tau_mean is None:
            print("No photons in the masked region; nothing to compute.")
            return
        print(f"\n{n_photons} photons pooled from {int(mask.sum())} masked {unit}")
        print(f"  tau_mean (IRF-corrected) = {tau_mean:.2f} ns  (std = {tau_mean_std:.3f} ns)")

        summed = cube[mask].sum(axis=0).astype(float)
        if getattr(args, "save_results", True):
            plot_pooled_mean_arrival(summed, s["tpb"], tau_mean, tau_mean_std,
                                     f"Whole image ({int(mask.sum())} masked {unit}): "
                                     f"pooled IRF-corrected mean arrival time",
                                     f"{prefix}_fit.png", show)

        np.savez(f"{prefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, n_photons=n_photons,
                 tau_mean_ns=tau_mean, tau_mean_std_ns=tau_mean_std, irf_shift=shift,
                 threshold=threshold, min_neighbors=min_neighbors, bin_size=plot_bin_size, fit_method=fit_method,
                 moving_average=(moving_avg or 1), time_bin=time_bin)
        print(f"  saved {prefix}_maps.npz")
        print(f"Done with {h5file}.")
        return

    if fit_method == "mean":
        if args.mono:
            print("  --mono has no effect with fit_method=mean (there's only one lifetime to compute); ignoring.")
        if args.fix_tau_short:
            print("  --fix-tau-short has no effect with fit_method=mean; ignoring.")
        tau_mean, tau_mean_std, n_photons = fit_window_sum_mean_arrival(cube, mask, s_like, offset=offset)
        if tau_mean is None:
            print("No photons in the masked region; nothing to compute.")
            return
        print(f"\n{n_photons} photons pooled from {int(mask.sum())} masked {unit}")
        print(f"  tau_mean = {tau_mean:.2f} ns  (std = {tau_mean_std:.3f} ns)")

        summed = cube[mask].sum(axis=0).astype(float)
        if getattr(args, "save_results", True):
            plot_pooled_mean_arrival(summed, s["tpb"], tau_mean, tau_mean_std,
                                     f"Whole image ({int(mask.sum())} masked {unit}): pooled mean arrival time",
                                     f"{prefix}_fit.png", show)

        np.savez(f"{prefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, n_photons=n_photons,
                 tau_mean_ns=tau_mean, tau_mean_std_ns=tau_mean_std,
                 threshold=threshold, min_neighbors=min_neighbors, bin_size=plot_bin_size, fit_method=fit_method,
                 moving_average=(moving_avg or 1), time_bin=time_bin)
        print(f"  saved {prefix}_maps.npz")
        print(f"Done with {h5file}.")
        return

    if fit_method == "irf":
        g, n_photons = fit_whole_image_reconv(cube, mask, irf_aligned, s["tpb"] * 1e9, args.mono, tau_short_fixed,
                                              tau_short_bounds=getattr(args, "tau_short_bounds", None),
                                              tau_long_bounds=getattr(args, "tau_long_bounds", None),
                                              tau_mono_bounds=getattr(args, "tau_mono_bounds", None),
                                              no_shift=no_shift)
        if g is None:
            print("No photons in the masked region; nothing to fit.")
            return
        print(f"\n{n_photons} photons pooled from {int(mask.sum())} masked {unit}")

        if args.mono:
            if args.fix_tau_short:
                print("  --fix-tau-short has no effect with --mono (only one lifetime to fit); ignoring.")
            print(f"  fit: tau = {g['tau']:.2f} ns")
        else:
            if args.fix_tau_short and tau_short_fixed is None:
                print(f"  (only one fit for the whole image -- --fix-tau-short has no further effect here)")
            print(f"  fit: tau_short = {g['tau_short']:.2f} ns, tau_long = {g['tau_long']:.2f} ns")

        summed = cube[mask].sum(axis=0).astype(float)
        if getattr(args, "save_results", True):
            plot_window_fit_reconv(summed, s["tpb"], g, args.mono,
                                   f"Whole image ({int(mask.sum())} masked {unit}): pooled reconvolution fit",
                                   f"{prefix}_fit.png", show)
        if getattr(args, "save_irf_check", True):
            plot_irf_diagnostic(irf, summed, shift, g, s["tpb"], f"{prefix}_irf_check.png", show,
                                n_signal=int(mask.sum()), unit=unit)

        if args.mono:
            print(f"  tau_std = {g.get('tau_std', float('nan')):.3f} ns  (n_photons = {n_photons})")
            np.savez(f"{prefix}_maps.npz",
                     photons_per_pixel=img, mask=mask, n_photons=n_photons,
                     tau_ns=g["tau"], tau_std_ns=g.get("tau_std", np.nan), model="mono", irf_shift=shift,
                     threshold=threshold, min_neighbors=min_neighbors, bin_size=plot_bin_size, fit_method=fit_method,
                 moving_average=(moving_avg or 1), time_bin=time_bin)
        else:
            print(f"  tau_short_std = {g.get('tau_short_std', float('nan')):.3f} ns, "
                  f"tau_long_std = {g.get('tau_long_std', float('nan')):.3f} ns  (n_photons = {n_photons})")
            np.savez(f"{prefix}_maps.npz",
                     photons_per_pixel=img, mask=mask, n_photons=n_photons,
                     tau_short_ns=g["tau_short"], tau_long_ns=g["tau_long"],
                     tau_short_std_ns=g.get("tau_short_std", np.nan), tau_long_std_ns=g.get("tau_long_std", np.nan),
                     model="bi", irf_shift=shift,
                     threshold=threshold, min_neighbors=min_neighbors, bin_size=plot_bin_size, fit_method=fit_method,
                 moving_average=(moving_avg or 1), time_bin=time_bin)
        print(f"  saved {prefix}_maps.npz")
        print(f"Done with {h5file}.")
        return

    g, n_photons = fit_whole_image(cube, mask, s_like, args.mono, tau_short_fixed, offset=offset,
                                   tau_short_bounds=getattr(args, "tau_short_bounds", None),
                                   tau_long_bounds=getattr(args, "tau_long_bounds", None),
                                   tau_mono_bounds=getattr(args, "tau_mono_bounds", None))
    if g is None:
        print("No photons in the masked region; nothing to fit.")
        return
    print(f"\n{n_photons} photons pooled from {int(mask.sum())} masked {unit}")

    if args.mono:
        if args.fix_tau_short:
            print("  --fix-tau-short has no effect with --mono (only one lifetime to fit); ignoring.")
        print(f"  fit: tau = {g['tau']:.2f} ns")
    else:
        if args.fix_tau_short and tau_short_fixed is None:
            # No manual value given: use this same (only) fit's own tau_short.
            # There's only one fit here, so "fixing" it doesn't change
            # anything numerically, but we still honor the flag for
            # consistency with the other scripts and note it in the log.
            print(f"  (only one fit for the whole image -- --fix-tau-short has no further effect here)")
        print(f"  fit: tau_short = {g['tau_short']:.2f} ns, tau_long = {g['tau_long']:.2f} ns")

    # --- fit diagnostic: pooled curve + fitted model, log scale ---
    summed = cube[mask].sum(axis=0).astype(float)
    _, _, s_idx, tn = _tail_axis(s_like, summed[None, None, :], offset=offset)
    if getattr(args, "save_results", True):
        plot_window_fit(summed, s_idx, tn, g, args.mono,
                        f"Whole image ({int(mask.sum())} masked {unit}): pooled fit",
                        f"{prefix}_fit.png", show)

    # --- save ---
    if args.mono:
        print(f"  tau_std = {g.get('tau_std', float('nan')):.3f} ns  (n_photons = {n_photons})")
        np.savez(f"{prefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, n_photons=n_photons,
                 tau_ns=g["tau"], tau_std_ns=g.get("tau_std", np.nan), model="mono",
                 threshold=threshold, min_neighbors=min_neighbors, bin_size=plot_bin_size,
                 moving_average=(moving_avg or 1), time_bin=time_bin)
    else:
        print(f"  tau_short_std = {g.get('tau_short_std', float('nan')):.3f} ns, "
              f"tau_long_std = {g.get('tau_long_std', float('nan')):.3f} ns  (n_photons = {n_photons})")
        np.savez(f"{prefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, n_photons=n_photons,
                 tau_short_ns=g["tau_short"], tau_long_ns=g["tau_long"],
                 tau_short_std_ns=g.get("tau_short_std", np.nan), tau_long_std_ns=g.get("tau_long_std", np.nan),
                 model="bi", threshold=threshold, min_neighbors=min_neighbors, bin_size=plot_bin_size,
                 moving_average=(moving_avg or 1), time_bin=time_bin)
    print(f"  saved {prefix}_maps.npz")
    print(f"Done with {h5file}.")


def main():
    ap = argparse.ArgumentParser(description="Normal FLIM analysis, whole-image (pooled) version")
    ap.add_argument("h5file", nargs="+",
                    help="one or more data files (raw .H5, exported, or .ds3d project)")
    ap.add_argument("--threshold", type=int, default=None,
                    help="min photons/pixel for the mask (skips the interactive prompt)")
    ap.add_argument("--fix-tau-short", action="store_true",
                    help="has no numerical effect here (only one pooled fit exists); kept for "
                         "consistency with the other scripts. No effect with --mono.")
    ap.add_argument("--tau-short-value", type=float, default=None,
                    help="use with --fix-tau-short to supply an exact tau_short value (ns) "
                         "yourself, instead of letting it float freely.")
    ap.add_argument("--tail-start-offset", type=int, default=None,
                    help=f"bins after the decay peak where the tail fit starts (default {TAIL_START_OFFSET}).")
    ap.add_argument("--fit-method", choices=["tail", "irf", "mean", "mean_irf"], default="tail",
                    help="'tail' (default): fit the pooled decay tail. 'irf': fit the FULL pooled curve "
                         "against a measured instrument response (reconvolution) -- requires --irf. "
                         "'mean': no fit -- compute the photon-weighted mean arrival time over the "
                         "pooled tail (fast, no bounds; --mono/--fix-tau-short have no effect). "
                         "'mean_irf': same, but IRF-corrected and computed over the FULL pooled curve "
                         "instead of just the tail -- requires --irf; measures a different (generally "
                         "lower, since it includes any fast component) quantity than plain 'mean'.")
    ap.add_argument("--irf", default=None,
                    help="IRF measurement file (any supported format), required with "
                         "--fit-method irf or mean_irf.")
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
                    help=f"spatial binning factor NxN (default {BIN_SIZE}). Ignored if "
                         "--moving-average is also given.")
    ap.add_argument("--moving-average", type=int, default=None,
                    help="spatial smoothing by a centered NxN moving average instead of block "
                         "binning (must be odd: 3, 5, 7, ...); overrides --bin if both are given. "
                         "See analyses/normal.py's --moving-average help for the full explanation.")
    ap.add_argument("--time-bin", type=int, default=1,
                    help="group every N adjacent microtime bins into one (default 1 = no binning). "
                         "See analyses/normal.py's --time-bin help for the full explanation.")
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
                    help="skip saving the results/pooled-fit plot")
    ap.add_argument("--no-irf-check-plot", dest="save_irf_check", action="store_false", default=True,
                    help="skip saving the IRF alignment/fit diagnostic plot (--fit-method irf only)")
    ap.add_argument("--no-show", action="store_true", help="save figures without displaying")
    ap.add_argument("--prefix", default=None, help="optional filename prefix")
    args = ap.parse_args()

    for h5file in args.h5file:
        process_file(h5file, args)

    if len(args.h5file) > 1:
        print(f"\nAll {len(args.h5file)} files done.")


def run(args):
    """GUI/plugin-interface entry point -- this plugin takes one file at a
    time via process_file, so run() is a thin single-file wrapper."""
    process_file(args.h5file, args)


if __name__ == "__main__":
    main()
