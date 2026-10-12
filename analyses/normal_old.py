#!/usr/bin/env python3
"""
Flimera / EzTime FLIM analysis -- Normal (pixel-wise)

Reads a Flimera FLIM HDF5 file (auto-detects RAW photon-stream, flat
EXPORTED EzTime FLIM, and native EzTime "project" files, e.g. .ds3d) and
fits a per-pixel lifetime map.

Two fit methods are available, chosen via --fit-method / EXTRA_ARGS:
  - "tail" (default): fit only the decay tail, a few bins past the peak.
  - "irf": fit the FULL decay curve against a measured instrument response
    (reconvolution) -- needs a separate --irf measurement file, but can
    resolve lifetimes shorter than the tail-fit method can.

Both paths share the same load -> mask -> fit -> plot -> save pipeline;
only the fitting step itself differs. Every other plugin in analyses/ that
supports both methods (periodic, timelapse, and their whole-image variants)
follows this same fit_method dispatch, so the GUI never needs to know that
"IRF fitting" is a different code path from "tail fitting" -- it just sets
args.fit_method and calls this plugin's run(), same as any other plugin.
"""

import argparse
import glob
from pathlib import Path

import numpy as np

from flim_core.config import MIN_NEIGHBORS, BIN_SIZE, DEFAULT_THRESHOLD, TAIL_START_OFFSET
from flim_core.io import load_flim_data, photon_count_map, build_decay_cube
from flim_core.masking import clean_mask, bin_cube, crop_to_roi, prompt_roi, report_dead_pixels, load_mask
from flim_core.fitting import _tail_axis, global_fit, fit_pixels_mono, fit_pixels, mean_arrival_time
from flim_core.plotting import plot_overview_and_mask, plot_results, plot_fit_quality
from flim_core.irf import (
    load_irf, estimate_shift, shift_array,
    global_reconv_fit, global_reconv_fit_mono,
    fit_pixels_reconv, fit_pixels_reconv_mono,
    plot_irf_diagnostic,
)
from analyses._common import build_group_cube

NAME = "Normal (pixel-wise)"
DESCRIPTION = "Fit a per-pixel lifetime map for a single acquisition (tail fit or IRF reconvolution)."
REQUIRES_RAW = False   # works on raw, exported, and .ds3d project files alike
IRF_CAPABLE = True
EXTRA_ARGS = []


def preview_slice(loaded_data, settings):
    """The GUI's mask-preview panel needs one representative (img, cube,
    description) triple to build a preview mask against before any
    analysis runs -- for this plugin there's no windowing/splitting, so
    it's always the whole acquisition, whether loaded_data came from a
    histogram file (already has 'cube') or a raw photon stream (needs
    build_group_cube first)."""
    if "cube" in loaded_data:
        return loaded_data["img"], loaded_data["cube"], "whole acquisition"
    d = loaded_data
    img, cube = build_group_cube(d, slice(None))
    return img, cube, "whole acquisition"


def process_file(h5file, args):
    """Run the full load -> mask -> fit -> plot -> save pipeline for one
    file, using either a tail fit or an IRF reconvolution fit depending on
    args.fit_method ("tail", the default, or "irf"). Output filenames
    always embed the source file's own name, so running this over several
    files never overwrites one file's results with another's."""
    fit_method = getattr(args, "fit_method", "tail")
    irf = None
    if fit_method == "irf":
        irf_path = getattr(args, "irf", None)
        if not irf_path:
            print("Error: --fit-method irf requires --irf <file> (an IRF measurement file).")
            return
        print(f"Loading IRF from {irf_path} ...")
        irf = load_irf(irf_path)
        print(f"  IRF: {irf['source']} | {irf['n_photons']} raw photons | "
              f"FWHM {irf['fwhm_bins']} bins ({irf['fwhm_bins']*irf['tpb']*1e9:.2f} ns) | "
              f"{irf['bg_fraction']*100:.0f}% of raw counts were flat background (removed)")

    stem = Path(h5file).stem
    prefix = f"{args.prefix}_{stem}" if args.prefix else stem
    show = not args.no_show

    print(f"\n=== {h5file} ===")
    print("Loading data ...")
    s = load_flim_data(h5file)
    print(f"  format: {s['source']} | {s['n_photons']} photons | sensor {s['W']}x{s['H']} | "
          f"bins {s['B0']}-{s['B1']} | {s['tpb']*1e12:.1f} ps/bin")

    if irf is not None and abs(s["tpb"] - irf["tpb"]) > 1e-15:
        print(f"  WARNING: data TimePerBin ({s['tpb']:.3e} s) differs from IRF's "
              f"({irf['tpb']:.3e} s) -- these were likely acquired under different "
              f"timing settings and this IRF may not be valid for this data.")

    img = photon_count_map(s)
    img_orig = img   # kept for the overview panel, before any ROI-crop/binning reassigns img
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

    binning_used = bool(args.bin and args.bin > 1)
    if binning_used:
        img, cube = bin_cube(img, cube, args.bin)
        nz_b = img[img > 0]
        print(f"  after {args.bin}x{args.bin} binning: median {int(np.median(nz_b))}, "
              f"90th pct {int(np.percentile(nz_b, 90))} photons/super-pixel")

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
            default = int(np.percentile(nz_b, 90)) if binning_used else DEFAULT_THRESHOLD
            print(f"\nChoose the mask: minimum photons per {unit} to include in the fit.")
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

    if getattr(args, "save_overview", True):
        plot_overview_and_mask(img_orig, img, mask, removed, threshold, min_neighbors,
                               args.bin, f"{prefix}_overview.png", show)

    manual_tau_short = getattr(args, "tau_short_value", None)
    tau_short_bounds = getattr(args, "tau_short_bounds", None)
    tau_long_bounds = getattr(args, "tau_long_bounds", None)
    tau_mono_bounds = getattr(args, "tau_mono_bounds", None)
    cancel_check = getattr(args, "cancel_check", None)

    if fit_method == "irf":
        signal_sum = cube[mask].sum(axis=0).astype(float)
        shift = estimate_shift(signal_sum, irf["irf_norm"])
        irf_aligned = shift_array(irf["irf_norm"], shift)
        print(f"\n  IRF-to-data alignment: shift = {shift} bins ({shift*s['tpb']*1e9:.3f} ns)")
        if args.mono:
            g = global_reconv_fit_mono(signal_sum, irf_aligned, s["tpb"] * 1e9, tau_mono_bounds=tau_mono_bounds)
            print(f"  global reconvolution fit (masked pixels only, mono-exponential): "
                  f"tau = {g['tau']:.2f} ns")
        else:
            g = global_reconv_fit(signal_sum, irf_aligned, s["tpb"] * 1e9,
                                  tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds)
            print(f"  global reconvolution fit (masked pixels only): tau_short = {g['tau_short']:.2f} ns, "
                  f"tau_long = {g['tau_long']:.2f} ns")
        if getattr(args, "save_irf_check", True):
            plot_irf_diagnostic(irf, signal_sum, shift, g, s["tpb"], f"{prefix}_irf_check.png", show,
                                n_signal=int(mask.sum()), unit=unit)

        if args.mono:
            if args.fix_tau_short:
                print("  --fix-tau-short has no effect with --mono (only one lifetime to fit); ignoring.")
            fit = fit_pixels_reconv_mono(cube, irf_aligned, s["tpb"] * 1e9, mask, tau_mono_bounds=tau_mono_bounds,
                                         cancel_check=cancel_check)
        else:
            if args.fix_tau_short:
                tau_short_fixed = manual_tau_short if manual_tau_short is not None else g["tau_short"]
            else:
                tau_short_fixed = None
            if tau_short_fixed:
                source = "manually set" if manual_tau_short is not None else "from the global fit"
                print(f"  fixing tau_short = {tau_short_fixed:.2f} ns ({source}); floating only tau_long")
            fit = fit_pixels_reconv(cube, irf_aligned, s["tpb"] * 1e9, mask, tau_short_fixed,
                                    tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                                    cancel_check=cancel_check)
    elif fit_method == "mean":
        if args.mono:
            print("  --mono has no effect with fit_method=mean (there's only one lifetime to compute); ignoring.")
        if args.fix_tau_short:
            print("  --fix-tau-short has no effect with fit_method=mean; ignoring.")
        _, peak_bin, s_idx, tn = _tail_axis(s, cube, offset=getattr(args, "tail_start_offset", None))
        fit = mean_arrival_time(cube, s_idx, tn, mask)
        tlv = fit["tau_mean"][np.isfinite(fit["tau_mean"])]
        if tlv.size:
            print(f"  tau_mean: median {np.median(tlv):.2f}  "
                  f"IQR[{np.percentile(tlv,25):.2f}-{np.percentile(tlv,75):.2f}] ns  (peak bin {peak_bin})")
        if getattr(args, "save_results", True):
            plot_results(img, mask, removed, fit["tau_mean"], threshold, min_neighbors,
                        f"{prefix}_results.png", show, bin_size=args.bin,
                        label="tau_mean", symbol="\u27e8t\u27e9")
        if getattr(args, "save_quality_plots", True):
            plot_fit_quality(fit["tau_mean_std"], fit["n_photons"], mask, f"{prefix}_quality.png", show,
                             symbol="\u27e8t\u27e9")
        np.savez(f"{prefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, removed_isolated=removed,
                 tau_mean_ns=fit["tau_mean"], tau_mean_std_ns=fit["tau_mean_std"],
                 n_photons_fit=fit["n_photons"],
                 threshold=threshold, min_neighbors=min_neighbors, bin_size=args.bin, fit_method=fit_method)
        print(f"  saved {prefix}_maps.npz")
        print(f"Done with {h5file}.")
        return
    else:
        summed, peak_bin, s_idx, tn = _tail_axis(s, cube, offset=getattr(args, "tail_start_offset", None))
        g = global_fit(summed, s, s_idx, tn, mono=args.mono,
                       tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                       tau_mono_bounds=tau_mono_bounds)
        if args.mono:
            print(f"  global fit (mono-exponential): tau = {g['tau']:.2f} ns  (peak bin {peak_bin})")
        else:
            print(f"  global fit: tau_short = {g['tau_short']:.2f} ns, "
                  f"tau_long = {g['tau_long']:.2f} ns  (peak bin {peak_bin})")

        if args.mono:
            if args.fix_tau_short:
                print("  --fix-tau-short has no effect with --mono (only one lifetime to fit); ignoring.")
            fit = fit_pixels_mono(cube, s_idx, tn, mask, tau_mono_bounds=tau_mono_bounds,
                                  cancel_check=cancel_check)
        else:
            if args.fix_tau_short:
                tau_short_fixed = manual_tau_short if manual_tau_short is not None else g["tau_short"]
            else:
                tau_short_fixed = None
            if tau_short_fixed:
                source = "manually set" if manual_tau_short is not None else "from the global fit"
                print(f"  fixing tau_short = {tau_short_fixed:.2f} ns ({source}); floating only tau_long")
            fit = fit_pixels(cube, s_idx, tn, mask, tau_short_fixed,
                             tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                             cancel_check=cancel_check)

    n_converged = int(fit["converged"][mask].sum())
    print(f"  optimizer converged for {n_converged}/{int(mask.sum())} fitted pixels")

    # --- results + save (same shape regardless of which fit method ran) ---
    if args.mono:
        tau_map = fit["tau"]
        tlv = tau_map[np.isfinite(tau_map)]
        print(f"  tau: median {np.median(tlv):.2f}  "
              f"IQR[{np.percentile(tlv,25):.2f}-{np.percentile(tlv,75):.2f}] ns")
        std_v = fit["tau_std"][np.isfinite(fit["tau_std"])]
        if std_v.size:
            print(f"  tau_std: median {np.median(std_v):.3f} ns")
        if getattr(args, "save_results", True):
            plot_results(img, mask, removed, tau_map, threshold, min_neighbors,
                         f"{prefix}_results.png", show, bin_size=args.bin, label="tau", symbol="\u03c4")
        if getattr(args, "save_quality_plots", True):
            plot_fit_quality(fit["tau_std"], fit["n_photons"], mask, f"{prefix}_quality.png", show,
                             symbol="\u03c4", converged_map=fit["converged"])
        np.savez(f"{prefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, removed_isolated=removed,
                 tau_ns=tau_map, tau_std_ns=fit["tau_std"], n_photons_fit=fit["n_photons"],
                 converged=fit["converged"], nll=fit["nll"], model="mono",
                 threshold=threshold, min_neighbors=min_neighbors, bin_size=args.bin,
                 tau_global=g["tau"], tau_global_std=g.get("tau_std", np.nan), fit_method=fit_method)
    else:
        tau_long, tau_short = fit["tau_long"], fit["tau_short"]
        tlv = tau_long[np.isfinite(tau_long)]
        print(f"  tau_long: median {np.median(tlv):.2f}  "
              f"IQR[{np.percentile(tlv,25):.2f}-{np.percentile(tlv,75):.2f}] ns")
        std_v = fit["tau_long_std"][np.isfinite(fit["tau_long_std"])]
        if std_v.size:
            print(f"  tau_long_std: median {np.median(std_v):.3f} ns")
        if getattr(args, "save_results", True):
            plot_results(img, mask, removed, tau_long, threshold, min_neighbors,
                         f"{prefix}_results.png", show, bin_size=args.bin)
        if getattr(args, "save_quality_plots", True):
            plot_fit_quality(fit["tau_long_std"], fit["n_photons"], mask, f"{prefix}_quality.png", show,
                             symbol="\u03c4_long", converged_map=fit["converged"])
        np.savez(f"{prefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, removed_isolated=removed,
                 tau_long_ns=tau_long, tau_short_ns=tau_short,
                 tau_long_std_ns=fit["tau_long_std"], tau_short_std_ns=fit["tau_short_std"],
                 n_photons_fit=fit["n_photons"], converged=fit["converged"], nll=fit["nll"], model="bi",
                 threshold=threshold, min_neighbors=min_neighbors, bin_size=args.bin,
                 tau_short_global=g["tau_short"], tau_long_global=g["tau_long"],
                 tau_short_global_std=g.get("tau_short_std", np.nan),
                 tau_long_global_std=g.get("tau_long_std", np.nan), fit_method=fit_method)
    print(f"  saved {prefix}_maps.npz")
    print(f"Done with {h5file}.")


def main():
    ap = argparse.ArgumentParser(description="Flimera FLIM analysis (normal, pixel-wise)")
    ap.add_argument("h5file", nargs="+",
                    help="one or more data files (raw .H5, exported, or .ds3d project); "
                         "a shell glob like *.H5 also works")
    ap.add_argument("--threshold", type=int, default=None,
                    help="min photons/pixel for the mask (skips the interactive prompt)")
    ap.add_argument("--tail-start-offset", type=int, default=None,
                    help=f"bins after the decay peak where the tail fit starts "
                         f"(default {TAIL_START_OFFSET}). No effect with --fit-method irf.")
    ap.add_argument("--fit-method", choices=["tail", "irf", "mean"], default="tail",
                    help="'tail' (default): fit the decay tail only. 'irf': fit the FULL curve "
                         "against a measured instrument response (reconvolution) -- requires --irf. "
                         "'mean': no fit at all -- compute the photon-weighted mean arrival time "
                         "(fast, no bounds; --mono/--fix-tau-short have no effect).")
    ap.add_argument("--irf", default=None,
                    help="IRF measurement file (any supported format), required with --fit-method irf.")
    ap.add_argument("--fix-tau-short", action="store_true",
                    help="fix tau_short to the global value; float only tau_long. No effect with --mono.")
    ap.add_argument("--tau-short-value", type=float, default=None,
                    help="use with --fix-tau-short to supply an exact tau_short value (ns) yourself.")
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
                    help=f"spatial binning factor NxN (default {BIN_SIZE} = no binning)")
    ap.add_argument("--roi", type=lambda s: tuple(int(p) for p in s.split(",")), default=None,
                    help="crop to a rectangular region of interest before analysis: "
                         "x0,y0,x1,y1 in pixels (x1/y1 exclusive)")
    ap.add_argument("--roi-prompt", action="store_true",
                    help="interactively ask for an ROI after showing the full-image overview")
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

    files = []
    for pattern in args.h5file:
        if any(ch in pattern for ch in "*?["):
            matches = sorted(glob.glob(pattern))
            if not matches:
                print(f"Warning: pattern '{pattern}' matched no files -- skipping.")
            files.extend(matches)
        else:
            files.append(pattern)
    if not files:
        print("No input files found.")
        return

    for h5file in files:
        process_file(h5file, args)

    if len(files) > 1:
        print(f"\nAll {len(files)} files done.")


def run(args):
    """GUI/plugin-interface entry point -- one file at a time, via
    process_file, matching every other plugin's run(args) signature."""
    process_file(args.h5file, args)


if __name__ == "__main__":
    main()
