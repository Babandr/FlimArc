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
import copy
import glob
from pathlib import Path

import numpy as np

from flim_core.config import MIN_NEIGHBORS, BIN_SIZE, DEFAULT_THRESHOLD, TAIL_START_OFFSET
from flim_core.io import load_flim_data, photon_count_map, build_decay_cube
from flim_core.masking import (
    clean_mask, bin_cube, bin_time_cube, moving_average_cube, crop_to_roi, prompt_roi,
    report_dead_pixels, load_mask,
)
from flim_core.fitting import _tail_axis, global_fit, fit_pixels_mono, fit_pixels, mean_arrival_time
from flim_core.plotting import plot_overview_and_mask, plot_results, plot_fit_quality
from flim_core.irf import (
    load_irf, estimate_shift, shift_array,
    global_reconv_fit, global_reconv_fit_mono,
    fit_pixels_reconv, fit_pixels_reconv_mono,
    mean_arrival_time_reconv,
    plot_irf_diagnostic,
)
from flim_core.pixel_offset import PixelOffsetCorrection
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

    stem = Path(h5file).stem
    prefix = f"{args.prefix}_{stem}" if args.prefix else stem
    show = not args.no_show

    print(f"\n=== {h5file} ===")
    print("Loading data ...")
    s = load_flim_data(h5file)
    print(f"  format: {s['source']} | {s['n_photons']} photons | sensor {s['W']}x{s['H']} | "
          f"bins {s['B0']}-{s['B1']} | {s['tpb']*1e12:.1f} ps/bin")

    if irf is not None and abs(s["tpb"] * (getattr(args, "time_bin", None) or 1) - irf["tpb"]) > 1e-15:
        print(f"  WARNING: data TimePerBin ({s['tpb'] * (getattr(args, 'time_bin', None) or 1):.3e} s) differs from IRF's "
              f"({irf['tpb']:.3e} s) -- these were likely acquired under different "
              f"timing settings and this IRF may not be valid for this data.")

    img = photon_count_map(s)
    img_orig = img   # kept for the overview panel, before any ROI-crop/binning reassigns img
    nz = img[img > 0]
    print(f"  per-pixel photons: median {int(np.median(nz))}, mean {nz.mean():.1f}, "
          f"90th pct {int(np.percentile(nz, 90))}, max {int(nz.max())}")
    report_dead_pixels(img)

    cube = build_decay_cube(s)

    # Per-pixel SPAD timing offset correction -- applied before any spatial
    # binning or ROI crop so every downstream step sees aligned histograms.
    # Skipped when args.pixel_offset is None (the default / checkbox unchecked).
    pixel_offset_path = getattr(args, "pixel_offset", None)
    pixel_offset_meta = "none"   # written into every .npz for reproducibility
    if pixel_offset_path:
        print(f"\nApplying pixel offset correction from {Path(pixel_offset_path).name} ...")
        try:
            corr = PixelOffsetCorrection.load(pixel_offset_path)
            cube = corr.apply(cube, s["tpb"])
            pixel_offset_meta = str(Path(pixel_offset_path).resolve())
        except Exception as exc:
            print(f"  WARNING: pixel offset correction failed ({exc}); continuing without it.")

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
    # exclusive alternatives (see masking.moving_average_cube's docstring
    # for why mixing them doesn't make sense -- block binning already
    # pools disjoint groups of pixels, so "moving-averaging" that result
    # again would just smear super-pixels into each other). If both are
    # given, moving-average wins and binning is ignored, with a warning.
    moving_avg = getattr(args, "moving_average", None)
    if moving_avg and moving_avg > 1:
        if args.bin and args.bin > 1:
            print(f"  WARNING: both --bin {args.bin} and --moving-average {moving_avg} given; "
                  f"using moving-average only, ignoring --bin.")
        img, cube = moving_average_cube(img, cube, moving_avg)
        nz_b = img[img > 0]
        print(f"  after {moving_avg}x{moving_avg} moving average: median {int(np.median(nz_b))}, "
              f"90th pct {int(np.percentile(nz_b, 90))} photons/pixel")
        binning_used = False   # for plotting purposes: output is still full-resolution, see below
        plot_bin_size = 1
    else:
        binning_used = bool(args.bin and args.bin > 1)
        if binning_used:
            img, cube = bin_cube(img, cube, args.bin)
            nz_b = img[img > 0]
            print(f"  after {args.bin}x{args.bin} binning: median {int(np.median(nz_b))}, "
                  f"90th pct {int(np.percentile(nz_b, 90))} photons/super-pixel")
        plot_bin_size = args.bin

    # Time binning: groups adjacent microtime channels together, reducing
    # noise per bin / smoothing out any single channel's own nonlinearity.
    # Orthogonal to the spatial smoothing above -- combines freely with
    # either block binning or moving-average. Updates s["tpb"] (and B1, to
    # stay consistent with the now-shorter cube) so every downstream ns-
    # based calculation keeps working in real ns without further changes
    # -- see bin_time_cube's docstring.
    time_bin = getattr(args, "time_bin", None) or 1
    if time_bin > 1:
        cube, tpb_binned = bin_time_cube(cube, time_bin, s["tpb"])
        s = dict(s)
        s["tpb"] = tpb_binned
        s["B1"] = s["B0"] + cube.shape[2] - 1
    # TAIL_START_OFFSET (and any --tail-start-offset override) is expressed
    # in bins at the ORIGINAL bin width -- after time binning, the same raw
    # number of bins covers time_bin times more real ns than intended.
    # Rescale it so the real ns margin past the peak stays constant
    # regardless of time binning, rather than silently growing with it.
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
                               plot_bin_size, f"{prefix}_overview.png", show)

    manual_tau_short = getattr(args, "tau_short_value", None)
    tau_short_bounds = getattr(args, "tau_short_bounds", None)
    tau_long_bounds = getattr(args, "tau_long_bounds", None)
    tau_mono_bounds = getattr(args, "tau_mono_bounds", None)
    cancel_check = getattr(args, "cancel_check", None)

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

    if fit_method == "irf":
        if args.mono:
            g = global_reconv_fit_mono(signal_sum, irf_aligned, s["tpb"] * 1e9, tau_mono_bounds=tau_mono_bounds,
                                       no_shift=no_shift)
            print(f"  global reconvolution fit (masked pixels only, mono-exponential): "
                  f"tau = {g['tau']:.2f} ns")
        else:
            g = global_reconv_fit(signal_sum, irf_aligned, s["tpb"] * 1e9,
                                  tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                                  no_shift=no_shift)
            print(f"  global reconvolution fit (masked pixels only): tau_short = {g['tau_short']:.2f} ns, "
                  f"tau_long = {g['tau_long']:.2f} ns")
        if getattr(args, "save_irf_check", True):
            plot_irf_diagnostic(irf, signal_sum, shift, g, s["tpb"], f"{prefix}_irf_check.png", show,
                                n_signal=int(mask.sum()), unit=unit)

        if args.mono:
            if args.fix_tau_short:
                print("  --fix-tau-short has no effect with --mono (only one lifetime to fit); ignoring.")
            fit = fit_pixels_reconv_mono(cube, irf_aligned, s["tpb"] * 1e9, mask, tau_mono_bounds=tau_mono_bounds,
                                         cancel_check=cancel_check, no_shift=no_shift)
        else:
            if args.fix_tau_short and manual_tau_short is None:
                # Source tau_short from the MEDIAN of a free per-pixel fit, not the
                # global (whole-mask-summed) fit. If pixels under the mask have
                # genuinely different tau_long (real heterogeneity, not noise), the
                # summed curve is a blend of decays with different rates -- its own
                # best-fit tau_short is a biased property of the *pooled* curve, not
                # a representative "typical pixel" value. On real data this pushed
                # tau_short to the edge of its bounds even though per-pixel fits were
                # well-resolved and nowhere near that edge. Costs one extra full pass
                # over the masked pixels; only runs when fix_tau_short is requested.
                print(f"  --fix-tau-short: running a free per-pixel fit first to get a "
                      f"representative tau_short (avoids biasing from the pooled global fit) ...")
                free_fit = fit_pixels_reconv(cube, irf_aligned, s["tpb"] * 1e9, mask, None,
                                             tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                                             cancel_check=cancel_check, no_shift=no_shift)
                good = free_fit["converged"] & mask
                if good.sum() == 0:
                    print(f"  WARNING: no converged pixels in the free pre-fit; "
                          f"falling back to the global fit's tau_short ({g['tau_short']:.2f} ns).")
                    tau_short_fixed = g["tau_short"]
                    source = "global fit (fallback: free pre-fit had no converged pixels)"
                else:
                    tau_short_fixed = float(np.median(free_fit["tau_short"][good]))
                    source = f"median of {int(good.sum())} converged per-pixel free fits"
            elif args.fix_tau_short:
                tau_short_fixed = manual_tau_short
                source = "manually set"
            else:
                tau_short_fixed = None
                source = None
            if tau_short_fixed:
                print(f"  fixing tau_short = {tau_short_fixed:.2f} ns ({source}); floating only tau_long")
            fit = fit_pixels_reconv(cube, irf_aligned, s["tpb"] * 1e9, mask, tau_short_fixed,
                                    tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                                    cancel_check=cancel_check, no_shift=no_shift)
    elif fit_method == "mean_irf":
        if args.mono:
            print("  --mono has no effect with fit_method=mean_irf (there's only one lifetime to compute); ignoring.")
        if args.fix_tau_short:
            print("  --fix-tau-short has no effect with fit_method=mean_irf; ignoring.")
        fit = mean_arrival_time_reconv(cube, irf_aligned, s["tpb"] * 1e9, mask, no_shift=no_shift)
        tlv = fit["tau_mean"][np.isfinite(fit["tau_mean"])]
        if tlv.size:
            print(f"  tau_mean (IRF-corrected): median {np.median(tlv):.2f}  "
                  f"IQR[{np.percentile(tlv,25):.2f}-{np.percentile(tlv,75):.2f}] ns")
        if getattr(args, "save_results", True):
            plot_results(img, mask, removed, fit["tau_mean"], threshold, min_neighbors,
                        f"{prefix}_results.png", show, bin_size=plot_bin_size,
                        label="tau_mean_irf", symbol="\u27e8t\u27e9")
        if getattr(args, "save_quality_plots", True):
            plot_fit_quality(fit["tau_mean_std"], fit["n_photons"], mask, f"{prefix}_quality.png", show,
                             symbol="\u27e8t\u27e9")
        np.savez(f"{prefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, removed_isolated=removed,
                 tau_mean_ns=fit["tau_mean"], tau_mean_std_ns=fit["tau_mean_std"],
                 n_photons_fit=fit["n_photons"], irf_shift=fit["shift"],
                 threshold=threshold, min_neighbors=min_neighbors, bin_size=plot_bin_size, fit_method=fit_method,
                 moving_average=(moving_avg or 1), time_bin=time_bin,
                 pixel_offset_calibration=pixel_offset_meta)
        print(f"  saved {prefix}_maps.npz")
        print(f"Done with {h5file}.")
        return
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
                        f"{prefix}_results.png", show, bin_size=plot_bin_size,
                        label="tau_mean", symbol="\u27e8t\u27e9")
        if getattr(args, "save_quality_plots", True):
            plot_fit_quality(fit["tau_mean_std"], fit["n_photons"], mask, f"{prefix}_quality.png", show,
                             symbol="\u27e8t\u27e9")
        np.savez(f"{prefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, removed_isolated=removed,
                 tau_mean_ns=fit["tau_mean"], tau_mean_std_ns=fit["tau_mean_std"],
                 n_photons_fit=fit["n_photons"],
                 threshold=threshold, min_neighbors=min_neighbors, bin_size=plot_bin_size, fit_method=fit_method,
                 moving_average=(moving_avg or 1), time_bin=time_bin,
                 pixel_offset_calibration=pixel_offset_meta)
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
            if args.fix_tau_short and manual_tau_short is None:
                # Same reasoning as the IRF branch above: source tau_short from the
                # median of a free per-pixel fit rather than the global-sum fit, since
                # the summed curve's best-fit tau_short is biased by pixel-to-pixel
                # tau_long heterogeneity, not representative of any individual pixel.
                print(f"  --fix-tau-short: running a free per-pixel fit first to get a "
                      f"representative tau_short (avoids biasing from the pooled global fit) ...")
                free_fit = fit_pixels(cube, s_idx, tn, mask, None,
                                      tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                                      cancel_check=cancel_check)
                good = free_fit["converged"] & mask
                if good.sum() == 0:
                    print(f"  WARNING: no converged pixels in the free pre-fit; "
                          f"falling back to the global fit's tau_short ({g['tau_short']:.2f} ns).")
                    tau_short_fixed = g["tau_short"]
                    source = "global fit (fallback: free pre-fit had no converged pixels)"
                else:
                    tau_short_fixed = float(np.median(free_fit["tau_short"][good]))
                    source = f"median of {int(good.sum())} converged per-pixel free fits"
            elif args.fix_tau_short:
                tau_short_fixed = manual_tau_short
                source = "manually set"
            else:
                tau_short_fixed = None
                source = None
            if tau_short_fixed:
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
                         f"{prefix}_results.png", show, bin_size=plot_bin_size, label="tau", symbol="\u03c4")
        if getattr(args, "save_quality_plots", True):
            plot_fit_quality(fit["tau_std"], fit["n_photons"], mask, f"{prefix}_quality.png", show,
                             symbol="\u03c4", converged_map=fit["converged"])
        np.savez(f"{prefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, removed_isolated=removed,
                 tau_ns=tau_map, tau_std_ns=fit["tau_std"], n_photons_fit=fit["n_photons"],
                 converged=fit["converged"], nll=fit["nll"], model="mono",
                 threshold=threshold, min_neighbors=min_neighbors, bin_size=plot_bin_size,
                 tau_global=g["tau"], tau_global_std=g.get("tau_std", np.nan), fit_method=fit_method,
                 moving_average=(moving_avg or 1), time_bin=time_bin,
                 pixel_offset_calibration=pixel_offset_meta)
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
                         f"{prefix}_results.png", show, bin_size=plot_bin_size)
        if getattr(args, "save_quality_plots", True):
            plot_fit_quality(fit["tau_long_std"], fit["n_photons"], mask, f"{prefix}_quality.png", show,
                             symbol="\u03c4_long", converged_map=fit["converged"])
        np.savez(f"{prefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, removed_isolated=removed,
                 tau_long_ns=tau_long, tau_short_ns=tau_short,
                 tau_long_std_ns=fit["tau_long_std"], tau_short_std_ns=fit["tau_short_std"],
                 n_photons_fit=fit["n_photons"], converged=fit["converged"], nll=fit["nll"], model="bi",
                 threshold=threshold, min_neighbors=min_neighbors, bin_size=plot_bin_size,
                 tau_short_global=g["tau_short"], tau_long_global=g["tau_long"],
                 tau_short_global_std=g.get("tau_short_std", np.nan),
                 tau_long_global_std=g.get("tau_long_std", np.nan), fit_method=fit_method,
                 moving_average=(moving_avg or 1), time_bin=time_bin,
                 pixel_offset_calibration=pixel_offset_meta)
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
                         f"(default {TAIL_START_OFFSET}). No effect with --fit-method irf or mean_irf.")
    ap.add_argument("--fit-method", choices=["tail", "irf", "mean", "mean_irf"], default="tail",
                    help="'tail' (default): fit the decay tail only. 'irf': fit the FULL curve "
                         "against a measured instrument response (reconvolution) -- requires --irf. "
                         "'mean': no fit at all -- compute the photon-weighted mean arrival time over "
                         "the tail only (fast, no bounds; --mono/--fix-tau-short have no effect). "
                         "'mean_irf': same, but IRF-corrected and computed over the FULL curve instead "
                         "of just the tail -- requires --irf; measures a different (generally lower, "
                         "since it includes any fast component) quantity than plain 'mean'.")
    ap.add_argument("--irf", default=None,
                    help="IRF measurement file (any supported format), required with "
                         "--fit-method irf or mean_irf.")
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
                    help=f"spatial binning factor NxN (default {BIN_SIZE} = no binning). Block "
                         "binning: pools disjoint NxN blocks into one super-pixel each, reducing "
                         "resolution. Ignored if --moving-average is also given.")
    ap.add_argument("--moving-average", type=int, default=None,
                    help="spatial smoothing by a centered NxN moving average instead of block "
                         "binning: output stays at full resolution, but neighbouring pixels share "
                         "most of their source photons (NOT statistically independent -- see "
                         "flim_core.masking.moving_average_cube). Must be odd (3, 5, 7, ...). "
                         "Overrides --bin if both are given.")
    ap.add_argument("--time-bin", type=int, default=1,
                    help="group every N adjacent microtime bins into one, reducing time resolution "
                         "but also photon noise per bin and sensitivity to any single bin's own "
                         "nonlinearity. Orthogonal to spatial binning/moving-average -- combines "
                         "freely with either. --tail-start-offset is automatically rescaled to keep "
                         "the same real ns margin regardless of this setting (default 1 = no binning).")
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
