"""
Helpers shared by the "split the acquisition into groups" family of plugins
(periodic, periodic_wholeimage, timelapse, timelapse_wholeimage) but not used
by the plain normal/normal_wholeimage plugins.

Underscore-prefixed so analyses/_registry.py (which mirrors NanoPrepare's
protocols/importer.py convention) skips this file when auto-discovering
selectable plugins -- it is a shared template/helper module, not a plugin.
"""

import copy

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from flim_core.io import _find_histogram_group
from flim_core.masking import (
    clean_mask, bin_cube, bin_time_cube, moving_average_cube, report_dead_pixels,
)
from flim_core.config import TAIL_START_OFFSET
from flim_core.fitting import _tail_axis, global_fit, fit_pixels, fit_pixels_mono, _resolve_bounds, _curve_fit_std, mean_arrival_time
from flim_core.plotting import plot_overview_and_mask, plot_results, plot_fit_quality
from flim_core.irf import (
    global_reconv_fit, global_reconv_fit_mono, global_reconv_fit_fixed_short,
    fit_pixels_reconv, fit_pixels_reconv_mono, mean_arrival_time_reconv,
)


def apply_spatial_and_time_binning(img, cube, d, args, allow_moving_average=True):
    """Shared by every grouped plugin (Periodic/Timelapse, pixel-wise and
    whole-image): applies the run's spatial smoothing and time binning to
    one image/cube pair and returns everything downstream code needs to
    stay consistent with the result, WITHOUT mutating the caller's `d` or
    `args` (both come back as shallow copies where anything changed, so
    this is safe to call once per group/window without compounding).

    Spatial smoothing is one of: block binning (--bin N), moving average
    (--moving-average N, which wins if both are given), or none. Moving
    average is skipped where allow_moving_average=False -- the whole-image
    plugins pool every masked pixel into one curve, and moving-averaged
    pixels share photons, so pooling them would count each photon up to
    N**2 times and overstate precision (see normal_wholeimage.py).
    Time binning (--time-bin N) is orthogonal and always allowed.

    Returns (img, cube, d, args, binning_used):
      d      -- tpb and B1 updated for the time-binned cube (B0 unchanged)
      args   -- tail_start_offset rescaled so the real-ns margin past the
                peak is unchanged by time binning; if moving average was
                applied, bin is set to 1 so every downstream plot/label
                that reads args.bin correctly sees a full-resolution
                image rather than mistaking the smoothing window for a
                binning factor
      binning_used -- True only for block binning (drives the
                "super-pixels" vs "pixels" wording)."""
    moving_avg = getattr(args, "moving_average", None) if allow_moving_average else None
    d = dict(d)
    args = copy.copy(args)
    if moving_avg and moving_avg > 1:
        if args.bin and args.bin > 1:
            print(f"  WARNING: both --bin {args.bin} and --moving-average {moving_avg} given; "
                  f"using moving-average only, ignoring --bin.")
        img, cube = moving_average_cube(img, cube, moving_avg)
        args.bin = 1
        binning_used = False
    else:
        if getattr(args, "moving_average", None) and getattr(args, "moving_average") > 1 \
                and not allow_moving_average:
            print(f"  WARNING: --moving-average is ignored in whole-image mode (pooled pixels "
                  f"would count each photon up to {args.moving_average**2}x, overstating precision).")
        binning_used = bool(args.bin and args.bin > 1)
        if binning_used:
            img, cube = bin_cube(img, cube, args.bin)
    time_bin = getattr(args, "time_bin", None) or 1
    if time_bin > 1:
        cube, tpb_binned = bin_time_cube(cube, time_bin, d["tpb"])
        d["tpb"] = tpb_binned
        d["B1"] = d["B0"] + cube.shape[2] - 1
    raw_offset = getattr(args, "tail_start_offset", None)
    if raw_offset is None:
        raw_offset = TAIL_START_OFFSET
    args.tail_start_offset = max(1, round(raw_offset / time_bin)) if time_bin > 1 else \
        getattr(args, "tail_start_offset", None)
    return img, cube, d, args, binning_used


def window_boundaries(window_idx, n_windows):
    """[start, end) photon-index boundaries of each time window, as an
    array of n_windows + 1 offsets (window w is slice(b[w], b[w + 1])).

    Valid because a raw photon stream is stored in acquisition order, so
    window_idx (a monotone function of each photon's time) is sorted --
    one searchsorted finds every boundary at once, and each window is then
    a SLICE (a view) instead of a fresh full-length boolean mask. At a few
    hundred million photons that mask is hundreds of MiB per window, which
    is what ran one machine out of memory (see timelapse.py's run()).
    A strided spot-check guards the sortedness assumption cheaply (a full
    np.diff would itself allocate gigabytes) and fails loudly instead of
    silently mis-assigning photons to windows."""
    step = max(1, len(window_idx) // 200000)
    if np.any(np.diff(window_idx[::step]) < 0):
        raise ValueError("Photon time stamps are not in increasing order, so time windows can't be "
                         "taken as contiguous slices of the photon stream.")
    return np.searchsorted(window_idx, np.arange(n_windows + 1))


def load_raw_with_time(path):
    """Decode a RAW Flimera photon-stream file into per-photon x, y, microtime,
    and elapsed-time arrays (one entry per detected photon), using the same
    frame-marker timestamping worked out for the GUI viewer: every frame-start
    marker in the stream gets a running frame index (via a cumulative count of
    markers-so-far), and that index is carried over to the photon records that
    follow it, then converted to milliseconds via the camera's frame rate.
    Raises a clear error if given an exported/project file instead, since
    those only store one pre-summed histogram per pixel -- no per-photon
    timing survives that export, so phase-splitting isn't possible on them."""
    with h5py.File(path, "r") as f:
        if _find_histogram_group(f) is not None:
            raise ValueError(
                f"'{path}' is a pre-computed-histogram file (exported or .ds3d project) -- "
                "it has no per-photon timing left, so it can't be split by phase. "
                "Phase-resolved analysis needs a RAW photon-stream file."
            )
        dset = None
        def _find(name, obj):
            nonlocal dset
            if isinstance(obj, h5py.Dataset) and obj.ndim == 1 and obj.dtype == np.uint32:
                dset = obj.name
        f.visititems(_find)
        if dset is None:
            dset = "Streams/Photon"
        d = f[dset]; data = d[:]; a = d.attrs
        tpb = float(np.ravel(a["TimePerBin"])[0])
        W = int(np.ravel(a["CameraWidth"])[0]); H = int(np.ravel(a["CameraHeight"])[0])
        try:
            B0 = int(np.ravel(a["BinRangeLowerBounds"])[0])
            B1 = int(np.ravel(a["BinRangeUpperBounds"])[0])
        except KeyError:
            B0 = B1 = None
        try:
            frame_rate_hz = float(np.ravel(a["FrameRateHz"])[0])
        except KeyError:
            frame_rate_hz = None
        if not frame_rate_hz:
            raise ValueError(f"'{path}' has no FrameRateHz metadata -- can't convert "
                             "frame position to elapsed time for phase splitting.")

    top = (data >> 28) & 0xF
    is_frame_marker = (top == 0xF) & (((data >> 16) & 0xF) == 1)
    frame_id_all = np.cumsum(is_frame_marker) - 1

    is_photon = top == 0x2
    ph = data[is_photon]
    frame_id = frame_id_all[is_photon]

    x = (ph & 0xFF).astype(np.int32) - 1
    y = ((ph >> 8) & 0xFF).astype(np.int32)
    micro = (((ph >> 24) & 0xF) << 8) | ((ph >> 16) & 0xFF)

    if B0 is None:
        B0, B1 = int(micro.min()), int(micro.max())

    keep = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (micro >= B0) & (micro <= B1)
    x, y, micro, frame_id = x[keep], y[keep], micro[keep], frame_id[keep]
    nb = B1 - B0 + 1
    n_frames = int(is_frame_marker.sum())
    time_ms = frame_id.astype(np.float64) / frame_rate_hz * 1000.0

    # Rotate 90 degrees counter-clockwise from the raw sensor readout, same
    # as the main script's _rotate_ccw, applied here at the photon-coordinate
    # level since this loader never builds a single [H,W] array of its own
    # (each phase group/time window builds its own image later). For a 90
    # deg CCW rotation of an HxW array: new_x = old_y, new_y = (W-1)-old_x.
    x, y = y.copy(), (W - 1 - x)
    W, H = H, W

    return dict(x=x, y=y, micro=micro, time_ms=time_ms, W=W, H=H, B0=B0, B1=B1,
                nb=nb, tpb=tpb, frame_rate_hz=frame_rate_hz, n_frames=n_frames,
                n_photons=int(keep.sum()), duration_ms=float(time_ms.max()) if time_ms.size else 0.0)


def assign_phase_groups(time_ms, period_ms, n_groups):
    """For each photon, which phase group (0..n_groups-1) does it belong to,
    based on where its elapsed time falls within one period (wrapping every
    period_ms, i.e. modulo)."""
    phase = np.mod(time_ms, period_ms)
    group_width = period_ms / n_groups
    group_idx = np.clip((phase // group_width).astype(int), 0, n_groups - 1)
    return group_idx


# Photons processed per pass when a contiguous slice is built into a cube.
# Each pass needs only a few bytes-per-photon of temporaries, so peak memory
# is set by this constant plus the cube itself -- NOT by how many photons the
# selection holds. (A single pass over hundreds of millions of photons needed
# gigabytes of intermediates.)
_BUILD_CHUNK = 10_000_000


def build_group_cube(d, sel):
    """Build the intensity image and decay cube from just the photons
    selected by `sel`: a boolean mask (e.g. one phase group's photons -- not
    contiguous in the stream) or a slice (a contiguous run, e.g. one time
    window, or slice(None) for every photon). Slices are processed in
    _BUILD_CHUNK-sized passes so memory stays bounded for any photon count;
    the result is identical to a single pass (counts are exact integers in
    both float64 and float32 up to 2**24 per bin)."""
    W, H, B0, nb = d["W"], d["H"], d["B0"], d["nb"]
    if isinstance(sel, slice):
        start, stop, _ = sel.indices(len(d["x"]))
        if stop - start > _BUILD_CHUNK:
            img = np.zeros(H * W)
            cube = np.zeros(H * W * nb, dtype=np.float32)
            for a in range(start, stop, _BUILD_CHUNK):
                b = min(a + _BUILD_CHUNK, stop)
                pix = d["y"][a:b] * W + d["x"][a:b]
                img += np.bincount(pix, minlength=H * W)
                cube += np.bincount(pix * nb + (d["micro"][a:b] - B0),
                                    minlength=H * W * nb).astype(np.float32)
            return img.reshape(H, W), cube.reshape(H, W, nb)
    xs, ys, ms = d["x"][sel], d["y"][sel], d["micro"][sel]
    img = np.bincount(ys * W + xs, minlength=H * W).reshape(H, W).astype(float)
    flat = (ys * W + xs) * nb + (ms - B0)
    cube = np.bincount(flat, minlength=H * W * nb).reshape(H, W, nb).astype(np.float32)
    return img, cube


def crop_photon_data(d, roi):
    """Crop the photon-level data dict (from load_raw_with_time) to a
    rectangular region of interest (x0, y0, x1, y1), x1/y1 exclusive.
    Filters every photon array down to just those inside the box, and
    shifts x/y so the cropped region's own origin becomes (0, 0) -- so
    downstream group/window cube-building (which assumes 0-indexed
    coordinates within [0, W) x [0, H)) works unchanged. Doing this ONCE,
    before any group/window splitting, means every group/window built
    afterward is automatically restricted to the ROI, and every downstream
    step (fewer pixels to bincount, mask, and fit) gets faster."""
    x0, y0, x1, y1 = roi
    W, H = d["W"], d["H"]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI ({roi}) is empty after clamping to the image bounds ({W}x{H}).")
    sel = (d["x"] >= x0) & (d["x"] < x1) & (d["y"] >= y0) & (d["y"] < y1)
    d2 = dict(d)
    d2["x"] = d["x"][sel] - x0
    d2["y"] = d["y"][sel] - y0
    d2["micro"] = d["micro"][sel]
    d2["time_ms"] = d["time_ms"][sel]
    d2["W"], d2["H"] = x1 - x0, y1 - y0
    d2["n_photons"] = int(sel.sum())
    print(f"  ROI crop: ({x0},{y0})-({x1},{y1})  ->  {x1-x0}x{y1-y0} pixels "
          f"(was {W}x{H})  |  {d2['n_photons']} photons remain (was {d['n_photons']})")
    return d2, (x0, y0, x1, y1)


def process_group(g_idx, n_groups, img, cube, d, prefix, args, threshold, min_neighbors, phase_desc,
                  irf_aligned=None, external_mask=None):
    """Run the mask -> fit -> plot -> save pipeline for one phase group's
    pooled decay cube, using an ALREADY-CHOSEN threshold (reused across every
    group, per the user's workflow: pick it once on group 1, apply to all).
    Returns a small summary dict for the final cross-phase comparison plot,
    or None if this group had no pixels pass the mask.

    irf_aligned: if given (and args.fit_method == "irf"), the already-
    aligned/background-subtracted IRF kernel to use for reconvolution
    fitting instead of the default tail fit. Alignment itself is computed
    ONCE per file by the caller (not per group), since it reflects the
    optical/detector system's timing, not something that varies within a
    single acquisition.

    external_mask: if given (a boolean array, already loaded by the
    caller), use it directly for EVERY group instead of threshold/
    min_neighbors -- the same fixed set of pixels across the whole
    acquisition, same idea as the whole-image plugins' fixed mask, just
    also honored here for the pixel-wise plugins."""
    fit_method = getattr(args, "fit_method", "tail")
    gprefix = f"{prefix}_group{g_idx+1}of{n_groups}"
    print(f"\n--- Group {g_idx+1}/{n_groups}  ({phase_desc}) ---")

    nz = img[img > 0]
    if nz.size:
        print(f"  photons: total {int(img.sum())} | per-pixel median {int(np.median(nz))}, "
              f"90th pct {int(np.percentile(nz, 90))}, max {int(nz.max())}")
    else:
        print("  no photons in this group at all -- skipping.")
        return None

    img_orig = img   # unbinned, for the overview panel
    # Spatial smoothing + time binning (see apply_spatial_and_time_binning).
    # d and args are rebound to updated COPIES here, so every later use in
    # this function -- d["tpb"] in the fits, args.bin / args.tail_start_offset
    # in plots and the tail axis -- automatically sees the post-binning
    # values with no further edits, and nothing leaks back to the caller.
    img, cube, d, args, binning_used = apply_spatial_and_time_binning(img, cube, d, args)

    if external_mask is not None:
        if external_mask.shape != img.shape:
            print(f"  Error: external mask shape {external_mask.shape} does not match this "
                  f"group's image shape {img.shape} -- check --bin/--roi match how the mask "
                  f"was saved. Skipping this group.")
            return None
        mask = external_mask
        removed = np.zeros_like(mask)
        unit = "super-pixels" if binning_used else "pixels"
        print(f"  external mask: {int(mask.sum())} {unit} selected")
        threshold_for_display = None
    else:
        raw_mask = img > threshold
        mask = clean_mask(raw_mask, min_neighbors)
        removed = raw_mask & ~mask
        unit = "super-pixels" if binning_used else "pixels"
        print(f"  mask (threshold={threshold}): {int(raw_mask.sum())} {unit} above threshold, "
              f"{int(mask.sum())} kept after neighbour filter")
        threshold_for_display = threshold
    if mask.sum() == 0:
        print(f"  no {unit} remain for this group; skipping fit.")
        return None

    if getattr(args, "save_overview", True):
        plot_overview_and_mask(img_orig, img, mask, removed, threshold_for_display, min_neighbors, args.bin,
                               f"{gprefix}_overview.png", not args.no_show)

    manual_tau_short = getattr(args, "tau_short_value", None)
    tau_short_bounds = getattr(args, "tau_short_bounds", None)
    tau_long_bounds = getattr(args, "tau_long_bounds", None)
    tau_mono_bounds = getattr(args, "tau_mono_bounds", None)
    cancel_check = getattr(args, "cancel_check", None)
    no_shift = getattr(args, "no_shift", False)

    if fit_method == "irf":
        # Per-group reconvolution reference fit (mirrors the tail-fit
        # global_fit() call below): fit this group's OWN masked-and-pooled
        # curve, used both as the printed reference and as the auto-derive
        # source for --fix-tau-short, exactly the same role global_fit()
        # plays for the tail-fit path.
        signal_sum = cube[mask].sum(axis=0).astype(float)
        if args.mono:
            g = global_reconv_fit_mono(signal_sum, irf_aligned, d["tpb"] * 1e9,
                                       tau_mono_bounds=tau_mono_bounds, no_shift=no_shift)
            print(f"  global reconvolution fit (mono): tau = {g['tau']:.2f} ns")
            fit = fit_pixels_reconv_mono(cube, irf_aligned, d["tpb"] * 1e9, mask,
                                         tau_mono_bounds=tau_mono_bounds, cancel_check=cancel_check,
                                         no_shift=no_shift)
        else:
            g = global_reconv_fit(signal_sum, irf_aligned, d["tpb"] * 1e9,
                                  tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                                  no_shift=no_shift)
            print(f"  global reconvolution fit: tau_short = {g['tau_short']:.2f} ns, "
                  f"tau_long = {g['tau_long']:.2f} ns")
            if args.fix_tau_short:
                tau_short_fixed = manual_tau_short if manual_tau_short is not None else g["tau_short"]
            else:
                tau_short_fixed = None
            fit = fit_pixels_reconv(cube, irf_aligned, d["tpb"] * 1e9, mask, tau_short_fixed,
                                    tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                                    cancel_check=cancel_check, no_shift=no_shift)
    elif fit_method == "mean_irf":
        if args.mono:
            print("  --mono has no effect with fit_method=mean_irf (there's only one lifetime to compute); ignoring.")
        if args.fix_tau_short:
            print("  --fix-tau-short has no effect with fit_method=mean_irf; ignoring.")
        fit = mean_arrival_time_reconv(cube, irf_aligned, d["tpb"] * 1e9, mask, no_shift=no_shift)
        tlv = fit["tau_mean"][np.isfinite(fit["tau_mean"])]
        if tlv.size:
            print(f"  tau_mean (IRF-corrected): median {np.median(tlv):.2f}  "
                  f"IQR[{np.percentile(tlv,25):.2f}-{np.percentile(tlv,75):.2f}] ns")
        if getattr(args, "save_results", True):
            plot_results(img, mask, removed, fit["tau_mean"], threshold, min_neighbors,
                         f"{gprefix}_results.png", not args.no_show, bin_size=args.bin,
                         label="tau_mean_irf", symbol="\u27e8t\u27e9")
        if getattr(args, "save_quality_plots", True):
            plot_fit_quality(fit["tau_mean_std"], fit["n_photons"], mask, f"{gprefix}_quality.png",
                             not args.no_show, symbol="\u27e8t\u27e9")
        np.savez(f"{gprefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, removed_isolated=removed,
                 tau_mean_ns=fit["tau_mean"], tau_mean_std_ns=fit["tau_mean_std"],
                 n_photons_fit=fit["n_photons"], irf_shift=fit["shift"], group=g_idx, n_groups=n_groups,
                 phase_desc=phase_desc, threshold=threshold, min_neighbors=min_neighbors,
                 bin_size=args.bin, fit_method=fit_method)
        print(f"  saved {gprefix}_maps.npz")
        summary_val, summary_iqr = (np.median(tlv), (np.percentile(tlv,25), np.percentile(tlv,75))) if tlv.size else (np.nan, (np.nan, np.nan))
        return dict(group=g_idx, median=summary_val, iqr=summary_iqr, n_pixels=int(mask.sum()))
    elif fit_method == "mean":
        if args.mono:
            print("  --mono has no effect with fit_method=mean (there's only one lifetime to compute); ignoring.")
        if args.fix_tau_short:
            print("  --fix-tau-short has no effect with fit_method=mean; ignoring.")
        s_like = dict(B0=d["B0"], B1=d["B1"], tpb=d["tpb"])
        summed, peak_bin, s_idx, tn = _tail_axis(s_like, cube, offset=getattr(args, "tail_start_offset", None))
        fit = mean_arrival_time(cube, s_idx, tn, mask)
        tlv = fit["tau_mean"][np.isfinite(fit["tau_mean"])]
        if tlv.size:
            print(f"  tau_mean: median {np.median(tlv):.2f}  "
                  f"IQR[{np.percentile(tlv,25):.2f}-{np.percentile(tlv,75):.2f}] ns")
        if getattr(args, "save_results", True):
            plot_results(img, mask, removed, fit["tau_mean"], threshold, min_neighbors,
                         f"{gprefix}_results.png", not args.no_show, bin_size=args.bin,
                         label="tau_mean", symbol="\u27e8t\u27e9")
        if getattr(args, "save_quality_plots", True):
            plot_fit_quality(fit["tau_mean_std"], fit["n_photons"], mask, f"{gprefix}_quality.png",
                             not args.no_show, symbol="\u27e8t\u27e9")
        np.savez(f"{gprefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, removed_isolated=removed,
                 tau_mean_ns=fit["tau_mean"], tau_mean_std_ns=fit["tau_mean_std"],
                 n_photons_fit=fit["n_photons"], group=g_idx, n_groups=n_groups,
                 phase_desc=phase_desc, threshold=threshold, min_neighbors=min_neighbors,
                 bin_size=args.bin, fit_method=fit_method)
        print(f"  saved {gprefix}_maps.npz")
        summary_val, summary_iqr = (np.median(tlv), (np.percentile(tlv,25), np.percentile(tlv,75))) if tlv.size else (np.nan, (np.nan, np.nan))
        return dict(group=g_idx, median=summary_val, iqr=summary_iqr, n_pixels=int(mask.sum()))
    else:
        # Minimal stand-in for the "s" metadata dict that _tail_axis/global_fit expect.
        s_like = dict(B0=d["B0"], B1=d["B1"], tpb=d["tpb"])
        summed, peak_bin, s_idx, tn = _tail_axis(s_like, cube, offset=getattr(args, "tail_start_offset", None))
        g = global_fit(summed, s_like, s_idx, tn, mono=args.mono,
                       tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                       tau_mono_bounds=tau_mono_bounds)
        if args.mono:
            print(f"  global fit (mono): tau = {g['tau']:.2f} ns")
            fit = fit_pixels_mono(cube, s_idx, tn, mask, tau_mono_bounds=tau_mono_bounds,
                                  cancel_check=cancel_check)
        else:
            print(f"  global fit: tau_short = {g['tau_short']:.2f} ns, tau_long = {g['tau_long']:.2f} ns")
            if args.fix_tau_short:
                tau_short_fixed = manual_tau_short if manual_tau_short is not None else g["tau_short"]
            else:
                tau_short_fixed = None
            fit = fit_pixels(cube, s_idx, tn, mask, tau_short_fixed,
                             tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                             cancel_check=cancel_check)

    n_converged = int(fit["converged"][mask].sum())
    print(f"  optimizer converged for {n_converged}/{int(mask.sum())} fitted pixels")

    if args.mono:
        if args.fix_tau_short:
            print("  --fix-tau-short has no effect with --mono; ignoring.")
        tau_map = fit["tau"]
        tlv = tau_map[np.isfinite(tau_map)]
        if tlv.size:
            print(f"  tau: median {np.median(tlv):.2f}  "
                  f"IQR[{np.percentile(tlv,25):.2f}-{np.percentile(tlv,75):.2f}] ns")
            std_v = fit["tau_std"][np.isfinite(fit["tau_std"])]
            if std_v.size:
                print(f"  tau_std: median {np.median(std_v):.3f} ns")
        if getattr(args, "save_results", True):
            plot_results(img, mask, removed, tau_map, threshold, min_neighbors,
                         f"{gprefix}_results.png", not args.no_show, bin_size=args.bin,
                         label="tau", symbol="\u03c4")
        if getattr(args, "save_quality_plots", True):
            plot_fit_quality(fit["tau_std"], fit["n_photons"], mask, f"{gprefix}_quality.png",
                             not args.no_show, symbol="\u03c4", converged_map=fit["converged"])
        np.savez(f"{gprefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, removed_isolated=removed,
                 tau_ns=tau_map, tau_std_ns=fit["tau_std"], n_photons_fit=fit["n_photons"],
                 converged=fit["converged"], nll=fit["nll"],
                 model="mono", group=g_idx, n_groups=n_groups,
                 phase_desc=phase_desc, threshold=threshold, min_neighbors=min_neighbors,
                 bin_size=args.bin, tau_global=g["tau"], tau_global_std=g.get("tau_std", np.nan),
                 fit_method=fit_method)
        summary_val, summary_iqr = (np.median(tlv), (np.percentile(tlv,25), np.percentile(tlv,75))) if tlv.size else (np.nan, (np.nan, np.nan))
    else:
        tau_long, tau_short = fit["tau_long"], fit["tau_short"]
        tlv = tau_long[np.isfinite(tau_long)]
        if tlv.size:
            print(f"  tau_long: median {np.median(tlv):.2f}  "
                  f"IQR[{np.percentile(tlv,25):.2f}-{np.percentile(tlv,75):.2f}] ns")
            std_v = fit["tau_long_std"][np.isfinite(fit["tau_long_std"])]
            if std_v.size:
                print(f"  tau_long_std: median {np.median(std_v):.3f} ns")
        if getattr(args, "save_results", True):
            plot_results(img, mask, removed, tau_long, threshold, min_neighbors,
                         f"{gprefix}_results.png", not args.no_show, bin_size=args.bin)
        if getattr(args, "save_quality_plots", True):
            plot_fit_quality(fit["tau_long_std"], fit["n_photons"], mask, f"{gprefix}_quality.png",
                             not args.no_show, symbol="\u03c4_long", converged_map=fit["converged"])
        np.savez(f"{gprefix}_maps.npz",
                 photons_per_pixel=img, mask=mask, removed_isolated=removed,
                 tau_long_ns=tau_long, tau_short_ns=tau_short,
                 tau_long_std_ns=fit["tau_long_std"], tau_short_std_ns=fit["tau_short_std"],
                 n_photons_fit=fit["n_photons"], converged=fit["converged"], nll=fit["nll"],
                 model="bi", group=g_idx, n_groups=n_groups, phase_desc=phase_desc,
                 threshold=threshold, min_neighbors=min_neighbors, bin_size=args.bin,
                 tau_short_global=g["tau_short"], tau_long_global=g["tau_long"],
                 tau_short_global_std=g.get("tau_short_std", np.nan),
                 tau_long_global_std=g.get("tau_long_std", np.nan),
                 fit_method=fit_method)
        summary_val, summary_iqr = (np.median(tlv), (np.percentile(tlv,25), np.percentile(tlv,75))) if tlv.size else (np.nan, (np.nan, np.nan))

    print(f"  saved {gprefix}_maps.npz")
    return dict(group=g_idx, median=summary_val, iqr=summary_iqr, n_pixels=int(mask.sum()))


def global_fit_fixed_short(summed, s, s_idx, tn, tau_short_fixed, tau_long_bounds=None):
    """Tail fit with tau_short pinned to a known value; only tau_long floats.
    global_fit() itself has no fixed-tau_short mode (that's only offered for
    the main script's per-pixel fit_pixels, since a global/pooled fit
    normally has plenty of photons for a free fit already) -- but a single
    time window here can still be photon-limited for a short --step, so this
    mirrors that same stabilization for the pooled-curve fit."""
    _, tau_long_bounds, _ = _resolve_bounds(None, tau_long_bounds, None)
    yv = summed[s_idx:].astype(float)

    def model(t, A1, A2, t2, C):
        return A1 * np.exp(-t / tau_short_fixed) + A2 * np.exp(-t / t2) + C

    p0 = [yv.max() * 0.6, yv.max() * 0.4, sum(tau_long_bounds) / 2, yv.min() + 1]
    popt, pcov = curve_fit(model, tn, yv, p0=p0, sigma=np.sqrt(np.maximum(yv, 1)),
                        bounds=([0, 0, tau_long_bounds[0], 0],
                                [np.inf, np.inf, tau_long_bounds[1], np.inf]),
                        maxfev=20000)
    A1, A2, t2, C = popt
    return dict(tau_short=tau_short_fixed, tau_long=t2, tau_long_std=_curve_fit_std(pcov, 2),
               popt=popt, model=model, yv=yv, mono=False)


def fit_window_sum(cube, mask, s_like, mono, tau_short_fixed=None, offset=None,
                   tau_short_bounds=None, tau_long_bounds=None, tau_mono_bounds=None):
    """Pool every masked super-pixel's decay histogram into one combined
    curve, then fit that single curve (mono, free biexponential, or
    biexponential with tau_short pinned). Returns a result dict (from
    global_fit / global_fit_fixed_short) plus the photon count used, or
    (None, 0) if the mask has no pixels within this window's cube."""
    if mask.sum() == 0:
        return None, 0
    summed = cube[mask].sum(axis=0).astype(float)
    n_photons = int(summed.sum())
    if n_photons == 0:
        return None, 0
    _, peak_bin, s_idx, tn = _tail_axis(s_like, cube[mask].sum(axis=0)[None, None, :], offset=offset)
    if mono:
        g = global_fit(summed, s_like, s_idx, tn, mono=True, tau_mono_bounds=tau_mono_bounds)
    elif tau_short_fixed is not None:
        g = global_fit_fixed_short(summed, s_like, s_idx, tn, tau_short_fixed, tau_long_bounds=tau_long_bounds)
    else:
        g = global_fit(summed, s_like, s_idx, tn, mono=False,
                       tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds)
    return g, n_photons


def fit_window_sum_reconv(cube, mask, irf_aligned, tpb_ns, mono, tau_short_fixed=None,
                          tau_short_bounds=None, tau_long_bounds=None, tau_mono_bounds=None, no_shift=False):
    """Reconvolution equivalent of fit_window_sum: pool every masked
    super-pixel's decay histogram into one combined curve, then fit the
    FULL curve (not just the tail) against the IRF-convolved model. No
    tail-start offset involved -- reconvolution uses every photon. Returns
    a result dict (from global_reconv_fit / _mono / _fixed_short) plus the
    photon count used, or (None, 0) if the mask is empty.

    no_shift: see flim_core.irf.global_reconv_fit_mono -- fixes the IRF/
    data timing offset at 0 instead of fitting it. The caller is
    responsible for passing an already-correctly-prepared irf_aligned when
    no_shift is True (i.e. NOT pre-shifted via estimate_shift+shift_array,
    since that would reintroduce an offset through the back door -- see
    analyses/normal.py's process_file for the pattern)."""
    if mask.sum() == 0:
        return None, 0
    summed = cube[mask].sum(axis=0).astype(float)
    n_photons = int(summed.sum())
    if n_photons == 0:
        return None, 0
    if mono:
        g = global_reconv_fit_mono(summed, irf_aligned, tpb_ns, tau_mono_bounds=tau_mono_bounds, no_shift=no_shift)
    elif tau_short_fixed is not None:
        g = global_reconv_fit_fixed_short(summed, irf_aligned, tpb_ns, tau_short_fixed,
                                          tau_long_bounds=tau_long_bounds, no_shift=no_shift)
    else:
        g = global_reconv_fit(summed, irf_aligned, tpb_ns,
                              tau_short_bounds=tau_short_bounds, tau_long_bounds=tau_long_bounds,
                              no_shift=no_shift)
    return g, n_photons


def fit_window_sum_mean_arrival(cube, mask, s_like, offset=None):
    """Mean-arrival-time equivalent of fit_window_sum/fit_window_sum_reconv:
    pool every masked super-pixel's decay histogram into one combined
    curve, then compute that single curve's photon-weighted mean arrival
    time (see flim_core/fitting.py's mean_arrival_time for the math) --
    no fit, no model, no bounds. Reuses mean_arrival_time itself by
    reshaping the pooled curve into a fake 1x1 "cube", the same trick
    fit_window_sum uses for _tail_axis. Returns (tau_mean, tau_mean_std,
    n_photons), or (None, None, 0) if the mask is empty."""
    if mask.sum() == 0:
        return None, None, 0
    summed = cube[mask].sum(axis=0).astype(float)
    n_photons = int(summed.sum())
    if n_photons == 0:
        return None, None, 0
    pooled_cube = summed[None, None, :]
    _, peak_bin, s_idx, tn = _tail_axis(s_like, pooled_cube, offset=offset)
    fit = mean_arrival_time(pooled_cube, s_idx, tn, np.array([[True]]))
    return float(fit["tau_mean"][0, 0]), float(fit["tau_mean_std"][0, 0]), n_photons


def fit_window_sum_mean_arrival_reconv(cube, mask, irf_aligned, tpb_ns, no_shift=False):
    """IRF-corrected equivalent of fit_window_sum_mean_arrival: pool every
    masked super-pixel's decay histogram into one combined curve, then
    compute that single curve's IRF-corrected mean arrival time (see
    flim_core/irf.py's mean_arrival_time_reconv for the math -- same
    moment-subtraction identity, same care around not clipping negative
    background-subtracted residuals and truncating to where the pooled
    signal has actually converged). Reuses mean_arrival_time_reconv itself
    by reshaping the pooled curve into a fake 1x1 "cube", the same trick
    fit_window_sum_mean_arrival uses. Returns (tau_mean, tau_mean_std,
    n_photons), or (None, None, 0) if the mask is empty.

    no_shift: see fit_window_sum_reconv -- same meaning and same caller
    responsibility for irf_aligned."""
    if mask.sum() == 0:
        return None, None, 0
    summed = cube[mask].sum(axis=0).astype(float)
    n_photons = int(summed.sum())
    if n_photons == 0:
        return None, None, 0
    pooled_cube = summed[None, None, :]
    fit = mean_arrival_time_reconv(pooled_cube, irf_aligned, tpb_ns, np.array([[True]]), no_shift=no_shift)
    return float(fit["tau_mean"][0, 0]), float(fit["tau_mean_std"][0, 0]), n_photons


def plot_pooled_mean_arrival(summed, tpb_ns, tau_mean, tau_mean_std, title, out_png, show):
    """Diagnostic for the mean-arrival-time fit method's pooled/whole-image
    plugins: the pooled decay curve (log scale, full curve) with a vertical
    line at tau_mean -- there's no fitted model curve to overlay (mean
    arrival time isn't a fit), so this is simpler than plot_window_fit."""
    nb = len(summed)
    t_ns = np.arange(nb) * tpb_ns * 1e9
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.semilogy(t_ns, summed + 1, color="black", lw=1.2, label="data (summed masked region)")
    ax.axvline(tau_mean, color="tab:red", ls="--", lw=1.2,
              label=f"tau_mean = {tau_mean:.2f} \u00b1 {tau_mean_std:.2f} ns")
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Photon counts")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"  saved {out_png}")
    if show:
        plt.show(block=False); plt.pause(0.1)
    plt.close(fig)


def plot_window_fit_reconv(summed, tpb, g, mono, title, out_png, show):
    """Reconvolution diagnostic: the window's pooled decay curve (log
    scale, FULL curve including the rise -- not just the tail) with the
    fitted reconvolution model overlaid. Mirrors plot_window_fit, but the
    reconvolution model's calling convention is model(popt) -> whole curve,
    not model(t, *popt) -> tail only, so this needs its own plotting logic."""
    nb = len(summed)
    t_ns = np.arange(nb) * tpb * 1e9
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.semilogy(t_ns, summed + 1, color="black", lw=1.2, label="data (summed masked region)")
    if mono:
        fit_label = f"fit (\u03c4={g['tau']:.2f} ns)"
    else:
        fit_label = f"fit (\u03c4s={g['tau_short']:.2f}, \u03c4l={g['tau_long']:.2f} ns)"
    ax.semilogy(t_ns, g["model"](g["popt"]) + 1, color="tab:red", lw=1.2, ls="--", label=fit_label)
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Photon counts")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"  saved {out_png}")
    if show:
        plt.show(block=False); plt.pause(0.1)
    plt.close(fig)


def plot_window_fit(summed, s_idx, tn, g, mono, title, out_png, show):
    """Diagnostic: the window's pooled decay curve (log scale) with the
    fitted model overlaid -- the per-window equivalent of a results plot,
    since there's no spatial map to show in this summed-ROI approach."""
    yv = summed[s_idx:].astype(float)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.semilogy(tn, yv + 1, color="black", lw=1.2, label="data (summed masked region)")
    if mono:
        fit_label = f"fit (\u03c4={g['tau']:.2f} ns)"
    else:
        fit_label = f"fit (\u03c4s={g['tau_short']:.2f}, \u03c4l={g['tau_long']:.2f} ns)"
    ax.semilogy(tn, g["model"](tn, *g["popt"]), color="tab:red", lw=1.2, ls="--", label=fit_label)
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("Photon counts")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"  saved {out_png}")
    if show:
        plt.show(block=False); plt.pause(0.1)
    plt.close(fig)


def nice_tick_interval(total_span, target_ticks=8):
    """Pick a round tick interval (from a standard set of 'nice' step sizes)
    so the axis ends up with roughly target_ticks ticks, instead of forcing
    exactly one tick per data point. Works for both short (sub-second) and
    long (many-minute) acquisitions."""
    if total_span <= 0:
        return 1
    raw = total_span / target_ticks
    candidates = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 30,
                  60, 120, 300, 600, 900, 1200, 1800, 3600]
    for c in candidates:
        if c >= raw:
            return c
    return candidates[-1]


def save_trend_npz(results, window_t0, window_t1, x, mono, out_npz, fit_method="tail"):
    """Save the exact per-window numbers behind the trend plot -- elapsed
    time, photon count, and the fitted lifetime(s) (or mean arrival time)
    -- as an npz, for later reloading/comparison (e.g. with
    flimera_flim_analysis_compare.py-style tooling, or plain numpy)."""
    n = len(results)
    n_photons = np.array([r["n_photons"] if r is not None else 0 for r in results])
    if fit_method in ("mean", "mean_irf"):
        tau_mean = np.array([r["val"] if r is not None else np.nan for r in results])
        tau_mean_std = np.array([r.get("tau_mean_std", np.nan) if r is not None else np.nan for r in results])
        np.savez(out_npz, window=np.arange(1, n + 1), t_start_s=np.array(window_t0),
                 t_end_s=np.array(window_t1), t_center_s=x, n_photons=n_photons,
                 tau_mean_ns=tau_mean, tau_mean_std_ns=tau_mean_std, fit_method=fit_method)
    elif mono:
        tau = np.array([r["g"]["tau"] if r is not None else np.nan for r in results])
        np.savez(out_npz, window=np.arange(1, n + 1), t_start_s=np.array(window_t0),
                 t_end_s=np.array(window_t1), t_center_s=x, n_photons=n_photons,
                 tau_ns=tau, model="mono")
    else:
        tau_short = np.array([r["g"]["tau_short"] if r is not None else np.nan for r in results])
        tau_long = np.array([r["g"]["tau_long"] if r is not None else np.nan for r in results])
        np.savez(out_npz, window=np.arange(1, n + 1), t_start_s=np.array(window_t0),
                 t_end_s=np.array(window_t1), t_center_s=x, n_photons=n_photons,
                 tau_short_ns=tau_short, tau_long_ns=tau_long, model="bi")
    print(f"Saved trend data: {out_npz}")
