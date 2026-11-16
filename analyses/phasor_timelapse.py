#!/usr/bin/env python3
"""
Flimera / EzTime FLIM analysis -- Phasor timelapse (pixel-wise)

Analysis type: time-resolved, pixel-wise phasor (G, S) -- model-free DFT of
each pixel's histogram at the laser repetition frequency, no per-pixel fit.
Fit method: none per pixel; the window follows fit_method exactly as in
analyses/phasor.py ("tail" -> tail window, "irf" -> full window + IRF
correction).

Splits the acquisition into consecutive time windows (step, optional rolling
window -- same windowing as analyses/timelapse.py) and computes a phasor map
per window, plus ONE pooled phasor per window (all masked photons summed --
the whole-image value, robust even when single pixels have few photons).

Per window, depending on decay_model (see analyses/phasor.py for the maths):
  bi + tau_ref1/tau_ref2 -> f_long map and pooled f_long
  single                 -> tau_single + off_curve maps, pooled tau_single
  bi without refs        -> G/S only (tau_phase is a lifetime only in IRF mode)

Computed ONCE from the whole acquisition (high SNR), then reused for every
window so all windows are analysed identically:
  - the pixel mask (fixed pixel set, so the time course follows the same
    pixels). Threshold = photons per pixel per OUTPUT window, averaged over
    the acquisition; or an external mask.
  - time axes, decay peak / tail start, decay start t0, reference phasors and
    the single-lifetime curve.
  - the IRF/data timing offset (one pooled reconvolution fit).
  - the background: per pixel (two-lifetime mode, linear tail fit) or one
    image-wide level (single-lifetime mode), measured on the whole
    acquisition and scaled to each window by its duration. Assumes the
    background RATE (dark counts, ambient light) is constant in time; a
    change in ambient light during the run would violate that.

Pixel-offset correction is applied to the photons (integer bins, as in
timelapse.py). The 8-bin pattern correction is not supported here: it is
defined on raw microtime bins and would have to be applied before the
per-pixel shifts, which the photon-level correction does not allow.

REQUIRES a RAW photon-stream file.

Outputs
-------
  <prefix>_overview.png             whole-acquisition overview + mask
  <prefix>_phasor_trajectory.png    pooled phasor per window (time-coloured)
                                    over the density of all pixel phasors
  <prefix>_phasor_timecourse.png    main quantity vs time (pixel median + IQR,
                                    pooled value), off_curve, photons
  <prefix>_phasor_frames.png        main-quantity map per window, shared scale
  <prefix>_maps.npz                 stacks [n_windows, H, W] + per-window traces

Usage
-----
  python -m analyses.phasor_timelapse data.H5 --step 1 --decay-model single
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flim_core.config import MIN_NEIGHBORS, DEFAULT_THRESHOLD, BIN_SIZE
from flim_core.masking import clean_mask, load_mask, report_dead_pixels
from flim_core.fitting import AnalysisCancelled, _tail_axis
from flim_core.plotting import plot_overview_and_mask
from analyses._common import (
    window_boundaries, apply_spatial_and_time_binning, load_raw_with_time,
    build_group_cube, crop_photon_data, apply_offset_to_photons,
)
from analyses.timelapse import assign_time_windows
from analyses import phasor as ph

NAME = "Phasor timelapse (pixel-wise)"
DESCRIPTION = ("Phasor (G, S) per time window: f_long or single-lifetime tau maps over time, "
               "plus a pooled whole-image phasor per window. No per-pixel fitting.")
REQUIRES_RAW = True
IRF_CAPABLE = True
EXTRA_ARGS = ["step", "rolling_window"] + list(ph.EXTRA_ARGS)
GUI_OPTIONS = dict(ph.GUI_OPTIONS, uses_bin_pattern=False)


def preview_slice(loaded_data, settings):
    """First time window, for the GUI's mask-preview panel."""
    step_s = float(settings["extra"]["step"])
    if step_s <= 0:
        raise ValueError("Step size must be greater than 0.")
    window_idx, n_win = assign_time_windows(loaded_data["time_ms"], step_s * 1000.0,
                                            loaded_data["duration_ms"])
    b = window_boundaries(window_idx, n_win)
    img, cube = build_group_cube(loaded_data, slice(b[0], b[1]))
    return img, cube, "first time window"


# ---------------------------------------------------------------------------
# Per-window phasor
# ---------------------------------------------------------------------------

class _Setup:
    """Everything fixed for the whole run (see module docstring)."""
    pass


def _window_phasor(cube_w, bg_w, su):
    """Phasor of one window: per-pixel maps and the pooled (whole-mask) value.
    Returns (maps dict, pooled dict)."""
    mask = su.mask
    cube_win = cube_w[:, :, su.win_slice]
    r = ph.compute_phasor(cube_win, su.t_abs_win, su.T, mask, harmonic=su.harmonic,
                          bg_per_bin=bg_w)
    # pooled: every masked photon in one histogram (background summed alike)
    pooled = cube_win[mask].sum(axis=0)[None, None, :].astype(np.float64)
    bg_pooled = None if bg_w is None else np.array([[bg_w[mask].sum()]])
    rp = ph.compute_phasor(pooled, su.t_abs_win, su.T, np.ones((1, 1), bool),
                           harmonic=su.harmonic, bg_per_bin=bg_pooled)
    for res in (r, rp):
        res["irf_corrected"] = False
        if su.irf_shift is not None:
            _quiet(ph.apply_irf_correction, res, su.irf_grid, su.tpb_ns, su.irf_shift,
                   harmonic=su.harmonic)

    maps = dict(G=r["G"], S=r["S"], n_photons=r["n_photons"])
    pooled_out = dict(G=float(rp["G"][0, 0]), S=float(rp["S"][0, 0]),
                      n_photons=float(rp["n_photons"][0, 0]))
    if su.decay_model == "single":
        ts, oc = ph.project_on_curve(r["G"], r["S"], mask, *su.curve)
        tsp, ocp = ph.project_on_curve(rp["G"], rp["S"], np.ones((1, 1), bool), *su.curve)
        maps.update(tau_single=ts, off_curve=oc)
        pooled_out.update(tau_single=float(tsp[0, 0]), off_curve=float(ocp[0, 0]))
    elif su.refs is not None:
        G1, S1, w1, G2, S2, w2 = su.refs
        fw = ph.fraction_long(r["G"], r["S"], G1, S1, G2, S2)
        fwp = ph.fraction_long(rp["G"], rp["S"], G1, S1, G2, S2)
        maps.update(f_long=ph._window_to_total_fraction(fw, w1, w2), f_long_window=fw)
        pooled_out.update(f_long=float(ph._window_to_total_fraction(fwp, w1, w2)[0, 0]))
    else:
        maps.update(tau_phase=r["tau_phase"].astype(np.float32))
        pooled_out.update(tau_phase=float(rp["tau_phase"][0, 0]))
    return maps, pooled_out


def _quiet(fn, *a, **kw):
    """Call fn with its print() output suppressed (per-window repeats)."""
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _primary(su):
    """(key, label, unit, fixed range or None) of the quantity tracked over time."""
    if su.decay_model == "single":
        return "tau_single", "τ_single", "ns", None
    if su.refs is not None:
        return "f_long", f"f_long (τ={su.tau_ref2} ns)", "intensity fraction", (0.0, 1.0)
    return "tau_phase", "τ_phase", "ns", None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_trajectory(su, stacks, pooled, t_mid, prefix):
    m = su.mask
    G_all = stacks["G"][:, m].ravel()
    S_all = stacks["S"][:, m].ravel()
    ok = np.isfinite(G_all) & np.isfinite(S_all)
    G_all, S_all = G_all[ok], S_all[ok]
    omega = 2 * np.pi * su.harmonic / su.T

    fig, ax = plt.subplots(figsize=(7, 5.4))
    th = np.linspace(0, np.pi, 400)
    ax.plot(0.5 + 0.5 * np.cos(th), 0.5 * np.sin(th), color="#cccccc", lw=1.3,
            label="universal semicircle")
    if su.curve is not None:
        c_tau, c_G, c_S = su.curve
        ax.plot(c_G, c_S, color="#7B1FA2", lw=1.2, label="single-lifetime curve (this window)")
        for tau in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
            Gt = np.interp(np.log(tau), np.log(c_tau), c_G)
            St = np.interp(np.log(tau), np.log(c_tau), c_S)
            ax.plot(Gt, St, "o", ms=2.5, color="#7B1FA2")
            ax.annotate(f"{tau:g}", (Gt, St), textcoords="offset points", xytext=(3, 2),
                        fontsize=6, color="#7B1FA2")
    if su.refs is not None:
        G1, S1, _, G2, S2, _ = su.refs
        ax.plot([G1, G2], [S1, S2], "--", color="#E05C3A", lw=1.3,
                label=f"tie-line τ={su.tau_ref1}–{su.tau_ref2} ns")
    if su.G_ref is not None:
        ax.plot(su.G_ref, su.S_ref, "D", color="#E05C3A", ms=7, label=f"τ_ref = {su.tau_ref} ns")

    if len(G_all) > 1:
        g_lo, g_hi = np.percentile(G_all, [0.5, 99.5])
        s_lo, s_hi = np.percentile(S_all, [0.5, 99.5])
        if g_hi > g_lo and s_hi > s_lo:
            h2d, xe, ye = np.histogram2d(G_all, S_all, bins=120, range=[[g_lo, g_hi], [s_lo, s_hi]])
            ax.pcolormesh(xe, ye, np.where(h2d > 0, np.log1p(h2d), np.nan).T,
                          cmap="Greys", alpha=0.6, zorder=1)
    else:
        g_lo = g_hi = s_lo = s_hi = np.nan

    Gp = np.array([p["G"] for p in pooled])
    Sp = np.array([p["S"] for p in pooled])
    ax.plot(Gp, Sp, "-", color="#555555", lw=0.8, zorder=3)
    sc = ax.scatter(Gp, Sp, c=t_mid, cmap="viridis", s=28, zorder=4, edgecolors="k",
                    linewidths=0.3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("time (s)")

    pts_g = [Gp.min() - 0.03, Gp.max() + 0.03] + ([g_lo, g_hi] if np.isfinite(g_lo) else [])
    pts_s = [Sp.min() - 0.03, Sp.max() + 0.03] + ([s_lo, s_hi] if np.isfinite(s_lo) else [])
    pad = 0.08
    ax.set_xlim(min(pts_g) - pad, max(pts_g) + pad)
    ax.set_ylim(max(-0.02, min(pts_s) - pad), max(pts_s) + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("G")
    ax.set_ylabel("S")
    irf_tag = " + IRF corr." if su.irf_shift is not None else ""
    ax.set_title(f"Pooled phasor per window (dots, coloured by time)\n"
                 f"over all pixel phasors (grey)  —  harmonic {su.harmonic}, "
                 f"T = {su.T:.2f} ns{irf_tag}", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    fig.tight_layout()
    out = f"{prefix}_phasor_trajectory.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_timecourse(su, stacks, pooled, t_mid, t_lo, t_hi, prefix):
    key, label, unit, _ = _primary(su)
    m = su.mask
    vals = stacks[key][:, m]
    med = np.nanmedian(vals, axis=1)
    q1, q3 = np.nanpercentile(vals, [25, 75], axis=1)
    pooled_v = np.array([p[key] for p in pooled])
    n_ph = np.array([p["n_photons"] for p in pooled])

    rows = 3 if su.decay_model == "single" else 2
    fig, axes = plt.subplots(rows, 1, figsize=(8, 2.6 * rows + 0.6), sharex=True)
    ax = axes[0]
    ax.fill_between(t_mid, q1, q3, color="#3477c4", alpha=0.2, label="pixels: IQR")
    ax.plot(t_mid, med, "o-", color="#3477c4", ms=4, lw=1.2, label="pixels: median")
    ax.plot(t_mid, pooled_v, "s--", color="#E05C3A", ms=4, lw=1.0, label="pooled (whole mask)")
    ax.set_ylabel(f"{label} ({unit})" if unit != "intensity fraction" else label)
    if key == "tau_phase" and su.window_mode == "tail":
        ax.set_title("τ_phase in a tail window is NOT a lifetime -- relative changes only",
                     fontsize=9, color="#a55")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    i = 1
    if su.decay_model == "single":
        oc = stacks["off_curve"][:, m]
        ax = axes[i]; i += 1
        ax.axhline(0, color="k", lw=0.6)
        ax.fill_between(t_mid, *np.nanpercentile(oc, [25, 75], axis=1), color="#7B1FA2", alpha=0.2)
        ax.plot(t_mid, np.nanmedian(oc, axis=1), "o-", color="#7B1FA2", ms=4, lw=1.2,
                label="pixels: median (IQR band)")
        ax.plot(t_mid, [p["off_curve"] for p in pooled], "s--", color="#E05C3A", ms=4, lw=1.0,
                label="pooled")
        ax.set_ylabel("off_curve\n(+ = inner side)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    ax = axes[i]
    ax.plot(t_mid, n_ph, "o-", color="#444444", ms=3, lw=1.0)
    ax.set_ylabel("photons in mask\nper window")
    ax.set_xlabel("time (s, window centre)")
    ax.grid(alpha=0.3)
    fig.suptitle("Phasor timelapse", fontsize=11)
    fig.tight_layout()
    out = f"{prefix}_phasor_timecourse.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_frames(su, stacks, t_lo, t_hi, prefix, max_frames=48):
    key, label, unit, fixed = _primary(su)
    data = stacks[key]
    n = data.shape[0]
    idx = np.arange(n) if n <= max_frames else np.unique(np.linspace(0, n - 1, max_frames).astype(int))
    vals = data[:, su.mask]
    vals = vals[np.isfinite(vals)]
    if fixed is not None:
        vmin, vmax = fixed
    elif vals.size:
        vmin, vmax = np.percentile(vals, [2, 98])
    else:
        vmin, vmax = 0, 1
    ys, xs = np.where(su.mask)
    pad = 3
    y0, y1 = max(0, ys.min() - pad), min(su.mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(su.mask.shape[1], xs.max() + pad + 1)
    aspect = (y1 - y0) / max(1, (x1 - x0))
    ncol = min(len(idx), 4 if aspect < 0.5 else 6)      # wide, flat crops get fewer, larger columns
    nrow = int(np.ceil(len(idx) / ncol))
    panel_w = 3.2 if aspect < 0.5 else 2.4
    fig, axes = plt.subplots(nrow, ncol, figsize=(panel_w * ncol + 1.2,
                                                  max(0.9, panel_w * aspect) * nrow + 1.0),
                             squeeze=False)
    cmap = "RdYlBu_r" if key == "f_long" else "plasma"
    im = None
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k >= len(idx):
            continue
        w = idx[k]
        img = np.where(su.mask, data[w], np.nan)[y0:y1, x0:x1]
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{t_lo[w]:.1f}–{t_hi[w]:.1f} s", fontsize=7)
    if im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
        cb.set_label(f"{label} ({unit})" if unit != "intensity fraction" else label)
    note = "" if len(idx) == n else f"  ({len(idx)} of {n} windows shown, evenly spaced)"
    fig.suptitle(f"{label} per window, shared colour scale{note}", fontsize=10)
    out = f"{prefix}_phasor_frames.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    """One raw file per call (args.h5file). Exceptions propagate."""
    ea = ph._extra_args(args)
    step_s = getattr(args, "step", None)
    try:
        step_s = float(step_s)
    except (TypeError, ValueError):
        raise ValueError(f"step = {step_s!r}: give the window length in seconds")
    if step_s <= 0:
        raise ValueError("Step size must be greater than 0.")
    rw = max(1, int(getattr(args, "rolling_window", 1) or 1))
    cancel_check = getattr(args, "cancel_check", None)

    su = _Setup()
    su.harmonic, su.window_mode, su.decay_model = ea["harmonic"], ea["window_mode"], ea["decay_model"]
    su.tau_ref, su.tau_ref1, su.tau_ref2 = ea["tau_ref"], ea["tau_ref1"], ea["tau_ref2"]

    stem = Path(args.h5file).stem
    prefix = f"{args.prefix}_{stem}" if getattr(args, "prefix", None) else stem

    # ---- IRF ----
    irf = None
    if getattr(args, "irf", None) and su.window_mode == "full":
        from flim_core.irf import load_irf
        print(f"Loading IRF from {args.irf} ...")
        irf = load_irf(args.irf, window=not getattr(args, "irf_full", False),
                       window_end_ns=getattr(args, "irf_window_end_ns", 15.0),
                       time_bin=getattr(args, "time_bin", None) or 1,
                       pixel_offset_path=getattr(args, "pixel_offset", None))
        print(f"  IRF: {irf['source']} | FWHM {irf['fwhm_bins']} bins "
              f"({irf['fwhm_bins']*irf['tpb']*1e9:.2f} ns)")
    elif getattr(args, "irf", None):
        print("  NOTE: IRF file ignored -- IRF correction needs the full window (fit_method='irf').")

    # ---- photons ----
    print(f"Loading raw photon stream from {args.h5file} ...")
    d = load_raw_with_time(args.h5file)
    print(f"  {d['n_photons']} photons | {d['W']}x{d['H']} | {d['n_frames']} frames | "
          f"{d['duration_ms']/1000:.3f} s")
    roi = getattr(args, "roi", None)
    if roi is not None:
        d, roi = crop_photon_data(d, roi)
    if getattr(args, "pixel_offset", None):
        print(f"Applying photon-level pixel offset correction from "
              f"{Path(args.pixel_offset).name} ...")
        d = apply_offset_to_photons(d, np.load(args.pixel_offset), d["tpb"])
    if getattr(args, "bin_pattern", False):
        print("  NOTE: 8-bin pattern correction is not supported in the phasor timelapse "
              "(see module docstring) -- ignored.")

    # ---- windows (same scheme as timelapse.py) ----
    step_ms = step_s * 1000.0
    window_idx, n_windows = assign_time_windows(d["time_ms"], step_ms, d["duration_ms"])
    boundaries = window_boundaries(window_idx, n_windows)
    rw_half, rw_tail = rw // 2, rw - rw // 2
    w_first, w_last = rw_half, n_windows - rw_tail + 1
    n_out = max(0, w_last - w_first)
    if n_out == 0:
        raise ValueError(f"rolling window ({rw}) is larger than the number of windows ({n_windows}).")
    dur_s = d["duration_ms"] / 1000.0
    t_lo = np.array([(w - rw_half) * step_s for w in range(w_first, w_last)])
    t_hi = np.array([min((w + rw_tail) * step_s, dur_s) for w in range(w_first, w_last)])
    t_mid = np.array([(w + 0.5) * step_s for w in range(w_first, w_last)])
    t_mid = np.minimum(t_mid, dur_s)
    print(f"\nStep = {step_s} s over {dur_s:.3f} s -> {n_windows} windows"
          + (f", rolling window {rw} -> {n_out} output windows" if rw > 1 else ""))

    # ---- whole acquisition: mask, axes, IRF offset, background ----
    img_all, cube_all = build_group_cube(d, slice(None))
    report_dead_pixels(img_all)
    # spatial smoothing first, time binning separately: the unbinned cube is
    # needed for the IRF offset fit (see ph.irf_offset_binned)
    time_bin = getattr(args, "time_bin", None) or 1
    args_fine = copy.copy(args)
    args_fine.time_bin = 1
    img_all, cube_fine, _d, args_b, _ = apply_spatial_and_time_binning(
        img_all.astype(np.float64), cube_all.astype(np.float64), d, args_fine)
    del cube_all
    from flim_core.config import TAIL_START_OFFSET
    from flim_core.masking import bin_time_cube
    tail_offset = getattr(args, "tail_start_offset", None)
    tail_offset = TAIL_START_OFFSET if tail_offset is None else tail_offset
    tpb_fine_ns = d["tpb"] * 1e9
    if time_bin > 1:
        cube_all, tpb_b = bin_time_cube(cube_fine, time_bin, d["tpb"])
        tail_offset = max(1, round(tail_offset / time_bin))   # keep the ns margin constant
    else:
        cube_all, tpb_b = cube_fine, d["tpb"]
    s = dict(tpb=tpb_b, B0=d["B0"], B1=d["B0"] + cube_all.shape[2] - 1)

    per_window_img = img_all * (rw / n_windows)       # photons per pixel per OUTPUT window
    ext = getattr(args, "external_mask", None)
    if ext:
        mask = load_mask(ext)
        if mask.shape != img_all.shape:
            raise ValueError(f"external mask shape {mask.shape} does not match the image "
                             f"shape {img_all.shape} -- check bin/ROI.")
        threshold = min_nb = None
        removed = np.zeros_like(mask)
        print(f"  external mask: {int(mask.sum())} pixels (used for every window)")
    else:
        threshold = getattr(args, "threshold", None)
        threshold = DEFAULT_THRESHOLD if threshold is None else threshold
        min_nb = getattr(args, "min_neighbors", None)
        min_nb = MIN_NEIGHBORS if min_nb is None else min_nb
        raw_mask = per_window_img > threshold
        mask = clean_mask(raw_mask, min_nb)
        removed = raw_mask & ~mask
        print(f"  mask: {int(mask.sum())} pixels with > {threshold} photons per output window "
              f"(acquisition average), used for every window")
    if mask.sum() == 0:
        raise ValueError("No pixels in the mask; lower the threshold or the neighbour requirement.")
    su.mask = mask

    if getattr(args, "save_overview", True):
        plot_overview_and_mask(per_window_img, per_window_img, mask, removed, threshold, min_nb,
                               getattr(args_b, "bin", None) or 1, f"{prefix}_overview.png", False)

    nb, su.tpb_ns, su.T, t_full = ph._build_time_axes(s, cube_all, ea["rep_rate_mhz"])
    _, peak_bin, s_tail_start, _ = _tail_axis(s, cube_all, offset=tail_offset)
    su.win_slice = slice(None) if su.window_mode == "full" else slice(s_tail_start, None)
    su.t_abs_win = t_full[su.win_slice]
    print(f"  window: {su.window_mode} "
          + (f"({nb} bins)" if su.window_mode == "full" else
             f"({nb - s_tail_start} bins from bin {s_tail_start})"))

    su.irf_grid = su.irf_shift = None
    if irf is not None:
        if abs(s["tpb"] - irf["tpb"]) > 1e-15:
            print(f"  WARNING: data bin width ({s['tpb']:.3e} s) differs from the IRF's "
                  f"({irf['tpb']:.3e} s) -- the IRF phasor will not match this data.")
        su.irf_grid = ph._irf_on_data_grid(irf["irf_norm"], nb)
        su.irf_shift = ph.irf_offset_binned(args, cube_fine[mask].sum(axis=0), tpb_fine_ns,
                                            time_bin, irf_binned=irf)
        if su.irf_shift is None:
            print("  WARNING: IRF correction skipped -- the timing offset could not be measured.")
        else:
            print(f"    IRF/data timing offset: {su.irf_shift:+.2f} bins "
                  f"({su.irf_shift * su.tpb_ns * 1000:+.0f} ps)")
            if nb * su.tpb_ns < 0.99 * su.T:
                print("  WARNING: the histogram window is shorter than the laser period -- "
                      "IRF correction is approximate.")
    irf_corrected = su.irf_shift is not None
    del cube_fine

    t0_ns = 0.0 if irf_corrected else (peak_bin - s["B0"]) * su.tpb_ns   # see phasor.py
    su.curve = su.refs = None
    su.G_ref = su.S_ref = None
    if su.decay_model == "single":
        su.curve = ph.single_lifetime_curve(su.t_abs_win, t0_ns, su.T, su.harmonic,
                                            continuous=irf_corrected)
        if su.tau_ref is not None:
            su.G_ref, su.S_ref, _ = ph._reference_phasor(su.tau_ref, su.t_abs_win, t0_ns, su.T,
                                                         su.harmonic, continuous=irf_corrected)
    elif su.tau_ref1 is not None:
        G1, S1, w1 = ph._reference_phasor(su.tau_ref1, su.t_abs_win, t0_ns, su.T, su.harmonic,
                                          continuous=irf_corrected)
        G2, S2, w2 = ph._reference_phasor(su.tau_ref2, su.t_abs_win, t0_ns, su.T, su.harmonic,
                                          continuous=irf_corrected)
        su.refs = (G1, S1, w1, G2, S2, w2)
    else:
        print("  NOTE: no reference lifetimes -- tracking G/S and tau_phase only"
              + (" (tau_phase is NOT a lifetime in a tail window)." if su.window_mode == "tail" else "."))

    bg_rate = None          # counts per bin per pixel over the WHOLE acquisition
    bg_info = None
    if ea["subtract_bg"]:
        if su.decay_model == "single":
            bg_rate, bg_info = ph.uniform_background(cube_all, mask, s, tail_offset)
            if bg_rate is not None:
                print("    NOTE: single-lifetime mode takes the background as the same in every pixel.")
        else:
            bg_rate, bg_info = ph.estimate_background(
                cube_all, mask, s, tail_offset, su.T,
                taus=(su.tau_ref1, su.tau_ref2) if su.tau_ref1 is not None else None)
        if bg_rate is not None:
            print("    background measured on the whole acquisition, scaled to each window "
                  "by its duration (assumes a constant background rate).")
    else:
        print("  Background subtraction off.")
    del cube_all

    # ---- per window ----
    print(f"\n  Phasor per window: harmonic={su.harmonic}, decay model '{su.decay_model}' ...")
    stack_lists, pooled = {}, []
    for k, w in enumerate(range(w_first, w_last)):
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelled(f"Cancelled after {k}/{n_out} windows.")
        img_w, cube_w = build_group_cube(d, slice(boundaries[w - rw_half], boundaries[w + rw_tail]))
        img_w, cube_w, _, _, _ = _quiet(apply_spatial_and_time_binning,
                                        img_w.astype(np.float64), cube_w.astype(np.float64), d, args)
        bg_w = None if bg_rate is None else bg_rate * ((t_hi[k] - t_lo[k]) / dur_s)
        maps, pooled_w = _window_phasor(cube_w, bg_w, su)
        for key, arr in maps.items():
            stack_lists.setdefault(key, []).append(np.asarray(arr, dtype=np.float32))
        pooled.append(pooled_w)
        key, label, unit, _ = _primary(su)
        print(f"    window {k+1}/{n_out} ({t_lo[k]:.2f}-{t_hi[k]:.2f} s): "
              f"{int(pooled_w['n_photons'])} photons in mask, pooled {key} = {pooled_w[key]:.4g}"
              + (f", off_curve {pooled_w['off_curve']:+.4f}" if su.decay_model == "single" else ""))
    stacks = {k: np.stack(v) for k, v in stack_lists.items()}

    # ---- plots ----
    if getattr(args, "save_results", True):
        plot_trajectory(su, stacks, pooled, t_mid, prefix)
        plot_timecourse(su, stacks, pooled, t_mid, t_lo, t_hi, prefix)
        plot_frames(su, stacks, t_lo, t_hi, prefix)

    # ---- npz ----
    save = {f"{k}": v for k, v in stacks.items()}
    for key in pooled[0]:
        save[f"pooled_{key}"] = np.array([p[key] for p in pooled], dtype=np.float64)
    save.update(
        mask=mask, removed_isolated=removed, photons_per_pixel=img_all,
        t_start_s=t_lo, t_end_s=t_hi, t_mid_s=t_mid, step_s=np.float64(step_s),
        rolling_window=np.int32(rw), T_laser_ns=np.float64(su.T), harmonic=np.int32(su.harmonic),
        window_mode=su.window_mode, decay_model=su.decay_model,
        irf_corrected=np.bool_(irf_corrected), t0_ns=np.float64(t0_ns),
        s_tail_start=np.int32(s_tail_start), threshold=threshold, min_neighbors=min_nb,
        background_subtracted=np.bool_(bg_rate is not None),
    )
    if irf_corrected:
        save["irf_shift"] = np.float64(su.irf_shift)
    if bg_rate is not None:
        save["background_per_bin_whole"] = bg_rate.astype(np.float32)
        save["background_taus_ns"] = np.array(bg_info["taus"], dtype=np.float64)
    if su.curve is not None:
        c_tau, c_G, c_S = su.curve
        keep = np.unique(np.linspace(0, len(c_tau) - 1, 500).astype(int))
        save.update(curve_tau_ns=c_tau[keep], curve_G=c_G[keep], curve_S=c_S[keep])
    if su.G_ref is not None:
        save.update(tau_ref_ns=np.float64(su.tau_ref), G_ref=np.float64(su.G_ref),
                    S_ref=np.float64(su.S_ref))
    if su.refs is not None:
        G1, S1, w1, G2, S2, w2 = su.refs
        save.update(tau_ref1_ns=np.float64(su.tau_ref1), tau_ref2_ns=np.float64(su.tau_ref2),
                    G_ref1=np.float64(G1), S_ref1=np.float64(S1), G_ref2=np.float64(G2),
                    S_ref2=np.float64(S2), w_window_ref1=np.float64(w1),
                    w_window_ref2=np.float64(w2))
    npz = f"{prefix}_maps.npz"
    np.savez(npz, **save)
    print(f"  saved {npz}")
    print(f"\nDone. {n_out} windows.")


def main():
    p = argparse.ArgumentParser(description="Phasor timelapse (pixel-wise, model-free)")
    p.add_argument("h5files", nargs="+", help="RAW Flimera photon-stream .H5 file(s)")
    p.add_argument("--step", type=float, required=True, help="time window length (s)")
    p.add_argument("--rolling-window", type=int, default=1,
                   help="pool N consecutive steps per output window (default 1)")
    p.add_argument("--prefix", default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="min photons per pixel per output window (acquisition average)")
    p.add_argument("--min-neighbors", type=int, default=None)
    p.add_argument("--external-mask", default=None, help="mask .npy (overrides threshold)")
    p.add_argument("--bin", type=int, default=BIN_SIZE)
    p.add_argument("--moving-average", type=int, default=None)
    p.add_argument("--time-bin", type=int, default=1)
    p.add_argument("--roi", type=lambda v: tuple(int(x) for x in v.split(",")), default=None,
                   help="x0,y0,x1,y1")
    p.add_argument("--pixel-offset", default=None, help="pixel offset calibration .npy")
    p.add_argument("--tail-start-offset", type=int, default=None)
    p.add_argument("--fit-method", choices=["tail", "irf"], default="tail")
    p.add_argument("--irf", default=None)
    p.add_argument("--no-shift", action="store_true")
    p.add_argument("--irf-full", action="store_true")
    p.add_argument("--irf-window-end-ns", type=float, default=15.0)
    p.add_argument("--harmonic", type=int, default=1)
    p.add_argument("--rep-rate-mhz", type=float, default=None)
    p.add_argument("--no-subtract-background", dest="subtract_background", action="store_false")
    p.add_argument("--decay-model", choices=["bi", "single"], default="bi")
    p.add_argument("--tau-ref", type=float, default=None)
    p.add_argument("--tau-ref1", type=float, default=None)
    p.add_argument("--tau-ref2", type=float, default=None)
    p.add_argument("--no-overview-plot", dest="save_overview", action="store_false")
    p.add_argument("--no-results-plot", dest="save_results", action="store_false")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()
    for f in args.h5files:
        a = copy.copy(args)
        a.h5file = f
        run(a)


if __name__ == "__main__":
    main()
