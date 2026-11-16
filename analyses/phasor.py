#!/usr/bin/env python3
"""
Flimera / EzTime FLIM analysis -- Phasor (pixel-wise)

Transforms per-pixel TCSPC histograms into phasor coordinates (G, S) using
a discrete Fourier transform at the laser repetition frequency (first harmonic
by default, or higher harmonics via --harmonic).

Phasor analysis is model-free: no exponential model is assumed, no optimizer
runs, and there is no convergence to fail.  For a biexponential decay such as
Flipper-TR the phasor point lies on the straight line connecting the two
single-lifetime reference points.  Changes in tau_long move the endpoint along
its arc; changes in the amplitude ratio move the pixel's point along the
connecting tie-line.  These two effects are geometrically separable in (G, S),
which is NOT possible from mean arrival time alone.

Time window modes
-----------------
  "tail" (default): Fourier-transform only the tail bins — the same window
      as the existing tail-fit plugins, beginning a few bins after the
      decay peak.  Excludes the IRF-dominated rise region.  No IRF file
      needed.  The resulting phasor points do NOT lie exactly on the
      universal semicircle (because early photons are excluded), but they
      move monotonically and reproducibly with lifetime changes, making
      this mode ideal for relative measurements such as timelapse tracking.

  "full": use the entire TCSPC window including the peak region.  Gives
      phasor points closer to the universal semicircle.  Recommended when
      an IRF measurement is available (--irf) for phasor-space correction,
      though it is still useful without one for relative comparisons.

IRF correction (--irf, optional)
---------------------------------
In phasor space, IRF correction is a single complex division:

    z_corrected = (G + i·S) / (z_irf · exp(i·w·shift))

This removes the phase shift and modulation attenuation introduced by the
finite IRF width.  The IRF file and the data generally have a timing offset
(separate acquisitions, cabling); it is measured with one reconvolution fit
of the pooled masked decay -- the same fit the reconvolution plugins use --
and applied as the exp(...) rotation. With --no-shift the offset is held at
zero.  Only valid with --window-mode full (skipped for 'tail'); it restores
points to the universal semicircle for a monoexponential decay.

For tracking RELATIVE lifetime changes in a timelapse on the same
instrument, IRF correction is optional -- the bias is constant across
frames and cancels in differences.

Reference lifetimes (--tau-ref1, --tau-ref2)
--------------------------------------------
When both are given (e.g. 1.41 ns and 4.30 ns for Flipper-TR), the plugin:
  - Marks both reference points on the phasor scatter plot.
  - Draws the tie-line connecting them.
  - Computes the intensity fraction of the long component for every pixel
    from its projected position on the tie-line:
      f_long_window -- share of the photons INSIDE the analysed window
      f_long        -- share of ALL photons (whole decay). For a tail window
                       the short component loses more photons before the
                       tail starts, so f_long_window overestimates; f_long
                       corrects for this using the fraction of each
                       reference component's photons that the window keeps.

The reference phasor coordinates are computed numerically from a pure
monoexponential decay (starting at the decay peak, repeating every laser
period) using the same time window as the data, so they are consistent with
the measured phasor points regardless of which bins are included.

Laser period (--rep-rate-mhz)
-----------------------------
The phasor frequency is the laser repetition rate. If it is not given, the
period is ASSUMED equal to the histogram window (nb * bin width); give the
real repetition rate whenever the two differ.

Background subtraction (subtract_background, default on)
--------------------------------------------------------
Uncorrelated counts (dark counts, ambient light) spread evenly over time.
They pull every phasor toward the background's own phasor (the origin for a
full-period window), lowering the modulation and biasing f_long. A flat
level B per pixel per bin is subtracted before the transform:

    G + iS = sum((n_k - B) e^{i w t_k}) / sum(n_k - B)

B cannot simply be read from the last bins: when the decay has not finished
within the laser period (e.g. a 4.3 ns component at 80 MHz) most of those
counts are real decay, and how much varies from pixel to pixel. Instead,
each pixel's tail bins are described as a1*p1 + a2*p2 + B with p1, p2 the
periodic decay shapes of two fixed lifetimes (tau_ref1/tau_ref2 if given,
else a pooled biexponential fit). With the lifetimes fixed this is linear,
so all pixels are solved in one least-squares step -- no optimizer, and B
is unbiased. The decay amplitudes are discarded; the phasor itself stays
model-free.

Outputs (per input file)
-------------------------
  <prefix>_phasor_scatter.png   -- 2D density phasor plot
  <prefix>_phasor_maps.png      -- spatial maps of tau_phase, tau_mod, f_long
  <prefix>_phasor.npz           -- all maps (G, S, tau_phase, tau_mod, ...)
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flim_core.config import MIN_NEIGHBORS, BIN_SIZE, DEFAULT_THRESHOLD, TAIL_START_OFFSET
from flim_core.io import load_flim_data, photon_count_map, build_decay_cube
from flim_core.masking import (
    clean_mask, bin_cube, bin_time_cube, moving_average_cube, crop_to_roi,
    report_dead_pixels, load_mask,
)
from flim_core.fitting import _tail_axis, global_fit
from flim_core.config import IRF_SHIFT_SEARCH
from flim_core.irf import global_reconv_fit, _BIN_CONVENTION_SHIFT
from flim_core.plotting import plot_overview_and_mask
from flim_core.pixel_offset import PixelOffsetCorrection
from flim_core.bin_pattern import BinPatternCorrection
from analyses._common import build_group_cube

NAME = "Phasor (pixel-wise)"
DESCRIPTION = (
    "Model-free phasor (G, S) analysis via DFT at the laser rep. frequency. "
    "No exponential model, no convergence failures. Works at low photon counts."
)
REQUIRES_RAW = False   # works on raw, exported, and .ds3d alike
IRF_CAPABLE = True     # optional IRF file for phasor-space correction
# Rendered by the GUI as plain text entries (values arrive as str, possibly
# empty) -- parsed by _extra_args() below. Empty = default.
EXTRA_ARGS = ["harmonic", "window_mode", "tau_ref1", "tau_ref2", "rep_rate_mhz",
              "subtract_background"]


def _extra_args(args):
    """Parse the plugin-specific settings. The GUI passes them as strings
    ("" when left blank); the CLI passes already-typed values."""
    def _num(name, cast, default):
        v = getattr(args, name, None)
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        try:
            return cast(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name} = {v!r} is not a valid number")

    harmonic = _num("harmonic", int, 1)
    if harmonic < 1:
        raise ValueError(f"harmonic must be >= 1, got {harmonic}")
    window_mode = (getattr(args, "window_mode", None) or "tail").strip().lower()
    if window_mode not in ("tail", "full"):
        raise ValueError(f"window_mode must be 'tail' or 'full', got {window_mode!r}")
    tau_ref1 = _num("tau_ref1", float, None)
    tau_ref2 = _num("tau_ref2", float, None)
    if (tau_ref1 is None) != (tau_ref2 is None):
        print("  NOTE: only one of tau_ref1/tau_ref2 given -- both are needed for "
              "the tie-line and f_long; skipping them.")
        tau_ref1 = tau_ref2 = None
    rep_rate_mhz = _num("rep_rate_mhz", float, None)
    if rep_rate_mhz is not None and rep_rate_mhz <= 0:
        raise ValueError(f"rep_rate_mhz must be > 0, got {rep_rate_mhz}")
    sb = getattr(args, "subtract_background", None)
    if sb is None or (isinstance(sb, str) and not sb.strip()):
        subtract_bg = True
    elif isinstance(sb, str):
        if sb.strip().lower() in ("1", "true", "yes", "y", "on"):
            subtract_bg = True
        elif sb.strip().lower() in ("0", "false", "no", "n", "off"):
            subtract_bg = False
        else:
            raise ValueError(f"subtract_background = {sb!r}: use yes/no")
    else:
        subtract_bg = bool(sb)
    return harmonic, window_mode, tau_ref1, tau_ref2, rep_rate_mhz, subtract_bg


def estimate_background(cube, mask, s, tail_offset, T_laser_ns, taus=None):
    """Per-pixel flat background level (counts per bin), [H, W].

    In the tail bins (past the IRF-dominated rise) every masked pixel is
    described as  n_k = a1 * p1(t_k) + a2 * p2(t_k) + B,  where p1, p2 are
    the periodic decay shapes of the two lifetimes (starting at the decay
    peak, repeating every laser period) and a1, a2, B are free per pixel.
    With the lifetimes fixed this is LINEAR, so all pixels are solved at once
    by one least-squares solve -- no optimizer.

    taus: (tau_short, tau_long) in ns -- the reference lifetimes if given,
    otherwise taken from one biexponential + background fit of the pooled
    masked decay (global_fit).

    The two decay terms are only there to separate B from the decay; their
    amplitudes are not used. B is NOT clipped at 0: clipping keeps only the
    positive half of the noise and biases every result.

    Returns (B or None, info dict).
    """
    nb = cube.shape[2]
    tpb_ns = s["tpb"] * 1e9
    pooled = cube[mask].sum(axis=0).astype(float)
    _, peak_bin, s_idx, tn = _tail_axis(s, pooled[None, None, :], offset=tail_offset)
    info = dict(taus=None, n_tail_bins=nb - s_idx)
    if nb - s_idx < 10:
        print("  WARNING: fewer than 10 tail bins -- background subtraction skipped.")
        return None, info

    if taus is None:
        try:
            g = global_fit(pooled, s, s_idx, tn, mono=False, fit_background=True)
            taus = (float(g["tau_short"]), float(g["tau_long"]))
            src = "pooled biexponential fit"
        except Exception as exc:
            print(f"  WARNING: pooled fit for the background estimate failed ({exc}); "
                  f"background subtraction skipped.")
            return None, info
    else:
        src = "reference lifetimes"
    info["taus"] = taus

    t0_ns = (peak_bin - s["B0"]) * tpb_ns
    t_tail = np.arange(s_idx, nb) * tpb_ns
    u = np.mod(t_tail - t0_ns, T_laser_ns)
    cols = [np.exp(-u / tau) / (1.0 - np.exp(-T_laser_ns / tau)) for tau in taus]
    X = np.column_stack(cols + [np.ones_like(u)])
    X_scaled = X / np.linalg.norm(X, axis=0)            # column scaling for the condition check
    cond = np.linalg.cond(X_scaled)
    if cond > 1e4:
        print(f"  WARNING: background cannot be told apart from the decay in this window "
              f"(condition number {cond:.0f}); background subtraction skipped.")
        return None, info

    Y = cube[mask][:, s_idx:].astype(np.float64).T       # (n_tail, n_pixels)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    B = np.zeros(cube.shape[:2])
    B[mask] = coef[2]

    n_tot = cube[mask].sum(axis=1)
    frac = coef[2] * nb / np.where(n_tot > 0, n_tot, np.nan)
    print(f"  Background: per-pixel fit in the tail (decay shapes {taus[0]:.2f} / "
          f"{taus[1]:.2f} ns from {src}): median {np.median(coef[2]):.2f} counts/bin "
          f"= {100 * np.nanmedian(frac):.1f}% of all photons")
    return B, info


# ---------------------------------------------------------------------------
# Plugin contract helpers
# ---------------------------------------------------------------------------

def preview_slice(loaded_data, settings):
    """Return (img, cube, description) for the GUI mask-preview panel."""
    if "cube" in loaded_data:
        return loaded_data["img"], loaded_data["cube"], "whole acquisition"
    img, cube = build_group_cube(loaded_data, slice(None))
    return img, cube, "whole acquisition"


# ---------------------------------------------------------------------------
# Core phasor maths
# ---------------------------------------------------------------------------

def _build_time_axes(s, cube, rep_rate_mhz=None):
    """Return (nb, tpb_ns, T_laser_ns, t_full) for the loaded data.

    nb         -- number of time bins stored in the cube
    tpb_ns     -- nanoseconds per bin
    T_laser_ns -- laser period: 1000 / rep_rate_mhz when the repetition rate
                  is given, otherwise ASSUMED equal to the histogram window
                  (nb * tpb_ns). The phasor frequency is 1/T_laser_ns, so a
                  wrong period puts every G, S at the wrong frequency.
    t_full     -- absolute time axis for every bin, shape (nb,), ns from B0
    """
    nb = cube.shape[2]          # authoritative after any time binning
    tpb_ns = s["tpb"] * 1e9
    window_ns = nb * tpb_ns
    if rep_rate_mhz is None:
        T_laser_ns = window_ns
        print(f"  NOTE: laser period assumed = histogram window ({window_ns:.2f} ns). "
              f"Set rep_rate_mhz if the repetition period differs.")
    else:
        T_laser_ns = 1000.0 / rep_rate_mhz
        print(f"  Laser period {T_laser_ns:.2f} ns ({rep_rate_mhz:g} MHz); "
              f"histogram window {window_ns:.2f} ns")
        if window_ns > T_laser_ns * 1.01:
            print(f"  WARNING: the histogram window is longer than the laser period -- "
                  f"photons from consecutive pulses overlap in the window; check rep_rate_mhz.")
    t_full = np.arange(nb, dtype=np.float64) * tpb_ns
    return nb, tpb_ns, T_laser_ns, t_full


def _reference_phasor(tau_ns, t_abs_win, t0_ns, T_laser_ns, harmonic=1):
    """Phasor of a pure monoexponential decay seen through the same time
    window as the data.  Returns (G_ref, S_ref, w_window).

    The decay starts at t0_ns (the excitation time on the data's time axis)
    and repeats every laser period, so the photons from the previous pulse
    that are still arriving are included -- exactly what the camera records.

    Needed for the tie-line and f_long projection: a tail-only (or
    shorter-than-period) window moves a pure lifetime's phasor off the
    universal semicircle, so the reference must be computed numerically with
    the same window to be consistent with the measured pixel phasors.

    w_window -- fraction of this component's photons (per laser period) that
    fall inside the window. Converts window photon fractions back to
    whole-decay intensity fractions (see _window_to_total_fraction).
    """
    omega = 2.0 * np.pi * harmonic / T_laser_ns
    u = np.mod(t_abs_win - t0_ns, T_laser_ns)               # time since the latest pulse
    p = np.exp(-u / tau_ns) / (1.0 - np.exp(-T_laser_ns / tau_ns))
    tpb_ns = (t_abs_win[1] - t_abs_win[0]) if len(t_abs_win) > 1 else 0.0
    w_window = float(np.sum(p) * tpb_ns / tau_ns)           # integral over a period = 1
    h_ref = p / p.sum()                                      # normalise; only shape matters
    G_ref = float(np.sum(h_ref * np.cos(omega * t_abs_win)))
    S_ref = float(np.sum(h_ref * np.sin(omega * t_abs_win)))
    return G_ref, S_ref, w_window


def _window_to_total_fraction(f_win, w_short, w_long):
    """Convert the long component's share of the photons INSIDE the window
    (what the tie-line projection measures) into its share of ALL photons.
    A tail window keeps a smaller part of the short component than of the
    long one, so f_win overestimates the long fraction; dividing each
    component's window photons by the part of it the window keeps undoes
    that."""
    a_long = f_win / w_long
    a_short = (1.0 - f_win) / w_short
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a_long / (a_long + a_short)).astype(np.float32)


def compute_phasor(cube, t_abs_window, T_laser_ns, mask, harmonic=1, bg_per_bin=None):
    """Compute per-pixel phasor coordinates (G, S) from TCSPC histograms.

    Parameters
    ----------
    cube : ndarray, shape (H, W, n_window)
        Per-pixel TCSPC histograms restricted to the chosen time window
        (either tail bins or all bins).  The caller slices cube[:, :, ...].
    t_abs_window : 1-D float array, shape (n_window,)
        Absolute time (ns) of each bin in the window, in the same units as
        T_laser_ns.  For tail-only: t_abs_window = t_full[s_tail_start:].
    T_laser_ns : float
        Full laser period in ns (determines the fundamental frequency).
    mask : ndarray bool, shape (H, W)
    harmonic : int
    bg_per_bin : ndarray (H, W) or None
        Flat background level subtracted from every bin before the
        transform (see estimate_background). n_photons stays the raw count.

    Returns
    -------
    dict with keys: G, S, modulation, tau_phase, tau_mod, n_photons
        All arrays shape (H, W), NaN outside mask or where n_photons == 0.
    """
    H, W = cube.shape[:2]
    omega = 2.0 * np.pi * harmonic / T_laser_ns

    h = cube.astype(np.float64)             # (H, W, n_window)
    n_photons = h.sum(axis=2)               # raw counts, before background subtraction
    n_photons[~mask] = np.nan
    if bg_per_bin is not None:
        h = h - bg_per_bin[:, :, None]
    total = h.sum(axis=2)                   # (H, W) signal photons

    with np.errstate(invalid='ignore', divide='ignore'):
        inv_total = np.where(total > 0, 1.0 / total, np.nan)

    cos_w = np.cos(omega * t_abs_window)    # (n_window,)
    sin_w = np.sin(omega * t_abs_window)    # (n_window,)

    G = (h * cos_w).sum(axis=2) * inv_total
    S = (h * sin_w).sum(axis=2) * inv_total

    G[~mask] = np.nan
    S[~mask] = np.nan

    modulation = np.sqrt(G**2 + S**2)
    phi = np.arctan2(S, G)                  # phasor phase angle (rad)
    tau_phase = np.tan(phi) / omega         # ns; negative for phi < 0 (un-physical)

    with np.errstate(invalid='ignore', divide='ignore'):
        m2 = modulation**2
        tau_mod = np.where(m2 < 1.0,
                           np.sqrt(1.0 / m2 - 1.0) / omega,
                           np.nan)          # ns; NaN when M >= 1 (also un-physical)

    return dict(
        G=G, S=S,
        modulation=modulation,
        tau_phase=tau_phase,
        tau_mod=tau_mod,
        n_photons=n_photons,
        T_laser_ns=T_laser_ns,
        harmonic=harmonic,
    )


def _irf_on_data_grid(irf_norm, nb):
    """The IRF kernel padded with zeros / truncated to the data's bin count,
    so it shares the data's bin grid (load_irf's kernel is compact -- zero
    outside the main peak -- so padding adds nothing)."""
    k = np.zeros(nb)
    n = min(nb, len(irf_norm))
    k[:n] = irf_norm[:n]
    return k


def irf_timing_shift(signal_sum, irf_grid, tpb_ns, no_shift=False):
    """Timing offset between the IRF file and the data, in bins, as the
    reconvolution model defines it (data = shift(IRF (x) decay, shift)).

    Measured with ONE biexponential reconvolution fit of the pooled masked
    decay (flim_core.irf.global_reconv_fit, free sub-bin shift) -- the same
    fit the reconvolution plugins use, so the phasor and the fits share one
    timing definition. The fit's shift includes the model's +0.5-bin
    sampling convention (see flim_core.irf._BIN_CONVENTION_SHIFT); it is
    removed here, so the value returned is the PHYSICAL offset. Dividing by
    the IRF rotated by the physical offset leaves the decay phasor as if
    sampled at bin CENTRES -- a close match to the continuous decay
    (tau_phase/tau_mod), and how the reference phasors are computed in this
    mode. With no_shift the physical offset is held at 0.

    Returns the shift, or None if the fit fails (correction then skipped
    rather than applied with a wrong offset).
    """
    if no_shift:
        return 0.0
    try:
        g = global_reconv_fit(signal_sum, irf_grid, tpb_ns)
        shift = float(g["shift"])
        if np.isfinite(shift) and abs(shift) < IRF_SHIFT_SEARCH:
            return shift - _BIN_CONVENTION_SHIFT
        print(f"  WARNING: fitted IRF timing offset {shift:.1f} bins is at the search limit.")
    except Exception as exc:
        print(f"  WARNING: reconvolution fit for the IRF timing offset failed ({exc}).")
    return None


def apply_irf_correction(result, irf_grid, tpb_ns, shift_bins, harmonic=1):
    """Correct phasor maps for the IRF by complex division:

        z_decay = z_data / (z_irf * exp(i w shift tpb))

    For periodic data recorded over one full laser period, the data
    histogram is the circular convolution of the decay with the IRF as it
    sits on the DATA's time axis. The IRF file's kernel is offset from that
    by shift_bins, the physical offset (see irf_timing_shift); a time shift is a pure rotation in
    phasor space, so it is applied as the exp(...) factor -- exact, no
    resampling. Without it, an offset of d ns rotates every corrected phasor
    by w*d and biases tau_phase by roughly d (plus a modulation error).

    irf_grid: IRF kernel on the data's bin grid (_irf_on_data_grid).
    Modifies result dict in-place (G, S, modulation, tau_phase, tau_mod).
    """
    T = result["T_laser_ns"]
    omega = 2.0 * np.pi * harmonic / T
    irf_h = irf_grid.astype(np.float64)
    t_irf = np.arange(len(irf_h)) * tpb_ns

    total_irf = irf_h.sum()
    if total_irf == 0:
        raise ValueError("IRF histogram is all zeros -- cannot compute IRF phasor.")
    z_irf = complex(np.sum(irf_h * np.cos(omega * t_irf)) / total_irf,
                    np.sum(irf_h * np.sin(omega * t_irf)) / total_irf)
    z_irf *= np.exp(1j * omega * shift_bins * tpb_ns)      # move the IRF onto the data's timing
    if abs(z_irf) < 1e-10:
        raise ValueError("IRF phasor magnitude ~0 -- check IRF file or harmonic.")
    print(f"    IRF phasor (aligned): G={z_irf.real:.4f}, S={z_irf.imag:.4f}, "
          f"|z|={abs(z_irf):.4f}")

    z = result["G"] + 1j * result["S"]
    z_corr = z / z_irf

    result["G"] = z_corr.real
    result["S"] = z_corr.imag
    result["modulation"] = np.abs(z_corr)
    phi = np.angle(z_corr)
    result["tau_phase"] = np.tan(phi) / omega
    with np.errstate(invalid='ignore', divide='ignore'):
        m2 = result["modulation"]**2
        result["tau_mod"] = np.where(m2 < 1.0,
                                     np.sqrt(1.0 / m2 - 1.0) / omega,
                                     np.nan)
    result["irf_corrected"] = True


def fraction_long(G, S, G1, S1, G2, S2):
    """Intensity fraction of the long-lifetime component from phasor position.

    For a two-component decay the phasor lies on the straight line connecting
    (G1, S1) [short] and (G2, S2) [long].  f_long is the fractional distance
    from (G1,S1) toward (G2,S2), clamped to [0, 1].

    (G1, S1), (G2, S2) should be computed numerically for the same time window
    as the data via _reference_phasor_tail -- NOT the analytical semicircle
    formula, because the tail window shifts points off the semicircle.

    Returns ndarray (H, W), NaN where G or S is NaN.
    """
    dG = G2 - G1
    dS = S2 - S1
    denom = dG**2 + dS**2
    if denom < 1e-15:
        raise ValueError("tau_ref1 and tau_ref2 give identical phasor points "
                         "for this laser period -- check the values and harmonic.")
    f2 = ((G - G1) * dG + (S - S1) * dS) / denom
    f2 = np.where(np.isfinite(G) & np.isfinite(S),
                  np.clip(f2, 0.0, 1.0), np.nan)
    return f2.astype(np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_phasor_scatter(result, mask, prefix, tau_ref1=None, tau_ref2=None,
                        G1_ref=None, S1_ref=None, G2_ref=None, S2_ref=None,
                        show=True):
    """2-D density phasor scatter plot with universal semicircle overlay."""
    G_px = result["G"][mask & np.isfinite(result["G"])]
    S_px = result["S"][mask & np.isfinite(result["S"])]
    T = result["T_laser_ns"]
    h = result["harmonic"]
    omega = 2.0 * np.pi * h / T

    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Universal semicircle (G + iS = 1/(1 + iωτ))
    theta = np.linspace(0, np.pi, 500)
    ax.plot(0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta),
            color="#cccccc", lw=1.5, zorder=1, label="universal semicircle")

    # Lifetime ticks on semicircle (reference orientation only)
    tick_taus = [0.5, 1.0, 1.41, 2.0, 3.0, 4.30, 6.0, 10.0]
    for tau in tick_taus:
        Gt = 1.0 / (1 + (omega * tau)**2)
        St = omega * tau / (1 + (omega * tau)**2)
        ax.plot(Gt, St, "o", ms=2.5, color="#bbbbbb", zorder=2)
        ax.annotate(f"{tau}", (Gt, St), textcoords="offset points",
                    xytext=(2, 2), fontsize=5.5, color="#999999")

    # Plot limits: always show the semicircle, but extend to include the data
    # and reference points -- tail-window phasors lie OFF the semicircle
    # (often at negative G), so fixed semicircle limits would clip them.
    g_pts = [-0.05, 1.05]
    s_pts = [-0.02, 0.65]
    if len(G_px):
        g_pts += [np.percentile(G_px, 0.5) - 0.05, np.percentile(G_px, 99.5) + 0.05]
        s_pts += [np.percentile(S_px, 0.5) - 0.05, np.percentile(S_px, 99.5) + 0.05]
    if G1_ref is not None:
        g_pts += [G1_ref - 0.05, G2_ref - 0.05, G1_ref + 0.05, G2_ref + 0.05]
        s_pts += [S1_ref - 0.05, S2_ref - 0.05, S1_ref + 0.05, S2_ref + 0.05]
    xlim = (min(g_pts), max(g_pts))
    ylim = (min(s_pts), max(s_pts))

    # Pixel density: log-scaled 2D histogram over the data's own extent
    g_lo, g_hi = (np.percentile(G_px, [0.5, 99.5]) if len(G_px) else (0, 0))
    s_lo, s_hi = (np.percentile(S_px, [0.5, 99.5]) if len(S_px) else (0, 0))
    if len(G_px) > 1 and g_hi > g_lo and s_hi > s_lo:
        h2d, xedg, yedg = np.histogram2d(
            G_px, S_px, bins=100,
            range=[[g_lo, g_hi], [s_lo, s_hi]])
        h2d_plot = np.where(h2d > 0, np.log1p(h2d), np.nan)
        ax.pcolormesh(xedg, yedg, h2d_plot.T,
                      cmap="viridis", alpha=0.9, zorder=3)
    else:
        ax.scatter(G_px, S_px, s=4, c="#2196F3", alpha=0.6, zorder=3)

    # Reference points and tie-line (computed numerically for this window)
    if tau_ref1 is not None and tau_ref2 is not None and G1_ref is not None:
        ax.plot([G1_ref, G2_ref], [S1_ref, S2_ref], "--",
                color="#E05C3A", lw=1.5, alpha=0.9, zorder=4,
                label=f"tie-line τ={tau_ref1}–{tau_ref2} ns")
        ax.plot(G1_ref, S1_ref, "^", color="#2196F3", ms=9, zorder=5,
                label=f"τ_short = {tau_ref1} ns")
        ax.plot(G2_ref, S2_ref, "s", color="#E05C3A", ms=9, zorder=5,
                label=f"τ_long  = {tau_ref2} ns")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("G  (cosine component)", fontsize=11)
    ax.set_ylabel("S  (sine component)", fontsize=11)
    ax.set_aspect("equal")
    irf_tag = " + IRF corr." if result.get("irf_corrected") else ""
    ax.set_title(f"Phasor plot  (harmonic {h}, T = {T:.2f} ns{irf_tag})", fontsize=12)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    fig.tight_layout()

    png_path = f"{prefix}_phasor_scatter.png"
    fig.savefig(png_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved phasor scatter -> {png_path}")
    return png_path


def plot_phasor_maps(result, mask, prefix, f_long=None, tau_ref1=None,
                     tau_ref2=None, show=True):
    """Spatial maps of tau_phase, tau_mod, and optionally f_long."""
    n_panels = 3 if f_long is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5),
                             squeeze=False)
    axes = axes[0]

    def _show(ax, data, title, cmap="plasma", vmin=None, vmax=None, unit="ns"):
        d = np.where(mask, data, np.nan).copy()
        finite = d[np.isfinite(d)]
        if finite.size == 0:
            ax.set_title(title + "\n(no valid pixels)")
            return
        v0 = np.percentile(finite, 2) if vmin is None else vmin
        v1 = np.percentile(finite, 98) if vmax is None else vmax
        im = ax.imshow(d, cmap=cmap, vmin=v0, vmax=v1, origin="upper")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cb.set_label(unit, fontsize=9)

    _show(axes[0], result["tau_phase"], "τ_phase  (ns)", cmap="plasma")
    _show(axes[1], result["tau_mod"],   "τ_mod  (ns)",   cmap="plasma")
    if f_long is not None:
        lbl = (f"f_long  (τ={tau_ref2} ns)"
               if tau_ref2 is not None else "f_long")
        _show(axes[2], f_long, lbl, cmap="RdYlBu_r", vmin=0.0, vmax=1.0,
              unit="intensity fraction")

    fig.suptitle("Phasor-derived maps", fontsize=12)
    fig.tight_layout()
    png_path = f"{prefix}_phasor_maps.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved phasor maps -> {png_path}")
    return png_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_file(h5file, args):
    """Load → mask → phasor → plot → save pipeline for one file."""
    harmonic, window_mode, tau_ref1, tau_ref2, rep_rate_mhz, subtract_bg = _extra_args(args)
    show = not getattr(args, "no_show", True)

    stem   = Path(h5file).stem
    prefix_arg = getattr(args, "prefix", None)
    prefix = f"{prefix_arg}_{stem}" if prefix_arg else stem

    # ---- optional IRF ----
    irf = None
    if getattr(args, "irf", None):
        from flim_core.irf import load_irf
        print(f"Loading IRF from {args.irf} ...")
        irf = load_irf(
            args.irf,
            window=not getattr(args, "irf_full", False),
            window_end_ns=getattr(args, "irf_window_end_ns", 15.0),
            time_bin=getattr(args, "time_bin", None) or 1,
            pixel_offset_path=getattr(args, "pixel_offset", None),
        )
        print(f"  IRF: {irf['source']} | FWHM {irf['fwhm_bins']} bins "
              f"({irf['fwhm_bins']*irf['tpb']*1e9:.2f} ns) | "
              f"{irf['bg_fraction']*100:.0f}% background removed")

    # ---- load data ----
    print(f"\n=== {h5file} ===")
    s = load_flim_data(h5file)
    print(f"  {s['source']} | {s['n_photons']} photons | "
          f"{s['W']}x{s['H']} | {s['tpb']*1e12:.1f} ps/bin")

    img  = photon_count_map(s)
    img_orig = img   # for the overview panel, before ROI-crop/binning
    report_dead_pixels(img)
    cube = build_decay_cube(s)

    # ---- 8-bin TDC pattern correction (optional; same order as normal.py) ----
    if getattr(args, "bin_pattern", False):
        print("  Applying 8-bin pattern correction ...")
        try:
            bp = BinPatternCorrection.from_cube(cube, s["B0"])
            cube = bp.apply(cube, s["B0"])
        except Exception as exc:
            print(f"  WARNING: 8-bin pattern correction failed ({exc}); continuing without it.")

    # ---- pixel-offset correction (optional) ----
    if getattr(args, "pixel_offset", None):
        try:
            corr = PixelOffsetCorrection.load(args.pixel_offset)
            cube = corr.apply(cube, s["tpb"])
        except Exception as exc:
            print(f"  WARNING: pixel offset correction failed ({exc}); continuing without it.")

    # ---- ROI ----
    roi = getattr(args, "roi", None)
    if roi is not None:
        img, cube, roi = crop_to_roi(img, cube, roi)
        img_orig = img

    # ---- spatial smoothing (moving average and block binning are exclusive) ----
    moving_avg = getattr(args, "moving_average", None)
    bin_size = getattr(args, "bin", None) or 1
    if moving_avg and moving_avg > 1:
        img, cube = moving_average_cube(img, cube, moving_avg)
        plot_bin_size = 1
    else:
        if bin_size > 1:
            img, cube = bin_cube(img, cube, bin_size)
        plot_bin_size = bin_size

    # ---- time binning (keeps s["tpb"]/B1 consistent with the cube) ----
    time_bin = getattr(args, "time_bin", None) or 1
    raw_offset = getattr(args, "tail_start_offset", None)
    if raw_offset is None:
        raw_offset = TAIL_START_OFFSET
    if time_bin > 1:
        cube, tpb_binned = bin_time_cube(cube, time_bin, s["tpb"])
        s = dict(s)
        s["tpb"] = tpb_binned
        s["B1"] = s["B0"] + cube.shape[2] - 1
        raw_offset = max(1, round(raw_offset / time_bin))   # keep the ns margin constant
    tail_offset = raw_offset

    if irf is not None and abs(s["tpb"] - irf["tpb"]) > 1e-15:
        print(f"  WARNING: data bin width ({s['tpb']:.3e} s) differs from the IRF's "
              f"({irf['tpb']:.3e} s) -- the IRF phasor will not match this data.")

    # ---- mask ----
    ext_mask_path = getattr(args, "external_mask", None) or getattr(args, "mask", None)
    if ext_mask_path:
        mask = load_mask(ext_mask_path)
        if mask.shape != img.shape:
            raise ValueError(f"external mask shape {mask.shape} does not match the image "
                             f"shape {img.shape} -- check that bin/ROI match how the mask was saved.")
        threshold = None
        min_nb = None
        removed = np.zeros_like(mask)
        print(f"  external mask: {int(mask.sum())} pixels")
    else:
        threshold = getattr(args, "threshold", None)
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        min_nb = getattr(args, "min_neighbors", None)
        if min_nb is None:
            min_nb = MIN_NEIGHBORS
        raw_mask = img > threshold
        mask     = clean_mask(raw_mask, min_nb)
        removed  = raw_mask & ~mask
        print(f"  mask (threshold={threshold}): {int(mask.sum())} pixels kept")

    if mask.sum() == 0:
        print("  No pixels remain in the mask; lower the threshold or the neighbour requirement.")
        return

    if getattr(args, "save_overview", True):
        plot_overview_and_mask(img_orig, img, mask, removed, threshold, min_nb,
                               plot_bin_size, f"{prefix}_overview.png", show)

    # ---- time axes ----
    nb, tpb_ns, T_laser_ns, t_full = _build_time_axes(s, cube, rep_rate_mhz)

    # Find tail start (same logic as all other plugins)
    summed, peak_bin, s_tail_start, _tn_tail = _tail_axis(s, cube, offset=tail_offset)

    # Choose window
    if window_mode == "full":
        win_slice = slice(None)
        win_label = f"full ({nb} bins)"
    else:
        win_slice = slice(s_tail_start, None)
        win_label = f"tail ({nb - s_tail_start} bins, start bin {s_tail_start})"
    cube_win  = cube[:, :, win_slice]
    t_abs_win = t_full[win_slice]                   # ns

    # ---- background ----
    bg_map, bg_info = None, None
    if subtract_bg:
        bg_map, bg_info = estimate_background(
            cube, mask, s, tail_offset, T_laser_ns,
            taus=(tau_ref1, tau_ref2) if tau_ref1 is not None else None)
    else:
        print("  Background subtraction off.")

    # ---- compute phasor ----
    print(f"  Phasor: harmonic={harmonic}, window='{window_mode}' [{win_label}], "
          f"T={T_laser_ns:.2f} ns ...")
    result = compute_phasor(cube_win, t_abs_win, T_laser_ns, mask, harmonic=harmonic,
                            bg_per_bin=bg_map)
    result["irf_corrected"] = False

    # ---- optional IRF correction ----
    if irf is not None and window_mode == "tail":
        # Dividing by the IRF phasor is only valid when the data phasor is the
        # transform of the FULL convolved decay (IRF * decay). A tail-only
        # phasor is not, so the division would give a wrong (G, S).
        print("  NOTE: IRF correction skipped -- it is only valid with window_mode='full'.")
    elif irf is not None:
        no_shift = getattr(args, "no_shift", False)
        irf_grid = _irf_on_data_grid(irf["irf_norm"], nb)
        signal_sum = cube[mask].sum(axis=0).astype(float)
        print("  Measuring the IRF/data timing offset (one pooled reconvolution fit) ..."
              if not no_shift else "  IRF/data timing offset held at 0 (no_shift).")
        irf_shift = irf_timing_shift(signal_sum, irf_grid, tpb_ns, no_shift=no_shift)
        if irf_shift is None:
            print("  WARNING: IRF correction skipped -- the timing offset could not be "
                  "measured, and correcting with a wrong offset would bias every lifetime.")
        else:
            print(f"    IRF/data timing offset: {irf_shift:+.2f} bins "
                  f"({irf_shift * tpb_ns * 1000:+.0f} ps; positive = data lag the IRF)")
            if nb * tpb_ns < 0.99 * T_laser_ns:
                print("  WARNING: the histogram window is shorter than the laser period, "
                      "so the data are not one full period -- IRF correction is approximate.")
            print("  Applying IRF correction in phasor space ...")
            apply_irf_correction(result, irf_grid, tpb_ns, irf_shift, harmonic=harmonic)
            result["irf_shift_bins"] = irf_shift
    elif window_mode == "full":
        print("  NOTE: 'full' window without --irf -- G/S include IRF phase "
              "contribution.  Results are valid for relative comparisons.")

    # ---- reference phasor points and f_long ----
    G1_ref = S1_ref = G2_ref = S2_ref = None
    f_long_map = f_long_win_map = None
    w1 = w2 = t0_ns = None

    if tau_ref1 is not None and tau_ref2 is not None:
        # Excitation time on the data's time axis. After IRF correction the
        # decay is referenced to the IRF's own time origin, as sampled at bin
        # centres (see irf_timing_shift) -- hence -half a bin on this
        # left-edge time axis. Without it, the decay peak is used -- a slight
        # approximation, because the true excitation is a little earlier than
        # the peak (by roughly the IRF rise), which slightly biases the
        # tail-window weights.
        t0_ns = -0.5 * tpb_ns if result["irf_corrected"] else (peak_bin - s["B0"]) * tpb_ns
        G1_ref, S1_ref, w1 = _reference_phasor(tau_ref1, t_abs_win, t0_ns, T_laser_ns, harmonic)
        G2_ref, S2_ref, w2 = _reference_phasor(tau_ref2, t_abs_win, t0_ns, T_laser_ns, harmonic)
        print(f"  Reference phasors (this window, decay start t0={t0_ns:.2f} ns):")
        print(f"    tau_short={tau_ref1} ns -> G={G1_ref:.4f}, S={S1_ref:.4f}, "
              f"{w1*100:.1f}% of its photons in the window")
        print(f"    tau_long ={tau_ref2} ns -> G={G2_ref:.4f}, S={S2_ref:.4f}, "
              f"{w2*100:.1f}% of its photons in the window")
        print(f"  Computing f_long ...")
        f_long_win_map = fraction_long(
            result["G"], result["S"], G1_ref, S1_ref, G2_ref, S2_ref)
        # f_long = share of ALL photons (whole decay); f_long_window = share of
        # the photons inside the analysed window (differs for a tail window).
        f_long_map = _window_to_total_fraction(f_long_win_map, w1, w2)
        for lbl, arr in (("f_long (whole decay)", f_long_map),
                         ("f_long_window", f_long_win_map)):
            v = arr[np.isfinite(arr)]
            if v.size:
                print(f"    {lbl}: median={np.median(v):.3f}, "
                      f"mean={np.mean(v):.3f}, std={np.std(v):.3f}")

    # ---- print summary stats ----
    for lbl, arr in [("G",       result["G"]),
                     ("S",       result["S"]),
                     ("tau_phase", result["tau_phase"]),
                     ("tau_mod",   result["tau_mod"])]:
        v = arr[mask & np.isfinite(arr)]
        if v.size:
            print(f"  {lbl:<10} median={np.median(v):.3f}  "
                  f"std={np.std(v):.3f}  "
                  f"[{np.percentile(v,2):.3f} to {np.percentile(v,98):.3f}]")

    # ---- plots ----
    if getattr(args, "save_results", True):
        plot_phasor_scatter(result, mask, prefix,
                            tau_ref1=tau_ref1, tau_ref2=tau_ref2,
                            G1_ref=G1_ref, S1_ref=S1_ref,
                            G2_ref=G2_ref, S2_ref=S2_ref,
                            show=show)
        plot_phasor_maps(result, mask, prefix, f_long=f_long_map,
                         tau_ref1=tau_ref1, tau_ref2=tau_ref2, show=show)

    # ---- save NPZ (always; project convention is {prefix}_maps.npz) ----
    npz_path = f"{prefix}_maps.npz"
    save_kw = dict(
        G                = result["G"].astype(np.float32),
        S                = result["S"].astype(np.float32),
        modulation       = result["modulation"].astype(np.float32),
        tau_phase        = result["tau_phase"].astype(np.float32),
        tau_mod          = result["tau_mod"].astype(np.float32),
        n_photons        = result["n_photons"].astype(np.float32),
        mask             = mask,
        removed_isolated = removed,
        T_laser_ns       = np.float64(T_laser_ns),
        harmonic         = np.int32(harmonic),
        window_mode      = window_mode,
        irf_corrected    = np.bool_(result["irf_corrected"]),
        s_tail_start     = np.int32(s_tail_start),
        threshold        = threshold,
        min_neighbors    = min_nb,
        bin_size         = plot_bin_size,
        moving_average   = (moving_avg or 1),
        time_bin         = time_bin,
        photons_per_pixel = img,
        background_subtracted = np.bool_(bg_map is not None),
    )
    if "irf_shift_bins" in result:
        save_kw["irf_shift"] = np.float64(result["irf_shift_bins"])   # physical offset, bins
    if bg_map is not None:
        save_kw["background_per_bin"] = bg_map.astype(np.float32)
        save_kw["background_taus_ns"] = np.array(bg_info["taus"], dtype=np.float64)
    if f_long_map is not None:
        save_kw["f_long"]     = f_long_map
        save_kw["f_long_window"] = f_long_win_map
        save_kw["w_window_ref1"] = np.float64(w1)
        save_kw["w_window_ref2"] = np.float64(w2)
        save_kw["t0_ns"]      = np.float64(t0_ns)
        save_kw["tau_ref1_ns"] = np.float64(tau_ref1)
        save_kw["tau_ref2_ns"] = np.float64(tau_ref2)
        save_kw["G_ref1"]     = np.float64(G1_ref)
        save_kw["S_ref1"]     = np.float64(S1_ref)
        save_kw["G_ref2"]     = np.float64(G2_ref)
        save_kw["S_ref2"]     = np.float64(S2_ref)
    np.savez(npz_path, **save_kw)
    print(f"  saved {npz_path}")


def run(args):
    """Entry point called by the GUI and batch runner -- one file per call
    (args.h5file), like every other plugin. Exceptions propagate so the GUI
    can show them as errors."""
    process_file(args.h5file, args)


def main():
    p = argparse.ArgumentParser(
        description="Phasor FLIM analysis (pixel-wise, model-free)"
    )
    p.add_argument("h5files", nargs="+",
                   help="Flimera HDF5 input file(s)")
    p.add_argument("--prefix", default="",
                   help="Output filename prefix")

    # Mask / spatial
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Minimum photons/pixel for the mask")
    p.add_argument("--min-neighbors", type=int, default=MIN_NEIGHBORS,
                   help="Min occupied neighbours to keep a pixel (noise filter)")
    p.add_argument("--mask", default=None,
                   help="External mask .npy file (overrides threshold/min-neighbors)")
    p.add_argument("--bin", type=int, default=BIN_SIZE,
                   help="Spatial binning factor (NxN super-pixels)")
    p.add_argument("--moving-average", type=int, default=None,
                   help="NxN moving-average spatial smoothing (alternative to --bin)")
    p.add_argument("--roi", nargs=4, type=int, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="Crop to a region of interest (pixels; X1/Y1 exclusive)")
    p.add_argument("--bin-pattern", action="store_true",
                   help="Apply the self-calibrating 8-bin TDC pattern correction")
    p.add_argument("--pixel-offset", default=None,
                   help="PixelOffsetCorrection .npy file (SPAD TDC timing correction)")

    # Phasor options
    p.add_argument("--harmonic", type=int, default=1,
                   help="DFT harmonic (1=fundamental, 2=second, …)")
    p.add_argument("--window-mode", choices=["tail", "full"], default="tail",
                   help="Time window: 'tail' (IRF-free, default) or "
                        "'full' (whole TCSPC window, best with --irf)")
    p.add_argument("--tau-ref1", type=float, default=None,
                   help="Short-component lifetime (ns) for tie-line + f_long, "
                        "e.g. 1.41 for Flipper-TR")
    p.add_argument("--tau-ref2", type=float, default=None,
                   help="Long-component lifetime (ns) for tie-line + f_long, "
                        "e.g. 4.30 for Flipper-TR")
    p.add_argument("--rep-rate-mhz", type=float, default=None,
                   help="Laser repetition rate (MHz). Default: assume the period "
                        "equals the histogram window")
    p.add_argument("--no-subtract-background", dest="subtract_background",
                   action="store_false",
                   help="Skip the flat-background subtraction (default: subtract)")

    # IRF correction
    p.add_argument("--irf", default=None,
                   help="IRF measurement file for phasor-space correction")
    p.add_argument("--no-shift", action="store_true",
                   help="Assume no IRF/data timing offset instead of measuring it")
    p.add_argument("--irf-full", action="store_true",
                   help="Keep full IRF trace (echoes); default: window to main peak")
    p.add_argument("--irf-window-end-ns", type=float, default=15.0,
                   help="End of IRF window in ns past the peak (--irf-full only)")

    # Misc
    p.add_argument("--tail-start-offset", type=int, default=None,
                   help="Bins after peak to start tail (overrides config default)")
    p.add_argument("--time-bin", type=int, default=None,
                   help="Group every N adjacent microtime bins before analysis")
    p.add_argument("--no-show", action="store_true",
                   help="Suppress interactive plot windows (always save PNGs)")

    args = p.parse_args()
    for h5file in args.h5files:
        file_args = copy.copy(args)
        file_args.h5file = h5file
        try:
            run(file_args)
        except Exception as exc:
            import traceback
            print(f"ERROR processing {h5file}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
