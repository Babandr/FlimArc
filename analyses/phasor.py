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

Time window (chosen by fit_method)
----------------------------------
  fit_method "tail" (default): Fourier-transform only the tail bins — the same window
      as the existing tail-fit plugins, beginning a few bins after the
      decay peak.  Excludes the IRF-dominated rise region.  No IRF file
      needed.  The resulting phasor points do NOT lie exactly on the
      universal semicircle (because early photons are excluded), but they
      move monotonically and reproducibly with lifetime changes, making
      this mode ideal for relative measurements such as timelapse tracking.

  fit_method "irf": use the entire TCSPC window including the peak region
      and divide out the IRF phasor (--irf required).  Points of a
      monoexponential decay land on the universal semicircle, so
      tau_phase / tau_mod are real lifetimes.  An uncorrected full window
      is not offered: its points are rotated by the IRF and the timing
      offset, which mimics a lifetime change.

IRF correction (--irf, optional)
---------------------------------
In phasor space, IRF correction is a single complex division:

    z_corrected = (G + i·S) / (z_irf · exp(i·w·shift))

This removes the phase shift and modulation attenuation introduced by the
finite IRF width.  The IRF file and the data generally have a timing offset
(separate acquisitions, cabling); it is measured with one reconvolution fit
of the pooled masked decay -- the same fit the reconvolution plugins use --
and applied as the exp(...) rotation. With --no-shift the offset is held at
zero.  Only valid with the full window (fit_method "irf"); it restores
points to the universal semicircle for a monoexponential decay.

For tracking RELATIVE lifetime changes in a timelapse on the same
instrument, IRF correction is optional -- the bias is constant across
frames and cancels in differences.

Per-pixel IRF (per_pixel_irf, IRF mode only)
--------------------------------------------
Every SPAD pixel has its own timing offset (~100-180 ps rms). One pooled
IRF cannot remove it: each pixel's phasor is rotated by w * its offset,
which stretches the cloud along the arc (seen as a lifetime spread). With
per_pixel_irf each pixel is divided by ITS OWN IRF phasor, from the same
pixel of the IRF file put through the same pixel-offset / ROI / binning /
time-binning steps as the data -- the pixel's offset is in both and
cancels, with or without a calibration map. The pooled IRF/data timing
offset is still measured once and applied to every pixel, after removing
from it the masked pixels' mean IRF timing relative to the whole sensor
(reference_per_pixel_irf) -- the per-pixel IRFs already contain that.
Per-pixel IRF background = mean of that pixel's far tail (same region
load_irf uses); bins kept = the pooled kernel's window +- 0.5 ns. Pixels
with < 100 IRF photons fall back to the pooled IRF (logged). The IRF's own
photon noise adds a little scatter (~0.02 ns at ~3500 IRF photons/pixel);
on fluorescein (40 MHz) the lifetime spread fell from 0.43 ns (no map,
pooled IRF) to 0.20 ns -- the same as a correct calibration map with the
pooled IRF (0.18 ns). Cost: one DFT of the IRF cube (< 0.1 s).

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

Single-lifetime mode (--decay-model single)
-------------------------------------------
For a sample with ONE lifetime (that may change across the image or over
time) every pixel's phasor should lie on the single-lifetime curve: the
phasors of pure monoexponential decays of every lifetime, computed
numerically through the same window as the data (the universal semicircle
in IRF-corrected mode; a different, known curve for a tail window). Each
pixel is projected onto the nearest point of that curve:
  tau_single -- lifetime of that point (ns). A real lifetime in BOTH window
                modes, unlike tau_phase/tau_mod in tail mode. Uses G and S
                together, so it is less noisy than the phase alone.
  off_curve  -- signed distance from the curve (phasor units). ~0 within
                noise for a truly single-exponential pixel; positive = on
                the inner side, where mixtures of lifetimes (including a
                zero-lifetime scatter/reflection) fall; negative = outer
                side (e.g. under-subtracted background in tail mode, or a
                timing/IRF error). Check it before trusting tau_single.
Background in this mode is ONE image-wide level per bin: the constant of a
biexponential + constant fit of the pooled masked tail, divided by the
number of pixels. It ASSUMES the background is the same in every pixel
(logged). Why not per pixel: the lifetime varies from pixel to pixel, so
one fixed decay shape biases B -- and with it tau -- wherever the lifetime
differs from the pooled value; from tail bins a per-pixel B cannot be
separated from a per-pixel lifetime; and iterating B with each pixel's own
tau_single runs away (B grows, points leave the curve) as soon as a pixel
is not truly single-exponential (tested on Flipper-TR data).
An optional reference lifetime (--tau-ref) is marked on the plot.

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
else a pooled biexponential fit). Single-lifetime mode differs, see above. With the lifetimes fixed this is linear,
so all pixels are solved in one least-squares step -- no optimizer, and B
is unbiased. The decay amplitudes are discarded; the phasor itself stays
model-free.

Outputs (per input file)
-------------------------
  <prefix>_phasor_scatter.png   -- 2D density phasor plot
  <prefix>_phasor_maps.png      -- spatial maps of tau_phase, tau_mod, f_long
                                   (single-lifetime mode: tau_single, off_curve)
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
# The GUI may pass these as str ("" = default), int or bool -- parsed by
# _extra_args() below. The time window is not an extra arg: it follows the
# common fit_method ("tail" -> tail window, "irf" -> full window + IRF).
EXTRA_ARGS = ["harmonic", "rep_rate_mhz", "subtract_background", "decay_model",
              "tau_ref", "tau_ref1", "tau_ref2", "per_pixel_irf"]

# Which common GUI settings apply (see analyses/_registry.py): no lifetime
# model, bounds or display range; the tail offset is also used in IRF mode
# (background estimate); only overview and results plots are produced.
GUI_OPTIONS = dict(
    fit_methods={"tail": "Tail window", "irf": "Full decay + IRF correction"},
    fit_method_title="Decay window",
    tail_offset_with_irf=True,
    uses_model=False,
    uses_lifetime_range=False,
    outputs=("save_overview", "save_results"),
)


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
    if getattr(args, "window_mode", None):
        raise ValueError("window_mode was replaced by fit_method: use fit_method='tail' "
                         "for the tail window or fit_method='irf' (with an IRF file) "
                         "for the full window with IRF correction.")
    fit_method = getattr(args, "fit_method", None) or "tail"
    if fit_method == "tail":
        window_mode = "tail"
    elif fit_method == "irf":
        if not getattr(args, "irf", None):
            raise ValueError("fit_method='irf' needs an IRF file (the full-window phasor "
                             "is only meaningful after IRF correction).")
        window_mode = "full"
    else:
        raise ValueError(f"Phasor supports fit_method 'tail' or 'irf', got {fit_method!r}.")
    dm = getattr(args, "decay_model", None)
    dm = "bi" if dm is None or not str(dm).strip() else str(dm).strip().lower()
    if dm in ("single", "mono", "1", "single lifetime"):
        decay_model = "single"
    elif dm in ("bi", "two", "2", "two lifetimes"):
        decay_model = "bi"
    else:
        raise ValueError(f"decay_model = {dm!r}: use 'single' or 'bi'")
    tau_ref = _num("tau_ref", float, None) if decay_model == "single" else None
    if tau_ref is not None and tau_ref <= 0:
        raise ValueError(f"tau_ref must be > 0, got {tau_ref}")
    tau_ref1 = _num("tau_ref1", float, None) if decay_model == "bi" else None
    tau_ref2 = _num("tau_ref2", float, None) if decay_model == "bi" else None
    if (tau_ref1 is None) != (tau_ref2 is None):
        print("  NOTE: only one of tau_ref1/tau_ref2 given -- both are needed for "
              "the tie-line and f_long; skipping them.")
        tau_ref1 = tau_ref2 = None
    rep_rate_mhz = _num("rep_rate_mhz", float, None)
    if rep_rate_mhz is not None and rep_rate_mhz <= 0:
        raise ValueError(f"rep_rate_mhz must be > 0, got {rep_rate_mhz}")
    def _flag(name, default):
        v = getattr(args, name, None)
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        if isinstance(v, str):
            if v.strip().lower() in ("1", "true", "yes", "y", "on"):
                return True
            if v.strip().lower() in ("0", "false", "no", "n", "off"):
                return False
            raise ValueError(f"{name} = {v!r}: use yes/no")
        return bool(v)

    subtract_bg = _flag("subtract_background", True)
    per_pixel_irf = _flag("per_pixel_irf", False)
    if per_pixel_irf and window_mode != "full":
        print("  NOTE: per_pixel_irf only applies with fit_method='irf' -- ignored.")
        per_pixel_irf = False
    return dict(harmonic=harmonic, window_mode=window_mode, decay_model=decay_model,
                tau_ref=tau_ref, tau_ref1=tau_ref1, tau_ref2=tau_ref2,
                rep_rate_mhz=rep_rate_mhz, subtract_bg=subtract_bg,
                per_pixel_irf=per_pixel_irf)


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
    B[mask] = coef[-1]

    n_tot = cube[mask].sum(axis=1)
    frac = coef[-1] * nb / np.where(n_tot > 0, n_tot, np.nan)
    print(f"  Background: per-pixel fit in the tail (decay shapes {taus[0]:.2f} / "
          f"{taus[1]:.2f} ns from {src}): median {np.median(coef[-1]):.2f} counts/bin "
          f"= {100 * np.nanmedian(frac):.1f}% of all photons")
    return B, info


def uniform_background(cube, mask, s, tail_offset):
    """One image-wide background level (counts per bin per pixel): the
    constant of a biexponential + constant fit of the pooled masked tail,
    divided by the number of masked pixels. Pooling makes it precise, and
    the two exponentials absorb a spread of lifetimes, so it is not biased
    by lifetime differences between pixels -- but it assumes the background
    is the same in every pixel. Returns (B map or None, info dict)."""
    pooled = cube[mask].sum(axis=0).astype(float)
    _, _pk, s_idx, tn = _tail_axis(s, pooled[None, None, :], offset=tail_offset)
    info = dict(taus=None, n_tail_bins=cube.shape[2] - s_idx)
    try:
        g = global_fit(pooled, s, s_idx, tn, mono=False, fit_background=True)
    except Exception as exc:
        print(f"  WARNING: pooled fit for the background failed ({exc}); "
              f"background subtraction skipped.")
        return None, info
    info["taus"] = (float(g["tau_short"]), float(g["tau_long"]))
    level = float(g["popt"][-1]) / int(mask.sum())
    B = np.zeros(cube.shape[:2])
    B[mask] = level
    n_tot = cube[mask].sum(axis=1)
    print(f"  Background: one image-wide level from a pooled biexponential + constant "
          f"fit: {level:.2f} counts/bin per pixel = "
          f"{100 * np.median(level * cube.shape[2] / np.maximum(n_tot, 1)):.1f}% of all photons "
          f"(median pixel)")
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


def _mono_phasors(taus_ns, t_abs_win, t0_ns, T_laser_ns, harmonic=1, continuous=False):
    """Phasors of pure periodic monoexponential decays seen through the data's
    window, for an array of lifetimes. Returns (G, S, w_window), each (n,).

    The decay starts at t0_ns on the data's time axis (left bin edges) and
    repeats every laser period, so photons still arriving from the previous
    pulse are included -- exactly what the camera records.

    continuous=False (tail window, no IRF correction): the data are bin
    counts -- integrals over each bin -- of the decay, transformed on the
    left-edge axis. For an exponential a bin integral is exactly proportional
    to the value at the bin's left edge, so point samples there reproduce the
    data's phasor exactly.

    continuous=True (IRF-corrected full window): dividing by the IRF phasor
    removes the binning together with the IRF (both are binned the same way),
    so the corrected phasor is that of the CONTINUOUS decay. Point samples
    lag it by half a bin of phase (-w*tpb/2: +0.025 ns on tau at 50 ps bins,
    +0.11 ns at 400 ps), so here each bin's decay is integrated exactly with
    its phase: sum_k int_bin e^{-(t-t0)/tau} e^{iwt} dt = point-sample sum
    times (e^{a*tpb} - 1)/a, a = -1/tau + iw.

    w_window -- fraction of this component's photons (per laser period) inside
    the window, from exact bin integrals: sum_k e^{-u_k/tau} (1 - e^{-tpb/tau})
    / (1 - e^{-T/tau}). (value x bin width overstates it by ~tpb/(2 tau): ~10 %
    for 1.4 ns at 300 ps bins). Converts window photon fractions back to
    whole-decay fractions (see _window_to_total_fraction).
    """
    tau = np.atleast_1d(np.asarray(taus_ns, dtype=np.float64))
    omega = 2.0 * np.pi * harmonic / T_laser_ns
    tpb_ns = (t_abs_win[1] - t_abs_win[0]) if len(t_abs_win) > 1 else 0.0
    u = np.mod(t_abs_win - t0_ns, T_laser_ns)                        # time since the latest pulse
    with np.errstate(over="ignore", under="ignore"):
        e = np.exp(-u[None, :] / tau[:, None])                        # (n, n_win)
    tot = e.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (e * np.exp(1j * omega * t_abs_win)[None, :]).sum(axis=1) / tot
        if continuous and tpb_ns > 0:
            a = -1.0 / tau + 1j * omega
            z = z * np.expm1(a * tpb_ns) / (a * tau * -np.expm1(-tpb_ns / tau))
        w_window = tot * -np.expm1(-tpb_ns / tau) / -np.expm1(-T_laser_ns / tau)
    return z.real, z.imag, w_window


def _reference_phasor(tau_ns, t_abs_win, t0_ns, T_laser_ns, harmonic=1, continuous=False):
    """Phasor of one pure monoexponential decay through the data's window --
    see _mono_phasors. Returns (G_ref, S_ref, w_window).

    Needed for the tie-line and f_long projection: a tail-only (or
    shorter-than-period) window moves a pure lifetime's phasor off the
    universal semicircle, so the reference must be computed numerically with
    the same window to be consistent with the measured pixel phasors."""
    G, S, w = _mono_phasors([tau_ns], t_abs_win, t0_ns, T_laser_ns, harmonic, continuous)
    return float(G[0]), float(S[0]), float(w[0])


def single_lifetime_curve(t_abs_win, t0_ns, T_laser_ns, harmonic=1,
                          tau_min_ns=0.01, n=4000, continuous=False):
    """The single-lifetime curve for this window: phasors of pure periodic
    monoexponential decays (same construction as _reference_phasor, see
    _mono_phasors) for a log-spaced grid of lifetimes. Returns (tau, G, S),
    tau increasing.

    In IRF-corrected full-window mode this is the universal semicircle (when
    the window spans the laser period); for a tail window it is a different
    curve, which is why it is computed numerically rather than from
    1/(1 + i w tau)."""
    tau = np.geomspace(tau_min_ns, 5.0 * T_laser_ns, n)
    G, S, _w = _mono_phasors(tau, t_abs_win, t0_ns, T_laser_ns, harmonic, continuous)
    ok = np.isfinite(G) & np.isfinite(S)
    return tau[ok], G[ok], S[ok]


def project_on_curve(G, S, mask, curve_tau, curve_G, curve_S, chunk=4096):
    """Project every masked pixel phasor onto the single-lifetime curve.

    Nearest grid point first, then the exact foot of the perpendicular on
    the neighbouring segments (lifetime interpolated in log tau), so the
    result does not depend on the grid spacing.

    Returns (tau_single, off_curve), both [H, W] float32, NaN outside the
    mask and where the nearest point is an END of the curve grid (the
    pixel lies beyond the range of physical lifetimes).
    off_curve is signed: positive = left of the curve walking towards
    longer lifetimes, i.e. the inner side (for the semicircle: inside it,
    where mixtures of lifetimes fall)."""
    H, W = G.shape
    tau_out = np.full((H, W), np.nan, dtype=np.float32)
    d_out = np.full((H, W), np.nan, dtype=np.float32)
    sel = mask & np.isfinite(G) & np.isfinite(S)
    if not sel.any():
        return tau_out, d_out
    P = np.column_stack([G[sel], S[sel]])
    C = np.column_stack([curve_G, curve_S])
    logtau = np.log(curve_tau)
    n = len(C)
    tau_px = np.empty(len(P))
    d_px = np.empty(len(P))
    for i0 in range(0, len(P), chunk):
        p = P[i0:i0 + chunk]
        d2 = ((p[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        k = np.argmin(d2, axis=1)
        best_d = np.full(len(p), np.inf)
        best_tau = np.full(len(p), np.nan)
        best_sign = np.zeros(len(p))
        for a_off in (-1, 0):                       # segments [k-1, k] and [k, k+1]
            a = np.clip(k + a_off, 0, n - 2)
            b = a + 1
            v = C[b] - C[a]
            w = p - C[a]
            vv = np.maximum((v ** 2).sum(axis=1), 1e-30)
            tt = np.clip((w * v).sum(axis=1) / vv, 0.0, 1.0)
            q = C[a] + tt[:, None] * v
            r = p - q
            dist = np.sqrt((r ** 2).sum(axis=1))
            better = dist < best_d
            best_d[better] = dist[better]
            best_tau[better] = np.exp(logtau[a] + tt * (logtau[b] - logtau[a]))[better]
            cross = v[:, 0] * r[:, 1] - v[:, 1] * r[:, 0]
            best_sign[better] = np.sign(cross)[better]
        at_end = (k == 0) | (k == n - 1)
        best_tau[at_end] = np.nan
        tau_px[i0:i0 + chunk] = best_tau
        d_px[i0:i0 + chunk] = np.where(at_end, np.nan, best_sign * best_d)
    tau_out[sel] = tau_px
    d_out[sel] = d_px
    return tau_out, d_out


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
    the IRF rotated by the physical offset leaves the phasor of the
    CONTINUOUS decay starting at t = 0 (the binning divides out with the
    IRF), which is how the reference phasors are computed in this mode
    (_mono_phasors, continuous=True). With no_shift the physical offset is
    held at 0.

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


def irf_offset_binned(args, signal_fine, tpb_fine_ns, time_bin, irf_binned=None):
    """IRF/data timing offset (physical, see irf_timing_shift) in units of the
    TIME-BINNED grid, measured at NATIVE time resolution.

    The reconvolution model is only accurate when bins are much narrower than
    the IRF: with time_bin > 1 its free shift absorbs model error (on
    noise-free synthetic data, 0.25 ns IRF: -199 ps at 400 ps bins, +281 ps at
    200 ps; true 0). A time shift is the same in ns on any grid, and the data
    and the IRF are both binned from their own first bin, so the offset fitted
    on the unbinned pooled decay and unbinned IRF converts exactly:
    offset_binned = offset_native / time_bin.

    signal_fine: pooled masked decay BEFORE time binning. irf_binned: the IRF
    already loaded at time_bin == 1 (reused then; otherwise it is reloaded
    unbinned). Returns the offset in binned bins, or None if the fit fails."""
    no_shift = getattr(args, "no_shift", False)
    if no_shift:
        print("  IRF/data timing offset held at 0 (no_shift).")
        return 0.0
    if time_bin > 1 or irf_binned is None:
        from flim_core.irf import load_irf
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            irf_f = load_irf(args.irf, window=not getattr(args, "irf_full", False),
                             window_end_ns=getattr(args, "irf_window_end_ns", 15.0), time_bin=1,
                             pixel_offset_path=getattr(args, "pixel_offset", None))
    else:
        irf_f = irf_binned
    print("  Measuring the IRF/data timing offset (one pooled reconvolution fit"
          + (", at native time resolution" if time_bin > 1 else "") + ") ...")
    grid = _irf_on_data_grid(irf_f["irf_norm"], len(signal_fine))
    sh = irf_timing_shift(np.asarray(signal_fine, dtype=float), grid, tpb_fine_ns)
    return None if sh is None else sh / time_bin


def irf_cube_like_data(irf_path, data_hw, pixel_offset=None, roi=None, smooth=None,
                       time_bin=1, integer_offsets=False):
    """The IRF file's per-pixel cube, put through the SAME steps as the data
    so pixel (r, c) of the result sees the same detector pixels as pixel
    (r, c) of the data: pixel-offset correction (if any), ROI crop, spatial
    smoothing (`smooth(img, cube) -> (img, cube)`, the data's own step) and
    time binning. data_hw: the data's (H, W) BEFORE the ROI crop; the IRF
    sensor must match it. integer_offsets: round offset shifts to whole bins,
    as raw photon data are corrected (phasor_timelapse)."""
    s = load_flim_data(irf_path)
    cube = build_decay_cube(s)
    if cube.shape[:2] != tuple(data_hw):
        raise ValueError(f"per-pixel IRF: the IRF file's sensor {cube.shape[:2]} does not match "
                         f"the data's {tuple(data_hw)}.")
    if pixel_offset:
        corr = PixelOffsetCorrection.load(pixel_offset)
        cube = corr.apply(cube, s["tpb"], integer_bins=integer_offsets)
    img = cube.sum(axis=2)
    if roi is not None:
        img, cube, _ = crop_to_roi(img, cube, roi)
    if smooth is not None:
        img, cube = smooth(img, cube)
    tpb = s["tpb"]
    if time_bin and time_bin > 1:
        cube, tpb = bin_time_cube(cube, time_bin, tpb)
    return cube, tpb


def per_pixel_irf_phasor(irf_cube, irf, tpb_ns, nb, T_laser_ns, harmonic, shift_bins,
                         min_counts=100, margin_ns=0.5):
    """Each pixel's own IRF phasor on the data's bin grid, rotated by the
    pooled physical IRF/data offset (shift_bins), like the pooled one in
    apply_irf_correction.

    irf_cube: from irf_cube_like_data (same grid as the data). irf: the
    load_irf dict (same time_bin) -- gives the kernel window and the start
    of the background region. Per pixel: background = mean of its far tail
    (bins >= irf["bg_start"]), subtracted without clipping; bins kept = the
    pooled window +- margin_ns (the pixel's own timing may sit off the
    pooled window by a few hundred ps).

    Returns (z, n): z (H, W) complex, NaN where the pixel has < min_counts
    background-subtracted IRF photons; n (H, W) those photon counts.
    """
    omega = 2.0 * np.pi * harmonic / T_laser_ns
    nbi = min(nb, irf_cube.shape[2])       # bins shared with the data grid
    c = irf_cube.astype(np.float64)
    bg_start = min(int(irf["bg_start"]), c.shape[2] - 1)
    bg = c[:, :, bg_start:].mean(axis=2)
    m = int(np.ceil(margin_ns / tpb_ns))
    lo = max(int(irf["window"][0]) - m, 0)
    hi = min(int(irf["window"][1]) + m + 1, nbi)
    h = c[:, :, lo:hi] - bg[:, :, None]
    t = np.arange(lo, hi) * tpb_ns
    n = h.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (h * np.exp(1j * omega * t)).sum(axis=2) / n
    z = z * np.exp(1j * omega * shift_bins * tpb_ns)
    z[~(n >= min_counts) | ~np.isfinite(z) | (np.abs(z) < 1e-10)] = np.nan
    return z, n


def reference_per_pixel_irf(z_pix, irf_grid, tpb_ns, T_laser_ns, harmonic, shift_bins,
                            weights, mask):
    """Re-reference per-pixel IRF phasors to the pooled timing fit.

    The IRF/data offset (irf_timing_shift) is fitted between the pooled
    MASKED data and the WHOLE-SENSOR pooled IRF kernel, so it also absorbs
    how much later/earlier the masked pixels' IRFs are on average than the
    whole sensor's (+15-17 ps on the 2026-11-17 fluorescein data). Per-pixel
    IRFs already contain each pixel's own timing; applying the fitted shift
    to them as well would count that difference twice (it lowered tau_phase
    by ~0.03 ns there). So all z_pix are rotated together until their
    data-weighted mean (weights = the pixels' data counts, as in the pooled
    fit) has the kernel's phase. Phase only; pixel-to-pixel differences are
    untouched. Returns (z_pix_rotated, delta_ps) with delta_ps the removed
    mean difference.
    """
    omega = 2.0 * np.pi * harmonic / T_laser_ns
    k = irf_grid.astype(np.float64)
    zk = (np.sum(k * np.exp(1j * omega * np.arange(len(k)) * tpb_ns)) / k.sum()
          * np.exp(1j * omega * shift_bins * tpb_ns))
    ok = mask & np.isfinite(z_pix) & (weights > 0)
    if not ok.any():
        return z_pix, 0.0
    zm = np.sum(weights[ok] * z_pix[ok]) / np.sum(weights[ok])
    rot = zk / zm
    rot /= abs(rot)
    return z_pix * rot, float(-np.angle(rot) / omega * 1000.0)


def apply_irf_correction(result, irf_grid, tpb_ns, shift_bins, harmonic=1,
                         z_irf_pixels=None, mask=None):
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
    z_irf_pixels: optional per-pixel IRF phasors (per_pixel_irf_phasor,
    already rotated by the offset), same shape as result["G"]. Each pixel is
    then divided by its own; NaN entries fall back to the pooled one. mask
    (optional) only limits which pixels the log statistics describe.
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
    if z_irf_pixels is None:
        z_corr = z / z_irf
    else:
        zp = np.asarray(z_irf_pixels)
        sel = np.isfinite(z) if mask is None else (mask & np.isfinite(z))
        own = sel & np.isfinite(zp)
        n_fb = int((sel & ~np.isfinite(zp)).sum())
        if own.any():
            dt_ps = np.angle(zp[own] / z_irf) / omega * 1000.0
            print(f"    per-pixel IRF: {int(own.sum())} pixels use their own IRF; timing "
                  f"spread of the per-pixel IRFs (removed by this) {np.std(dt_ps):.0f} ps rms")
        if n_fb:
            print(f"    per-pixel IRF: {n_fb} pixel(s) have too few IRF photons -> pooled IRF")
        z_corr = z / np.where(np.isfinite(zp), zp, z_irf)

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

def _draw_pixel_cloud(ax, G, S, cmap="viridis", alpha=0.9, zorder=3, bins_across=220):
    """Log-density of pixel phasors with bins sized relative to the AXES (about
    bins_across bins across the x-range), not to the cloud: a 100x100 grid
    over a tight cloud (e.g. a single lifetime, spread ~0.03) makes every bin
    smaller than a screen pixel and the cloud vanishes from the figure. Call
    after the axis limits are set."""
    ok = np.isfinite(G) & np.isfinite(S)
    G, S = G[ok], S[ok]
    if len(G) == 0:
        return
    x0, x1 = ax.get_xlim()
    bw = (x1 - x0) / bins_across
    g_lo, g_hi = np.percentile(G, [0.1, 99.9])
    s_lo, s_hi = np.percentile(S, [0.1, 99.9])
    xe = np.arange(g_lo - bw, g_hi + 2 * bw, bw)
    ye = np.arange(s_lo - bw, s_hi + 2 * bw, bw)
    h2d, xe, ye = np.histogram2d(G, S, bins=[xe, ye])
    ax.pcolormesh(xe, ye, np.where(h2d > 0, np.log1p(h2d), np.nan).T,
                  cmap=cmap, alpha=alpha, zorder=zorder)


def _nice_tau_ticks(lo, hi, max_ticks=7):
    """Round lifetime values (ns) between lo and hi for curve ticks."""
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0):
        ticks = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
        if len(ticks) <= max_ticks:
            return [round(x, 3) for x in ticks]
    return []


def _add_zoom_inset(ax, G, S, curve=None, refs=(), tie=None, min_frac=0.15):
    """When the pixel cloud is small compared with the plot, add a zoomed
    inset around it: pixel density, the single-lifetime curve (or universal
    semicircle) with fine lifetime ticks, any reference points / tie-line,
    and the cloud median. Returns the inset axes or None.

    curve: (tau, G, S) or None (-> universal semicircle at this omega is
    passed in as a curve by the caller). refs: [(G, S, marker, color, label)].
    tie: ((G1, S1), (G2, S2)) or None."""
    ok = np.isfinite(G) & np.isfinite(S)
    G, S = G[ok], S[ok]
    if len(G) < 2:
        return None
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    g_lo, g_hi = np.percentile(G, [0.5, 99.5])
    s_lo, s_hi = np.percentile(S, [0.5, 99.5])
    if (g_hi - g_lo) > min_frac * (x1 - x0) or (s_hi - s_lo) > min_frac * (y1 - y0):
        return None
    gm, sm = float(np.median(G)), float(np.median(S))
    half = 0.65 * max(g_hi - g_lo, s_hi - s_lo, 0.01)
    zx, zy = (gm - half, gm + half), (sm - half, sm + half)

    # put the inset in the lower part of the plot, away from the cloud
    fx = (gm - x0) / (x1 - x0)
    left = 0.06 if fx > 0.5 else 0.52
    ins = ax.inset_axes([left, 0.06, 0.42, 0.42])
    if curve is not None:
        c_tau, c_G, c_S = curve
        ins.plot(c_G, c_S, color="#7B1FA2", lw=1.2, zorder=2)
        sel = (c_G > zx[0]) & (c_G < zx[1]) & (c_S > zy[0]) & (c_S < zy[1])
        if sel.any():
            for tau in _nice_tau_ticks(c_tau[sel].min(), c_tau[sel].max()):
                Gt = np.interp(np.log(tau), np.log(c_tau), c_G)
                St = np.interp(np.log(tau), np.log(c_tau), c_S)
                ins.plot(Gt, St, "o", ms=3, color="#7B1FA2", zorder=2)
                ins.annotate(f"{tau:g}", (Gt, St), textcoords="offset points", xytext=(4, 2),
                             fontsize=6.5, color="#7B1FA2", zorder=7,
                             bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    if tie is not None:
        (G1, S1), (G2, S2) = tie
        ins.plot([G1, G2], [S1, S2], "--", color="#E05C3A", lw=1.2, zorder=4)
    for (Gr, Sr, mk, col, _lbl) in refs:
        ins.plot(Gr, Sr, mk, mfc="none", mec=col, mew=1.5, ms=8, zorder=5)
    ins.set_xlim(*zx)
    ins.set_ylim(*zy)
    ins.set_aspect("equal")
    _draw_pixel_cloud(ins, G, S, bins_across=60)
    ins.plot(gm, sm, "+", color="k", ms=9, mew=1.2, zorder=6)
    ins.tick_params(labelsize=6)
    ins.set_title("zoom (+ = median)", fontsize=7)
    ax.indicate_inset_zoom(ins, edgecolor="#555555")
    return ins


def plot_phasor_scatter(result, mask, prefix, tau_ref1=None, tau_ref2=None,
                        G1_ref=None, S1_ref=None, G2_ref=None, S2_ref=None,
                        show=True, curve=None, tau_ref=None, G_ref=None, S_ref=None):
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
    if curve is None:
        for tau in tick_taus:
            Gt = 1.0 / (1 + (omega * tau)**2)
            St = omega * tau / (1 + (omega * tau)**2)
            ax.plot(Gt, St, "o", ms=2.5, color="#bbbbbb", zorder=2)
            ax.annotate(f"{tau}", (Gt, St), textcoords="offset points",
                        xytext=(2, 2), fontsize=5.5, color="#999999")
    else:
        # Single-lifetime curve for THIS window, with lifetime ticks on it
        c_tau, c_G, c_S = curve
        ax.plot(c_G, c_S, color="#7B1FA2", lw=1.3, zorder=2,
                label="single-lifetime curve (this window)")
        for tau in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
            Gt, St = np.interp(np.log(tau), np.log(c_tau), c_G), \
                     np.interp(np.log(tau), np.log(c_tau), c_S)
            ax.plot(Gt, St, "o", ms=2.5, color="#7B1FA2", zorder=2)
            ax.annotate(f"{tau:g}", (Gt, St), textcoords="offset points",
                        xytext=(3, 2), fontsize=6, color="#7B1FA2")

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
    if G_ref is not None:
        g_pts += [G_ref - 0.05, G_ref + 0.05]
        s_pts += [S_ref - 0.05, S_ref + 0.05]
    xlim = (min(g_pts), max(g_pts))
    ylim = (min(s_pts), max(s_pts))

    # Reference points and tie-line (computed numerically for this window)
    if tau_ref1 is not None and tau_ref2 is not None and G1_ref is not None:
        ax.plot([G1_ref, G2_ref], [S1_ref, S2_ref], "--",
                color="#E05C3A", lw=1.5, alpha=0.9, zorder=4,
                label=f"tie-line τ={tau_ref1}–{tau_ref2} ns")
        ax.plot(G1_ref, S1_ref, "^", color="#2196F3", ms=9, zorder=5,
                label=f"τ_short = {tau_ref1} ns")
        ax.plot(G2_ref, S2_ref, "s", color="#E05C3A", ms=9, zorder=5,
                label=f"τ_long  = {tau_ref2} ns")

    if G_ref is not None:
        ax.plot(G_ref, S_ref, "D", color="#E05C3A", ms=8, zorder=5,
                label=f"τ_ref = {tau_ref} ns")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("G  (cosine component)", fontsize=11)
    ax.set_ylabel("S  (sine component)", fontsize=11)
    ax.set_aspect("equal")

    # Pixel density (after the limits: bin size follows the axes, see helper)
    _draw_pixel_cloud(ax, G_px, S_px)
    zoom_curve = curve
    if zoom_curve is None:
        tt = np.geomspace(0.01, 5 * T, 3000)
        zoom_curve = (tt, 1 / (1 + (omega * tt) ** 2), omega * tt / (1 + (omega * tt) ** 2))
    refs = []
    if G1_ref is not None:
        refs += [(G1_ref, S1_ref, "^", "#2196F3", ""), (G2_ref, S2_ref, "s", "#E05C3A", "")]
    if G_ref is not None:
        refs.append((G_ref, S_ref, "D", "#E05C3A", ""))
    _add_zoom_inset(ax, G_px, S_px, curve=zoom_curve, refs=refs,
                    tie=((G1_ref, S1_ref), (G2_ref, S2_ref)) if G1_ref is not None else None)
    irf_tag = ((" + per-pixel IRF corr." if result.get("irf_per_pixel") else " + IRF corr.")
               if result.get("irf_corrected") else "")
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
                     tau_ref2=None, show=True, tau_single=None, off_curve=None):
    """Spatial maps of tau_phase, tau_mod, and optionally f_long -- or, in
    single-lifetime mode, tau_single and off_curve."""
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

    if tau_single is not None:
        _show(axes[0], tau_single, "τ_single  (ns)", cmap="plasma")
        oc = off_curve[mask & np.isfinite(off_curve)]
        lim = float(np.percentile(np.abs(oc), 98)) if oc.size else 0.05
        lim = max(lim, 1e-3)
        _show(axes[1], off_curve, "distance from single-lifetime curve\n(+ = inner side)",
              cmap="RdBu_r", vmin=-lim, vmax=lim, unit="phasor units")
    else:
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
    ea = _extra_args(args)
    harmonic, window_mode, decay_model = ea["harmonic"], ea["window_mode"], ea["decay_model"]
    tau_ref, tau_ref1, tau_ref2 = ea["tau_ref"], ea["tau_ref1"], ea["tau_ref2"]
    rep_rate_mhz, subtract_bg = ea["rep_rate_mhz"], ea["subtract_bg"]
    per_pixel_irf = ea["per_pixel_irf"]
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
    sensor_hw = cube.shape[:2]          # before ROI -- the per-pixel IRF must match it

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
            corr.check_settings(rep_rate_mhz)
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
    cube_fine, tpb_fine_ns = cube, s["tpb"] * 1e9      # unbinned, for the IRF offset fit
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
        if decay_model == "single":
            bg_map, bg_info = uniform_background(cube, mask, s, tail_offset)
            if bg_map is not None:
                print("    NOTE: single-lifetime mode takes the background as the same in "
                      "every pixel (a per-pixel level cannot be separated reliably from a "
                      "lifetime that varies per pixel).")
        else:
            bg_map, bg_info = estimate_background(
                cube, mask, s, tail_offset, T_laser_ns,
                taus=(tau_ref1, tau_ref2) if tau_ref1 is not None else None)
    else:
        print("  Background subtraction off.")

    # ---- optional IRF correction: timing offset measured once ----
    irf_grid = irf_shift = None
    if irf is not None and window_mode == "tail":
        # Dividing by the IRF phasor is only valid when the data phasor is the
        # transform of the FULL convolved decay (IRF * decay). A tail-only
        # phasor is not, so the division would give a wrong (G, S).
        print("  NOTE: IRF file ignored -- IRF correction needs the full window "
              "(fit_method='irf').")
    elif irf is not None:
        irf_grid = _irf_on_data_grid(irf["irf_norm"], nb)
        irf_shift = irf_offset_binned(args, cube_fine[mask].sum(axis=0), tpb_fine_ns,
                                      time_bin, irf_binned=irf)
        if irf_shift is None:
            print("  WARNING: IRF correction skipped -- the timing offset could not be "
                  "measured, and correcting with a wrong offset would bias every lifetime.")
        else:
            print(f"    IRF/data timing offset: {irf_shift:+.2f} bins "
                  f"({irf_shift * tpb_ns * 1000:+.0f} ps; positive = data lag the IRF)")
            if nb * tpb_ns < 0.99 * T_laser_ns:
                print("  WARNING: the histogram window is shorter than the laser period, "
                      "so the data are not one full period -- IRF correction is approximate.")

    # ---- optional per-pixel IRF phasors (same pixels/grid as the data) ----
    z_irf_pix = n_irf_pix = None
    if per_pixel_irf and irf_shift is not None:
        print("  Per-pixel IRF: building each pixel's IRF phasor from the IRF file ...")
        def _smooth(i, c):
            if moving_avg and moving_avg > 1:
                return moving_average_cube(i, c, moving_avg)
            return bin_cube(i, c, bin_size) if bin_size > 1 else (i, c)
        irf_cube, _ = irf_cube_like_data(
            args.irf, sensor_hw, pixel_offset=getattr(args, "pixel_offset", None),
            roi=roi, smooth=_smooth, time_bin=time_bin)
        if irf_cube.shape[:2] != cube.shape[:2]:
            raise ValueError(f"per-pixel IRF cube {irf_cube.shape[:2]} does not match the "
                             f"data cube {cube.shape[:2]} after ROI/binning.")
        z_irf_pix, n_irf_pix = per_pixel_irf_phasor(
            irf_cube, irf, tpb_ns, nb, T_laser_ns, harmonic, irf_shift)
        del irf_cube
        z_irf_pix, delta_ps = reference_per_pixel_irf(
            z_irf_pix, irf_grid, tpb_ns, T_laser_ns, harmonic, irf_shift,
            weights=cube.sum(axis=2), mask=mask)
        print(f"    masked pixels' IRFs are {delta_ps:+.1f} ps vs the whole-sensor IRF on "
              f"average; already in the fitted offset, so removed from the per-pixel IRFs")
        n_med = float(np.median(n_irf_pix[mask]))
        # phase noise of a phasor from N photons ~ 1/(|z| sqrt(2N)) rad
        noise_ps = 1000.0 / (2 * np.pi * harmonic / T_laser_ns) / np.sqrt(2 * max(n_med, 1.0))
        print(f"    median {n_med:.0f} IRF photons per pixel -> the IRF's own photon noise "
              f"adds ~{noise_ps:.0f} ps timing scatter per pixel")
    elif per_pixel_irf:
        print("  NOTE: per-pixel IRF skipped -- the IRF correction itself is off.")

    # ---- compute phasor ----
    print(f"  Phasor: harmonic={harmonic}, window='{window_mode}' [{win_label}], "
          f"T={T_laser_ns:.2f} ns, decay model '{decay_model}' ...")
    result = compute_phasor(cube_win, t_abs_win, T_laser_ns, mask, harmonic=harmonic,
                            bg_per_bin=bg_map)
    result["irf_corrected"] = False
    if irf_shift is not None:
        print("  Applying IRF correction in phasor space ...")
        apply_irf_correction(result, irf_grid, tpb_ns, irf_shift, harmonic=harmonic,
                             z_irf_pixels=z_irf_pix, mask=mask)
        result["irf_shift_bins"] = irf_shift
        result["irf_per_pixel"] = z_irf_pix is not None

    del cube_fine

    # ---- reference phasor points and f_long ----
    G1_ref = S1_ref = G2_ref = S2_ref = None
    f_long_map = f_long_win_map = None
    w1 = w2 = None
    G_ref = S_ref = None
    curve = tau_single_map = off_curve_map = None

    # Excitation time on the data's time axis. After IRF correction the
    # decay starts at the IRF's time origin, t = 0, and the reference is the
    # continuous decay (see _mono_phasors). Without it, the decay peak is
    # used -- a slight approximation, because the true excitation is a little
    # earlier than the peak (by roughly the IRF rise), which slightly biases
    # the tail-window weights.
    irf_corr = result["irf_corrected"]
    t0_ns = 0.0 if irf_corr else (peak_bin - s["B0"]) * tpb_ns

    if decay_model == "single":
        curve = single_lifetime_curve(t_abs_win, t0_ns, T_laser_ns, harmonic,
                                      continuous=irf_corr)
        print(f"  Projecting pixels onto the single-lifetime curve (this window, "
              f"decay start t0={t0_ns:.2f} ns) ...")
        tau_single_map, off_curve_map = project_on_curve(
            result["G"], result["S"], mask, *curve)
        n_end = int((mask & np.isfinite(result["G"]) & ~np.isfinite(tau_single_map)).sum())
        if n_end:
            print(f"    {n_end} pixel(s) lie beyond the ends of the curve -> NaN")
        if tau_ref is not None:
            G_ref, S_ref, _w = _reference_phasor(tau_ref, t_abs_win, t0_ns, T_laser_ns, harmonic,
                                                 continuous=irf_corr)
            print(f"    tau_ref={tau_ref} ns -> G={G_ref:.4f}, S={S_ref:.4f}")
        for lbl, arr in (("tau_single", tau_single_map), ("off_curve", off_curve_map)):
            v = arr[np.isfinite(arr)]
            if v.size:
                print(f"    {lbl}: median={np.median(v):.4f}, std={np.std(v):.4f}, "
                      f"[{np.percentile(v, 2):.4f} to {np.percentile(v, 98):.4f}]")

    if tau_ref1 is not None and tau_ref2 is not None:
        G1_ref, S1_ref, w1 = _reference_phasor(tau_ref1, t_abs_win, t0_ns, T_laser_ns, harmonic,
                                               continuous=irf_corr)
        G2_ref, S2_ref, w2 = _reference_phasor(tau_ref2, t_abs_win, t0_ns, T_laser_ns, harmonic,
                                               continuous=irf_corr)
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
                            show=show, curve=curve,
                            tau_ref=tau_ref, G_ref=G_ref, S_ref=S_ref)
        plot_phasor_maps(result, mask, prefix, f_long=f_long_map,
                         tau_ref1=tau_ref1, tau_ref2=tau_ref2, show=show,
                         tau_single=tau_single_map, off_curve=off_curve_map)

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
        decay_model      = decay_model,
        t0_ns            = np.float64(t0_ns),
        irf_corrected    = np.bool_(result["irf_corrected"]),
        irf_per_pixel    = np.bool_(z_irf_pix is not None and result["irf_corrected"]),
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
    if n_irf_pix is not None:
        save_kw["irf_photons_per_pixel"] = n_irf_pix.astype(np.float32)
    if bg_map is not None:
        save_kw["background_per_bin"] = bg_map.astype(np.float32)
        save_kw["background_taus_ns"] = np.array(bg_info["taus"], dtype=np.float64)
    if f_long_map is not None:
        save_kw["f_long"]     = f_long_map
        save_kw["f_long_window"] = f_long_win_map
        save_kw["w_window_ref1"] = np.float64(w1)
        save_kw["w_window_ref2"] = np.float64(w2)
        save_kw["tau_ref1_ns"] = np.float64(tau_ref1)
        save_kw["tau_ref2_ns"] = np.float64(tau_ref2)
        save_kw["G_ref1"]     = np.float64(G1_ref)
        save_kw["S_ref1"]     = np.float64(S1_ref)
        save_kw["G_ref2"]     = np.float64(G2_ref)
        save_kw["S_ref2"]     = np.float64(S2_ref)
    if tau_single_map is not None:
        save_kw["tau_single"] = tau_single_map
        save_kw["off_curve"]  = off_curve_map
        c_tau, c_G, c_S = curve
        keep = np.unique(np.linspace(0, len(c_tau) - 1, 500).astype(int))
        save_kw["curve_tau_ns"] = c_tau[keep]
        save_kw["curve_G"]      = c_G[keep]
        save_kw["curve_S"]      = c_S[keep]
    if G_ref is not None:
        save_kw["tau_ref_ns"] = np.float64(tau_ref)
        save_kw["G_ref"]      = np.float64(G_ref)
        save_kw["S_ref"]      = np.float64(S_ref)
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
    p.add_argument("--fit-method", choices=["tail", "irf"], default="tail",
                   help="Time window: 'tail' (IRF-free, default) or 'irf' "
                        "(whole TCSPC window + IRF correction; needs --irf)")
    p.add_argument("--decay-model", choices=["bi", "single"], default="bi",
                   help="'bi' (default): two reference lifetimes, tie-line, f_long. "
                        "'single': project onto the single-lifetime curve -> "
                        "tau_single + off_curve maps")
    p.add_argument("--tau-ref", type=float, default=None,
                   help="Single-lifetime mode: optional reference lifetime (ns), "
                        "marked on the phasor plot")
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
    p.add_argument("--per-pixel-irf", action="store_true",
                   help="Divide each pixel by its own IRF phasor (cancels per-pixel "
                        "timing offsets; --fit-method irf only)")

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
