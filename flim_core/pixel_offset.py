"""
Per-pixel SPAD timing offset correction.

SPAD arrays have a fixed, pixel-specific timing offset caused by within-cluster
TDC skew that the manufacturer's factory calibration does not fully remove.  On
the Flimera camera used here the residual spread is ~100-180 ps rms and is
reproducible across objectives, samples, focus and ND filter at FIXED laser /
sync settings (r = 0.99-1.00 between independent foil IRFs).

THE MAP DEPENDS ON THE ACQUISITION SETTINGS. Changing the laser repetition
rate and/or the sync delay changes the pattern completely (measured
2026-11-17: 20 MHz / delay 2880 vs 40 MHz / delay 3560 maps correlate at
r = -0.6; the 20 MHz map applied to 40 MHz data widened the pooled IRF from
~490 to ~670 ps instead of narrowing it to ~190 ps). Build one map per
setting (build_calibration / the CLI below) and record the setting in the
sidecar JSON (rep_rate_mhz, sync_delay). load_irf() checks every map against
the IRF it is applied to (check_against_irf) and warns when it does not fit.

Building a map (from a foil/scatter IRF taken with the data's settings):

    python -m flim_core.pixel_offset IRF.ds3d --out calibration/NAME.npy \
        --rep-rate-mhz 40 --sync-delay 3560

Correcting for it before fitting meaningfully reduces the residual timing error
on each pixel from ~22 ps (uncorrected, 1 s IRF) to ~8 ps (corrected with the
120 s calibration map), improving per-pixel lifetime accuracy especially for
short-lifetime components where the IRF overlap is largest.

Usage
-----
Load the calibration map once, then apply it to every decay cube before fitting:

    from flim_core.pixel_offset import PixelOffsetCorrection
    corr = PixelOffsetCorrection.load("spad_pixel_offset_calibration.npy")
    cube_corrected = corr.apply(cube, tpb)   # cube: (H, W, nb), tpb: seconds/bin

The corrected cube has the same shape and dtype as the input -- each pixel's
decay histogram is circularly shifted by its own offset so that all peaks
align to the sensor-wide median.

Design decisions
----------------
- The offset map is zero-centred (stored as Δt in ps relative to the median),
  so applying it is equivalent to aligning every pixel to a common reference
  rather than to an absolute time.  This preserves the data's own timing
  reference and means the correction is a pure histogram shift with no
  information added or removed.

- Circular shift via FFT phase ramp (the same fractional-bin shift already
  used in irf.shift_signal_subbin) is used per pixel.  Integer rounding would
  introduce ±0.5 bin quantisation error on every pixel; the sub-bin FFT shift
  brings this down to the level of the calibration noise (~8 ps / 37.4 ps/bin
  ≈ 0.2 bins).

- NaN pixels in the calibration map (insufficient photons during the IRF
  measurement) are left uncorrected -- the cube is returned unchanged for
  those pixels.  A warning is printed once on load so the user knows which
  fraction of pixels will be skipped.

- The correction is completely independent of fit method (tail, IRF
  reconvolution, mean arrival time) and spatial binning.  Apply it once
  immediately after load_flim_data, before any binning or masking, so the
  corrected histograms flow through all downstream steps unchanged.

Enabling / disabling
--------------------
The correction is opt-in: call apply() to use it, skip the call to leave the
cube unchanged.  In the GUI this is controlled by the "Pixel offset correction"
checkbox in the Fit method settings panel -- when unchecked, apply() is never
called and results are identical to uncorrected behaviour.  This makes it easy
to compare corrected vs uncorrected results on the same data.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PixelOffsetCorrection:
    """Loaded calibration map + apply() method.

    Attributes
    ----------
    offset_ps : ndarray, shape (H, W), dtype float64
        Per-pixel timing offset in picoseconds, relative to the sensor-wide
        median.  NaN where the calibration measurement had insufficient counts.
    tpb_ps : float
        Bin width of the calibration measurement in picoseconds.  Must match
        the data's bin width; a mismatch is flagged as a warning.
    valid_mask : ndarray bool, shape (H, W)
        True for pixels where the calibration is defined (not NaN).
    n_nan : int
        Number of pixels with no calibration data.
    source : str
        Path string of the file this map was loaded from.
    meta : dict or None
        Contents of the sidecar JSON file if present, else None.
    """

    def __init__(self, offset_ps: np.ndarray, tpb_ps: float,
                 source: str = "", meta: dict | None = None):
        self.offset_ps  = np.asarray(offset_ps, dtype=np.float64)
        self.tpb_ps     = float(tpb_ps)
        self.valid_mask = np.isfinite(self.offset_ps)
        self.n_nan      = int((~self.valid_mask).sum())
        self.source     = source
        self.meta       = meta

        if self.n_nan > 0:
            frac = self.n_nan / self.offset_ps.size * 100
            print(f"  [pixel_offset] calibration map: {self.n_nan} / {self.offset_ps.size} "
                  f"pixels ({frac:.1f}%) have no calibration data (NaN) and will be skipped.")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "PixelOffsetCorrection":
        """Load a calibration map from a .npy file.

        The .npy file must contain a 2-D float array with shape (H, W) — the
        same spatial layout as the decay cube returned by
        flim_core.io.load_flim_data (i.e. after the 90° CCW rotation).
        Values are offsets in picoseconds relative to the sensor-wide median.

        If a sidecar JSON file with the same stem and a ``_meta.json`` suffix
        exists alongside the .npy file, its contents are loaded into
        self.meta so callers can inspect the calibration provenance.

        Parameters
        ----------
        path : str or Path
            Path to the ``*.npy`` calibration file.
        """
        path = Path(path)
        offset_ps = np.load(path)

        if offset_ps.ndim != 2:
            raise ValueError(
                f"Calibration map must be a 2-D array (H, W), "
                f"got shape {offset_ps.shape} from {path}."
            )

        # Optional sidecar metadata
        meta = None
        meta_path = path.with_name(path.stem + "_meta.json")
        if meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
            tpb_ps = float(meta.get("tpb_ps", 37.4))
        else:
            # Fall back to the standard Flimera bin width; warn so the user
            # knows no metadata was found.
            tpb_ps = 37.4
            warnings.warn(
                f"No sidecar metadata file found at {meta_path}. "
                f"Assuming tpb = {tpb_ps} ps. "
                f"Create a _meta.json alongside the .npy to make this explicit.",
                stacklevel=2,
            )

        print(f"  [pixel_offset] loaded calibration map from {path.name} "
              f"| shape {offset_ps.shape} | tpb={tpb_ps} ps "
              f"| offset std={np.nanstd(offset_ps):.1f} ps "
              f"| valid pixels={int(np.isfinite(offset_ps).sum())}/{offset_ps.size}")
        if meta:
            print(f"  [pixel_offset]   source: {meta.get('source_file','?')} "
                  f"| {meta.get('exposure_s','?')} s "
                  f"| noise sigma ~ {meta.get('noise_sigma_ps','?')} ps")
            print(f"  [pixel_offset]   settings: rep. rate "
                  f"{_fmt_setting(meta.get('rep_rate_mhz'), ' MHz')}, sync delay "
                  f"{_fmt_setting(meta.get('sync_delay'))}")

        return cls(offset_ps, tpb_ps, source=str(path), meta=meta)

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def check_settings(self, rep_rate_mhz=None):
        """Warn when the data's laser repetition rate differs from the one
        recorded in the map's metadata (the map depends on it). Silent when
        either is unknown. Returns True when they match or can't be compared."""
        cal = (self.meta or {}).get("rep_rate_mhz")
        if rep_rate_mhz is None or cal is None:
            return True
        if abs(float(cal) - float(rep_rate_mhz)) > 1e-6 * max(abs(float(cal)), 1.0):
            print(f"  [pixel_offset] WARNING: this calibration was made at {float(cal):g} MHz "
                  f"but the data are {float(rep_rate_mhz):g} MHz. The offset pattern changes "
                  f"with the laser/sync settings -- use a map made at the data's settings.")
            return False
        return True

    def check_against_irf(self, irf_cube, tpb, warn_ratio=0.7):
        """Test the map on an IRF measurement (UNcorrected cube): measure each
        pixel's IRF arrival time and compare the spread with and without the
        map. A map made at the same settings removes almost all of it (to the
        ~10 ps estimator noise); a map from other settings leaves most of it
        or makes it worse. Prints the result, warns when the residual is more
        than warn_ratio x the uncorrected spread. Returns a dict."""
        if irf_cube.shape[:2] != self.offset_ps.shape:
            return None
        meas, _cnt = estimate_offsets_from_irf(irf_cube, tpb, iters=2)
        both = np.isfinite(meas) & self.valid_mask
        if both.sum() < 100:
            return None
        m = meas[both] - np.median(meas[both])
        c = self.offset_ps[both] - np.median(self.offset_ps[both])
        before, after = float(np.std(m)), float(np.std(m - c))
        r = float(np.corrcoef(m, c)[0, 1])
        print(f"  [pixel_offset] check on this IRF: per-pixel timing spread {before:.0f} ps "
              f"-> {after:.0f} ps with the map (r = {r:+.2f})")
        ok = after <= warn_ratio * before
        if not ok:
            print(f"  [pixel_offset] WARNING: the calibration map does not fit this IRF "
                  f"(spread {before:.0f} -> {after:.0f} ps). It was probably made at other "
                  f"laser/sync settings; per-pixel timing will be {'worse' if after > before else 'barely better'} "
                  f"with it. Build a map from this IRF (python -m flim_core.pixel_offset).")
        return dict(spread_before_ps=before, spread_after_ps=after, r=r, ok=ok)

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    def apply(self, cube: np.ndarray, tpb: float, integer_bins: bool = False,
              chunk: int = 4096) -> np.ndarray:
        """Shift each pixel's decay histogram to remove its timing offset.

        Each pixel (r, c) is shifted by  -offset_ps[r, c] / tpb_data  bins
        using a sub-bin FFT phase ramp, so that all peaks align to the
        sensor-wide median.  Pixels where the calibration is NaN are returned
        unchanged. All pixels are transformed together (chunks of `chunk`
        pixels), not one by one.

        Parameters
        ----------
        cube : ndarray, shape (H, W, nb), float
            Decay histograms in the orientation produced by load_flim_data
            (after the 90° CCW rotation).
        tpb : float
            Bin width of the data in seconds.
        integer_bins : bool
            Round each shift to whole bins, exactly as the raw photon-stream
            path does (analyses._common.apply_offset_to_photons), so a cube
            can be corrected identically to photon data.

        Returns
        -------
        ndarray, same shape and dtype as cube
            Corrected decay cube.  A copy is always returned; the input is
            never modified in place.

        Raises
        ------
        ValueError
            If cube's spatial shape (H, W) does not match the calibration map.
        """
        H, W, nb = cube.shape
        if (H, W) != self.offset_ps.shape:
            raise ValueError(
                f"Cube spatial shape ({H}, {W}) does not match the calibration "
                f"map shape {self.offset_ps.shape}. "
                f"Check that the same rotation was applied to both, and that "
                f"the cube has not been spatially binned before correction "
                f"(apply pixel_offset correction before bin_cube / moving_average_cube)."
            )

        tpb_ps_data = tpb * 1e12
        if abs(tpb_ps_data - self.tpb_ps) > 1.0:   # >1 ps tolerance
            warnings.warn(
                f"Data tpb ({tpb_ps_data:.2f} ps) differs from calibration tpb "
                f"({self.tpb_ps:.2f} ps). The offset map was built at a different "
                f"timing resolution and shifts will be scaled accordingly, but "
                f"accuracy may be reduced. Ensure the calibration was acquired at "
                f"the same camera settings as the data.",
                stacklevel=2,
            )

        out = cube.copy()   # never modify caller's array
        flat_in = cube.reshape(H * W, nb)
        flat_out = out.reshape(H * W, nb)

        # Shift in bins: negative because we subtract the pixel's own offset
        # to align it to the global median.
        shift = np.zeros(H * W)
        valid = self.valid_mask.ravel()
        shift[valid] = -self.offset_ps.ravel()[valid] / tpb_ps_data
        if integer_bins:
            shift = np.round(shift)
        idx = np.flatnonzero(valid & (np.abs(shift) >= 1e-6))   # zero shift -> skip FFT

        # scipy.fft: ~2x numpy's speed at awkward lengths (535 = 5 x 107)
        from scipy import fft as _sfft
        freqs = np.fft.rfftfreq(nb)   # shape (nb//2 + 1,), same for every pixel
        for a in range(0, idx.size, chunk):
            sel = idx[a:a + chunk]
            spec = _sfft.rfft(flat_in[sel].astype(np.float64), axis=1)
            ang = (-2.0 * np.pi) * freqs[None, :] * shift[sel, None]
            ramp = np.empty(ang.shape, dtype=np.complex128)
            ramp.real = np.cos(ang)
            ramp.imag = np.sin(ang)
            spec *= ramp
            flat_out[sel] = _sfft.irfft(spec, n=nb, axis=1)

        print(f"  [pixel_offset] applied to {idx.size} pixels "
              f"({self.n_nan} skipped - no calibration data)")
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """One-line human-readable summary of the calibration map."""
        v = self.offset_ps[self.valid_mask]
        return (
            f"PixelOffsetCorrection | source: {Path(self.source).name} "
            f"| shape {self.offset_ps.shape} "
            f"| sigma={np.std(v):.1f} ps, range=[{v.min():.0f}, {v.max():.0f}] ps "
            f"| valid={self.valid_mask.sum()}/{self.offset_ps.size}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fft_shift(trace: np.ndarray, shift_bins: float,
               freqs: np.ndarray) -> np.ndarray:
    """Shift a 1-D real array by shift_bins (fractional OK) using an rfft
    phase ramp.  Circular shift — negligible at the array boundaries because
    the real decay has returned to baseline well before the recorded window
    ends.  Matches the implementation in irf.shift_signal_subbin, kept here
    to avoid importing irf (which carries scipy and many other heavy imports)
    just for this one primitive."""
    spectrum = np.fft.rfft(trace)
    phase    = np.exp(-1j * 2 * np.pi * freqs * shift_bins)
    return np.fft.irfft(spectrum * phase, n=len(trace))


def _fmt_setting(v, unit=""):
    return "unknown" if v is None else f"{v:g}{unit}" if isinstance(v, (int, float)) else f"{v}{unit}"


# ---------------------------------------------------------------------------
# Building a calibration map from an IRF measurement
# ---------------------------------------------------------------------------

def estimate_offsets_from_irf(cube, tpb, min_counts=30, iters=3, max_lag_bins=20):
    """Per-pixel IRF arrival time from a foil/scatter IRF cube (H, W, nb).

    Each pixel's background-subtracted IRF (window peak-30 .. peak+60 bins of
    the pooled IRF) is cross-correlated with the pooled IRF; the sub-bin lag
    comes from a parabola through the correlation peak. The template is then
    rebuilt from the pixels aligned on their estimates (sharper than the
    smeared raw sum) and the lags re-estimated, `iters` times in all.
    Estimator noise: ~6 ps per pixel at ~3500 IRF photons (measured as the
    difference between two independent foil IRFs).

    Returns (offset_ps, counts): offset_ps (H, W) in ps relative to the median
    pixel, positive = the pixel's IRF arrives late; NaN where the pixel has
    fewer than min_counts background-subtracted IRF photons. counts (H, W).
    """
    H, W, nb = cube.shape
    c = cube.reshape(H * W, nb).astype(np.float64)
    pooled = c.sum(axis=0)
    pk = int(np.argmax(pooled))
    bg_start = min(pk + 200, nb - 50)          # far tail: flat dark-count floor
    bg = c[:, bg_start:].mean(axis=1)
    lo, hi = max(pk - 30, 0), min(pk + 60, nb)
    w = c[:, lo:hi] - bg[:, None]              # not clipped (keeps noise symmetric)
    counts = w.sum(axis=1)
    ok = counts >= min_counts
    if ok.sum() < 10:
        raise ValueError(f"too few pixels with >= {min_counts} IRF photons to build a map")
    L = hi - lo
    n = 2 * L                                   # zero padding: no wrap-around in the lags
    R = int(max_lag_bins)
    wk = w[ok]
    Fw = np.fft.rfft(wk, n, axis=1)
    f = np.fft.rfftfreq(n)
    tmpl = wk.sum(axis=0)
    lag = np.zeros(wk.shape[0])
    rows = np.arange(wk.shape[0])
    for _ in range(iters):
        xc = np.fft.irfft(Fw * np.conj(np.fft.rfft(tmpl, n))[None, :], n, axis=1)
        xc = np.concatenate([xc[:, n - R:], xc[:, :R + 1]], axis=1)   # lags -R..R
        k = np.clip(np.argmax(xc, axis=1), 1, 2 * R - 1)
        y0, y1, y2 = xc[rows, k - 1], xc[rows, k], xc[rows, k + 1]
        den = y0 - 2 * y1 + y2
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(den != 0, 0.5 * (y0 - y2) / den, 0.0)
        lag = (k - R) + np.clip(frac, -0.5, 0.5)
        # sharper template: every pixel shifted back by its own lag, re-summed
        back = np.exp(2j * np.pi * f[None, :] * (lag - np.median(lag))[:, None])
        tmpl = np.fft.irfft(Fw * back, n, axis=1)[:, :L].sum(axis=0)
    off = np.full(H * W, np.nan)
    off[ok] = (lag - np.median(lag)) * tpb * 1e12
    return off.reshape(H, W), counts.reshape(H, W)


def build_calibration(irf_path, out_path, rep_rate_mhz=None, sync_delay=None,
                      min_counts=30, notes=""):
    """Build a calibration map from an IRF file and save it as out_path (.npy)
    plus the sidecar out_path stem + '_meta.json' (with rep_rate_mhz and
    sync_delay, which the map depends on). Returns the PixelOffsetCorrection."""
    from .io import load_flim_data
    s = load_flim_data(str(irf_path))
    tpb = s["tpb"]
    off, counts = estimate_offsets_from_irf(s["cube"], tpb, min_counts=min_counts)
    v = off[np.isfinite(off)]
    out_path = Path(out_path)
    if out_path.suffix != ".npy":
        out_path = out_path.with_suffix(".npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, off)
    import datetime
    meta = {
        "description": ("Per-pixel SPAD timing offset (ps) relative to the sensor-wide median, "
                        "from cross-correlation of each pixel's IRF with the re-aligned pooled "
                        "IRF (sub-bin; flim_core.pixel_offset.estimate_offsets_from_irf). "
                        "Orientation as returned by flim_core.io.load_flim_data. "
                        f"NaN = fewer than {min_counts} IRF photons."),
        "source_file": Path(irf_path).name,
        "rep_rate_mhz": rep_rate_mhz,
        "sync_delay": sync_delay,
        "tpb_ps": round(tpb * 1e12, 3),
        "sensor_shape_hw": list(off.shape),
        "offset_std_ps": float(v.std()),
        "median_irf_counts_per_pixel": float(np.median(counts[np.isfinite(off)])),
        "valid_pixels": int(v.size),
        "total_pixels": int(off.size),
        "units": "picoseconds",
        "created": datetime.date.today().isoformat(),
    }
    if notes:
        meta["notes"] = notes
    with open(out_path.with_name(out_path.stem + "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved {out_path} (+ _meta.json): offset std {v.std():.1f} ps, "
          f"{v.size}/{off.size} pixels, median {meta['median_irf_counts_per_pixel']:.0f} "
          f"IRF photons/pixel")
    if meta["median_irf_counts_per_pixel"] < 1000:
        print("  NOTE: under ~1000 IRF photons per pixel the map gets noisy; a longer or "
              "brighter IRF acquisition gives a better map.")
    return PixelOffsetCorrection(off, tpb * 1e12, source=str(out_path), meta=meta)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Build a per-pixel SPAD timing-offset calibration map from a foil/scatter "
                    "IRF file. Use an IRF taken with the SAME laser repetition rate and sync "
                    "delay as the data -- the map depends on both.")
    p.add_argument("irf", help="IRF file (.ds3d / .h5)")
    p.add_argument("--out", required=True, help="output .npy path (a _meta.json is written next to it)")
    p.add_argument("--rep-rate-mhz", type=float, default=None, help="laser repetition rate, MHz")
    p.add_argument("--sync-delay", default=None, help="sync delay setting (as on the camera)")
    p.add_argument("--min-counts", type=int, default=30, help="min IRF photons for a pixel")
    p.add_argument("--notes", default="", help="free text stored in the metadata")
    a = p.parse_args(argv)
    sd = a.sync_delay
    try:
        sd = float(sd) if sd is not None else None
        sd = int(sd) if sd is not None and sd.is_integer() else sd
    except ValueError:
        pass
    build_calibration(a.irf, a.out, rep_rate_mhz=a.rep_rate_mhz, sync_delay=sd,
                      min_counts=a.min_counts, notes=a.notes)


if __name__ == "__main__":
    main()
