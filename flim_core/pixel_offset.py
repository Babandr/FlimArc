"""
Per-pixel SPAD timing offset correction.

SPAD arrays have a fixed, pixel-specific timing offset caused by within-cluster
TDC skew that the manufacturer's factory calibration does not fully remove.  On
the Flimera camera used here the residual spread is ~103 ps rms, with a total
range of ~66 bins (37.4 ps/bin), and is reproducible across objectives,
samples, and acquisition sessions (Pearson r ≈ 0.96 between independent IRF
measurements).

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
                  f"| noise σ ≈ {meta.get('noise_sigma_ps','?')} ps")

        return cls(offset_ps, tpb_ps, source=str(path), meta=meta)

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    def apply(self, cube: np.ndarray, tpb: float) -> np.ndarray:
        """Shift each pixel's decay histogram to remove its timing offset.

        Each pixel (r, c) is shifted by  -offset_ps[r, c] / tpb_data  bins
        using a sub-bin FFT phase ramp, so that all peaks align to the
        sensor-wide median.  Pixels where the calibration is NaN are returned
        unchanged.

        Parameters
        ----------
        cube : ndarray, shape (H, W, nb), float
            Decay histograms in the orientation produced by load_flim_data
            (after the 90° CCW rotation).
        tpb : float
            Bin width of the data in seconds.

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

        # Pre-compute the FFT frequency ramp (same for every pixel)
        freqs = np.fft.rfftfreq(nb)   # shape (nb//2 + 1,)

        n_corrected = 0
        for r in range(H):
            for c in range(W):
                if not self.valid_mask[r, c]:
                    continue
                # Shift in bins: negative because we subtract the pixel's
                # own offset to align it to the global median.
                shift_bins = -self.offset_ps[r, c] / tpb_ps_data
                if abs(shift_bins) < 1e-6:
                    continue   # effectively zero shift — skip FFT
                trace = cube[r, c].astype(np.float64)
                out[r, c] = _fft_shift(trace, shift_bins, freqs)
                n_corrected += 1

        print(f"  [pixel_offset] applied to {n_corrected} pixels "
              f"({self.n_nan} skipped — no calibration data)")
        return out.astype(cube.dtype)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """One-line human-readable summary of the calibration map."""
        v = self.offset_ps[self.valid_mask]
        return (
            f"PixelOffsetCorrection | source: {Path(self.source).name} "
            f"| shape {self.offset_ps.shape} "
            f"| σ={np.std(v):.1f} ps, range=[{v.min():.0f}, {v.max():.0f}] ps "
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
