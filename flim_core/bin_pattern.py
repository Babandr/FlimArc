"""
8-bin TDC clock pattern correction.

SPAD cameras with on-chip TDCs produce a periodic sensitivity variation with
period exactly 8 bins.  The root cause is that the TDC clock cycle is not a
perfect integer multiple of the nominal bin width (37.4 ps × 8 = 299.2 ps,
which is not a round number), so the TDC cycles through 8 slightly unequal
effective bin widths before repeating.  In measured data this appears as a
sawtooth oscillation on the decay tail with amplitude ~10–15% per bin.

Unlike the per-pixel timing offset (flim_core.pixel_offset), the 8-bin pattern
has a FIXED SHAPE but a FLOATING PHASE that can differ between acquisition
sessions because it depends on where the laser trigger falls within the TDC's
8-bin clock cycle at the moment recording starts.  Across four independent
measurements the aligned factor profiles correlate at r > 0.996, confirming the
shape is a stable camera property; but the phase shifts freely so it cannot be
pre-calibrated.

This means the correction must be SELF-CALIBRATING: the 8 sensitivity factors
are measured from the sum histogram of the data (or IRF) file being corrected,
then divided out of every pixel's histogram.  The sum histogram contains
millions of counts even in a 1-second acquisition, giving a very clean
phase-factor estimate.

Detrending strategy
-------------------
To isolate the 8-bin oscillation we divide the sum histogram by a smooth trend
that approximates the true underlying decay shape.  The trend MUST NOT be
computed from a functional model (mono-exponential, etc.) because for biexponential
decays -- especially ones with a dominant short component -- a wrong model
introduces errors that contaminate the phase factors and then corrupt every
pixel's corrected histogram.

Instead we use a running average with a window size that is an EXACT MULTIPLE
of 8 bins.  A window of N = 8k bins averages over k complete cycles of the
8-bin pattern, so the pattern cancels identically in the trend estimate
regardless of the underlying decay shape.  No assumptions about the number of
lifetime components, their values, or the IRF shape are required.

Error analysis (Monte Carlo, seven decay scenarios):
    mono-exp detrend:        max factor error up to 14% (fast decays)
    running-avg ×8 detrend:  max factor error ≤ 0.7% (all decay types)

Usage
-----
    from flim_core.bin_pattern import BinPatternCorrection

    # Measure factors from the data itself (no separate calibration needed):
    corr = BinPatternCorrection.from_cube(cube, B0)
    cube_corrected = corr.apply(cube, B0)

    # Or from a dedicated IRF file when working on very dim data:
    corr = BinPatternCorrection.from_cube(irf_cube, irf_B0)
    cube_corrected = corr.apply(cube, B0)

Ordering relative to other corrections
---------------------------------------
Apply BEFORE the pixel-offset correction and BEFORE any spatial or time
binning.  This matters because:

  - Pixel-offset correction shifts histograms via FFT, which would smear the
    sharp 8-bin boundary between adjacent bins if the pattern has not been
    removed first.
  - Spatial binning (3×3) pools pixels whose 8-bin phases differ by up to 7
    bins.  Correcting beforehand means all 9 pixels share the same flat
    baseline before they are summed, rather than relying on the approximate
    phase averaging that uncorrected binning provides.

The current pipeline order therefore becomes:
    load → bin-pattern correction → pixel-offset correction
         → ROI crop → spatial binning → time binning → fitting
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BinPatternCorrection:
    """Measured 8-bin sensitivity factors + apply() method.

    Attributes
    ----------
    factors : ndarray, shape (8,), float64
        Sensitivity factor for each absolute bin phase (0..7).  Normalised
        so their mean is 1.0.  Dividing a decay histogram by
        factors[(B0 + bin_index) % 8] removes the pattern.
    amplitude : float
        Peak-to-trough amplitude of the pattern as a fraction (e.g. 0.12
        for a 12% oscillation).
    n_photons_used : int
        Total counts in the sum histogram used to estimate the factors.
        Fewer counts → noisier estimates; the correction is reliable when
        this exceeds ~500 000.
    """

    # Minimum number of bins in the tail window used for estimation.
    # Fewer bins means fewer samples per phase → noisier factors.
    _MIN_TAIL_BINS = 32

    def __init__(self, factors: np.ndarray, n_photons_used: int = 0):
        factors = np.asarray(factors, dtype=np.float64)
        if factors.shape != (8,):
            raise ValueError(f"factors must have shape (8,), got {factors.shape}")
        self.factors        = factors / factors.mean()   # always unit-mean
        self.amplitude      = float(self.factors.max() - self.factors.min())
        self.n_photons_used = int(n_photons_used)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_cube(cls, cube: np.ndarray, B0: int,
                  trend_window_bins: int = 32,
                  tail_skip_bins: int = 8) -> "BinPatternCorrection":
        """Measure the 8-bin pattern from a decay cube's sum histogram.

        The sum histogram is detrended using a running average whose window
        is rounded to the nearest multiple of 8 bins, so the pattern cancels
        exactly in the trend regardless of the underlying decay shape.  This
        is model-free and works correctly for mono-exponential, biexponential,
        and any other decay type.

        Parameters
        ----------
        cube : ndarray, shape (H, W, nb)
            Decay histogram cube.  Summed over all pixels to form the
            high-SNR integrated histogram used for factor estimation.
        B0 : int
            Absolute bin offset of the cube within the full sensor timing
            window (the ``B0`` field returned by ``load_flim_data``).  Used
            to map local bin indices to absolute bin indices for phase
            assignment.
        trend_window_bins : int, optional
            Running-average window width in bins.  Rounded UP to the nearest
            multiple of 8 so that exactly k complete 8-bin cycles are
            averaged at every position.  Default 32 (= 4 cycles).  Wider
            windows give a smoother trend at the cost of edge effects; the
            default is a good balance for typical FLIM tail lengths.
        tail_skip_bins : int, optional
            Number of bins after the histogram peak to skip before starting
            the tail region used for estimation.  Skipping the IRF peak
            region avoids contaminating the pattern estimate with the sharp
            rising edge.  Default 8 (= one full pattern cycle).
        """
        # Round window UP to nearest multiple of 8 so pattern cancels exactly
        k = max(1, int(np.ceil(trend_window_bins / 8)))
        window = k * 8

        sumh   = cube.sum(axis=(0, 1)).astype(np.float64)
        nb     = len(sumh)
        pk     = int(np.argmax(sumh))

        tail_start = pk + tail_skip_bins
        tail_end   = nb   # use everything to the end of the window
        tail_len   = tail_end - tail_start

        if tail_len < cls._MIN_TAIL_BINS:
            raise ValueError(
                f"Tail region too short ({tail_len} bins) to estimate 8-bin "
                f"pattern factors reliably (need ≥ {cls._MIN_TAIL_BINS}).  "
                f"Check that the histogram window extends well past the IRF peak."
            )

        seg = sumh[tail_start:tail_end]

        # Running average with window = k×8: pattern cancels in the trend
        trend = uniform_filter1d(seg, size=window, mode="nearest")

        # Ratio: raw / trend ≈ pattern × (decay_shape / smooth_decay_shape)
        # The second factor is ~1.0 on the 8-bin scale because the decay
        # shape is smooth relative to the pattern period.
        ratio = seg / np.maximum(trend, 1.0)

        # Map each bin to its absolute phase and average within each group
        abs_bins = B0 + tail_start + np.arange(tail_len)
        factors  = np.array([ratio[abs_bins % 8 == p].mean() for p in range(8)])

        n_photons = int(sumh.sum())
        corr = cls(factors, n_photons_used=n_photons)

        print(f"  [bin_pattern] measured 8-bin factors from sum histogram "
              f"({n_photons:,} counts, tail {tail_len} bins, window {window})")
        print(f"  [bin_pattern] amplitude = {corr.amplitude*100:.1f}%  "
              f"factors: {' '.join(f'{v:.3f}' for v in corr.factors)}")

        if n_photons < 500_000:
            warnings.warn(
                f"[bin_pattern] Only {n_photons:,} counts in the sum histogram. "
                f"Factor estimates may be noisy — consider using a longer exposure "
                f"or a brighter IRF sample to calibrate the pattern.",
                stacklevel=2,
            )

        return corr

    # ------------------------------------------------------------------
    # Core operation
    # ------------------------------------------------------------------

    def apply(self, cube: np.ndarray, B0: int) -> np.ndarray:
        """Divide each bin of every pixel's histogram by its sensitivity factor.

        The correction is a pointwise division: for each temporal bin i,
        every pixel's count at that bin is divided by
        ``self.factors[(B0 + i) % 8]``.  This flattens the 8-bin
        oscillation so the corrected histogram reflects the true underlying
        decay shape.

        Because the factors were measured from the same (or equivalent)
        B0-aligned cube, the absolute phase is consistent between
        measurement and correction.

        Parameters
        ----------
        cube : ndarray, shape (H, W, nb)
            Decay histogram cube to correct.  A copy is returned; the
            input is never modified.
        B0 : int
            Absolute bin offset matching the B0 used in ``from_cube``.

        Returns
        -------
        ndarray, same shape and dtype as cube.
        """
        H, W, nb = cube.shape

        # Build a (nb,) divisor array: factor[(B0+i)%8] for i=0..nb-1
        phases  = (B0 + np.arange(nb)) % 8
        divisor = self.factors[phases]   # (nb,)

        # Broadcast over (H, W): divide each bin by its factor
        out = cube.astype(np.float64) / divisor[np.newaxis, np.newaxis, :]
        print(f"  [bin_pattern] applied to cube shape {cube.shape}")
        return out.astype(cube.dtype)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        return (
            f"BinPatternCorrection | amplitude={self.amplitude*100:.1f}% "
            f"| factors={[round(v,4) for v in self.factors]} "
            f"| n_photons={self.n_photons_used:,}"
        )
