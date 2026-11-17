"""
Hot-pixel detection from the decay histograms themselves.

Some SPAD pixels have a dark-count rate several times the typical one
("hot" pixels). Their extra counts are flat in time, so in a phasor they
pull the pixel toward the phasor of a flat background, and in a fit they
raise the fitted background / lifetime. They are a fixed property of the
sensor (2026-11-17: 90-98 % of the hot pixels of a fluorescein file were
also hot in foil/DASPI IRFs and a cell measurement); the camera already
switches some off (0 photons), the rest still deliver data.

Detection uses the PRE-RISE bins -- the stretch of the window before the
decay starts, where a pixel records only background plus the tiny remnant
of the previous laser pulse. Unlike the tail, these bins do not depend on
the pixel's lifetime, so a per-pixel background read there is unbiased.
Per pixel (mean pre-rise count m_i, total photons I_i):

    expected m_i = b0 + k * I_i      (b0: normal dark level; k * I_i: the
                                      previous pulse's remnant, which scales
                                      with brightness)

b0 and k are fitted robustly over the live pixels (hot ones excluded,
iterated). A pixel is hot when its pre-rise counts are above that
expectation at n_sigma (one-sided Poisson tail probability of the normal
distribution's n_sigma). Its excess (counts per bin, not clipped) is
m_i - expected_i.

Requirements: the window must start before the rise with at least
`min_bins` clean bins; the pooled pre-rise stretch is checked for flatness
(a warning is printed when its two halves differ by > 15 %).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, poisson


def prerise_slice(summed, min_bins=10, guard_bins=3, rise_frac=0.01):
    """Bins before the decay rise, from the pooled histogram `summed`.

    Baseline = median of the first min_bins bins; the rise starts at the
    first bin before the peak whose pooled level exceeds the baseline by
    rise_frac of (peak - baseline). The slice ends guard_bins before that.
    Returns slice(0, end), or None when fewer than min_bins clean bins exist.
    """
    s = np.asarray(summed, dtype=np.float64)
    pk = int(np.argmax(s))
    if pk < min_bins + guard_bins:
        return None
    base = float(np.median(s[:min_bins]))
    above = np.nonzero(s[:pk + 1] - base > rise_frac * (s[pk] - base))[0]
    rise = int(above[0]) if above.size else pk
    end = rise - guard_bins
    return slice(0, end) if end >= min_bins else None


def detect_hot_pixels(cube, n_sigma=3.0, live=None, min_bins=10, iters=4):
    """Find hot pixels in a decay cube (H, W, nb) at native time bins.

    live: bool map of pixels to consider (default: any photons). Returns a
    dict, or None when the window has no usable pre-rise stretch:
      hot        (H, W) bool
      excess     (H, W) float, extra background in counts per bin (0 where
                 not hot)
      level      (H, W) float, measured pre-rise mean counts per bin
      b0, k      fitted normal model (counts/bin, per total photon)
      pre        the pre-rise slice used
      flat_ratio second-half / first-half pooled pre-rise level
    """
    H, W, nb = cube.shape
    n = cube.sum(axis=2).astype(np.float64)
    live = (n > 0) if live is None else (live & (n > 0))
    if live.sum() < 50:
        return None
    pre = prerise_slice(cube[live].sum(axis=0), min_bins=min_bins)
    if pre is None:
        return None
    npre = pre.stop - pre.start
    cnt = cube[:, :, pre].sum(axis=2).astype(np.float64)
    m = cnt / npre
    ok = live.copy()
    b0, k = float(np.median(m[live])), 0.0
    p_cut = norm.sf(n_sigma)
    hot = np.zeros((H, W), bool)
    for _ in range(iters):
        x, y = n[ok], m[ok]
        A = np.column_stack([np.ones_like(x), x])
        (b0, k), *_ = np.linalg.lstsq(A, y, rcond=None)
        expected = np.maximum(b0 + k * n, 1e-12)
        hot = live & (poisson.sf(cnt - 1, expected * npre) < p_cut)
        ok = live & ~hot
    expected = b0 + k * n
    excess = np.where(hot, m - expected, 0.0)
    half = npre // 2
    pooled = cube[live & ~hot][:, pre].sum(axis=0).astype(np.float64)
    flat_ratio = (float(pooled[half:].mean() / pooled[:half].mean())
                  if half >= 3 and pooled[:half].mean() > 0 else np.nan)
    return dict(hot=hot, excess=excess, level=m, b0=float(b0), k=float(k),
                pre=pre, flat_ratio=flat_ratio, n_sigma=n_sigma)


def report(res, tpb_ns, n_live):
    """Print a short summary of detect_hot_pixels' result (ASCII)."""
    pre = res["pre"]
    hot = res["hot"]
    print(f"  Hot pixels: pre-rise bins {pre.start}-{pre.stop - 1} "
          f"({(pre.stop - pre.start) * tpb_ns:.2f} ns); normal level {res['b0']:.3f} counts/bin "
          f"(+ previous-pulse remnant {res['k'] * 1e4:.3f} per 10^4 photons)")
    if hot.any():
        ex = res["excess"][hot]
        print(f"    {int(hot.sum())} hot pixel(s) at {res['n_sigma']:g} sigma "
              f"({100 * hot.sum() / max(n_live, 1):.1f}% of live pixels); extra background "
              f"median {np.median(ex):.2f}, max {ex.max():.1f} counts/bin")
    else:
        print("    no hot pixels found")
    fr = res["flat_ratio"]
    if np.isfinite(fr) and abs(fr - 1) > 0.15:
        print(f"    WARNING: the pre-rise stretch is not flat (second half / first half = {fr:.2f}) "
              f"-- the window may start too close to the rise or contain an artefact; "
              f"hot-pixel levels may be off.")
