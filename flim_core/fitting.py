"""
Tail-fit lifetime fitting (Poisson maximum likelihood): global (whole-image)
fit and per-pixel fit, mono- or bi-exponential. The IRF/reconvolution fit
family lives in flim_core/irf.py instead, since it needs its own model
builders and diagnostic plot.

The three optimizer bound pairs (tau_short_bounds, tau_long_bounds,
tau_mono_bounds) are real parameters here, not module constants -- every
public function takes them with a default of None, which means "use the
config.py default". This is what lets a caller (a plugin's CLI, or the
GUI's lifetime-bounds preset) override them per run instead of having to
edit this file, e.g. to switch between dyes with very different lifetimes.

Fit-quality outputs: every per-pixel fitting function (fit_pixels*) returns
a dict of same-shape [H,W] maps -- not just the lifetime(s) -- so a
photon-starved pixel's bad fit is visible rather than silently indistinguishable
from a good one:
  tau[_short/_long]      the fitted lifetime(s), NaN where not fitted
  tau[_short/_long]_std  1-sigma uncertainty on that lifetime, from the local
                         curvature of the negative log-likelihood at the fit
                         (see _param_std) -- large where the fit is poorly
                         constrained (typically: too few photons)
  n_photons              photons actually used in that pixel's fit
  converged               scipy.optimize.minimize's own success flag
  nll                     the negative log-likelihood at the optimum (lower
                         = better description of the data, for pixels with
                         similar photon counts -- not comparable across very
                         different photon counts)
The whole-image pooled fits (global_fit and friends) gain the equivalent
single-number tau_std/tau_short_std/tau_long_std in their returned dict.
"""

import warnings

import numpy as np
from scipy.optimize import minimize, curve_fit

from .config import TAU_SHORT_BOUNDS, TAU_LONG_BOUNDS, TAU_MONO_BOUNDS, TAIL_START_OFFSET


def _resolve_bounds(tau_short_bounds, tau_long_bounds, tau_mono_bounds):
    """Fill in any bound left as None with the config.py default. Centralized
    here so every fitting function applies the same fallback rule."""
    ts = tau_short_bounds if tau_short_bounds is not None else TAU_SHORT_BOUNDS
    tl = tau_long_bounds if tau_long_bounds is not None else TAU_LONG_BOUNDS
    tm = tau_mono_bounds if tau_mono_bounds is not None else TAU_MONO_BOUNDS
    return ts, tl, tm


def _param_std(nll_func, p_hat, idx, args=()):
    """1-sigma uncertainty on parameter p_hat[idx], from the local curvature
    of nll_func at the fit -- the standard asymptotic maximum-likelihood
    variance estimate (inverse Fisher information), evaluated with a cheap
    central finite difference rather than a full Hessian:

        sigma = 1 / sqrt(d^2(NLL)/dp^2)

    All other parameters are held fixed at their fitted values (a profile-
    likelihood simplification -- ignores cross-parameter correlations, but
    is the right order of magnitude and costs only 2 extra NLL evaluations
    per parameter, which matters when this runs once per pixel). Returns
    NaN if the curvature isn't usable (non-positive, or the fit landed
    exactly on a bound where the local curvature isn't meaningful)."""
    p_hat = np.asarray(p_hat, dtype=float)
    val0 = p_hat[idx]
    h = max(abs(val0) * 0.01, 1e-3)

    def f(v):
        p = p_hat.copy()
        p[idx] = v
        return nll_func(p, *args)

    try:
        f0, fp, fm = f(val0), f(val0 + h), f(val0 - h)
    except Exception:
        return np.nan
    curvature = (fp - 2 * f0 + fm) / h**2
    if not np.isfinite(curvature) or curvature <= 0:
        return np.nan
    return 1.0 / np.sqrt(curvature)


def _curve_fit_std(pcov, idx):
    """1-sigma uncertainty on parameter `idx` from curve_fit's covariance
    matrix -- the free, standard result curve_fit already computes but the
    original code discarded. Returns NaN if the fit was ill-conditioned
    (pcov containing inf is curve_fit's own signal of that)."""
    if pcov is None or not np.all(np.isfinite(pcov)):
        return np.nan
    var = pcov[idx, idx]
    return float(np.sqrt(var)) if var > 0 else np.nan


def _poisson_fit(model, yv, p0, bounds, ftol=1e-12, gtol=1e-8, maxiter=2000, return_result=False):
    """Poisson maximum-likelihood fit of model(params) to the counts yv with
    L-BFGS-B, written to stay accurate at HIGH photon counts.

    Why not minimize sum(m - y*log(m)) directly: that objective carries a huge
    constant offset (about 4e7 at a few million photons) and the parameters
    differ in size by orders of magnitude (A ~ 1e5, tau ~ 6, C ~ 40).
    L-BFGS-B's relative stopping rule and its finite-difference gradient then
    declare convergence well short of the optimum -- tau came out 3-7 % too
    long on exact synthetic data (6.17 ns for a 6 ns decay). Two changes
    remove that, without changing the optimum:

      * the objective is the Poisson deviance  sum(m - y + y*log(y/m)),
        i.e. the same likelihood minus a constant, so its value is ~O(nbins)
        instead of ~O(photons);
      * every parameter is divided by its starting value, so the optimizer
        works on O(1) numbers (bounds are rescaled the same way).

    Returns (params in the ORIGINAL units, objective(params in original
    units)). The objective has the same curvature as the negative
    log-likelihood, so it can be passed straight to _param_std /
    _param_std_full. With return_result=True the scipy OptimizeResult is
    returned as a third element (for success flags)."""
    yv = np.asarray(yv, dtype=float)
    pos = yv > 0
    y_safe = np.where(pos, yv, 1.0)
    p0 = np.asarray(p0, dtype=float)
    scale = np.where(np.abs(p0) > 1e-6, np.abs(p0), 1.0)

    def dev(p):
        m = np.maximum(model(p), 1e-9)
        return np.sum(m - yv + np.where(pos, yv * np.log(y_safe / m), 0.0))

    z_bounds = [(lo / s, hi / s) for (lo, hi), s in zip(bounds, scale)]
    r = minimize(lambda z: dev(z * scale), p0 / scale, method="L-BFGS-B",
                 bounds=z_bounds, options=dict(ftol=ftol, gtol=gtol, maxiter=maxiter))
    if return_result:
        return r.x * scale, dev, r
    return r.x * scale, dev


def _poisson_refine(model, tn, yv, p_seed, bounds):
    """Poisson ML refinement of a tail-fit model(t, *params), starting from
    p_seed (e.g. a curve_fit result). Returns (popt, objective) or None if
    the refinement failed (exception, non-finite result), so callers can fall
    back to the seed instead of aborting a long run. Never returns a point
    with a worse Poisson deviance than the seed."""
    try:
        seed = np.asarray(p_seed, dtype=float)
        x, dev = _poisson_fit(lambda p: model(tn, *p), yv, seed, bounds)
        d_new, d_seed = dev(x), dev(seed)
        if not (np.all(np.isfinite(x)) and np.isfinite(d_new)):
            return None
        if np.isfinite(d_seed) and d_new > d_seed:
            x = seed
        return x, dev
    except Exception:
        return None


def _param_std_full(nll_func, p_hat, idx, args=(), free=None, bounds=None):
    """1-sigma uncertainty on parameter p_hat[idx] from the FULL numerical
    Hessian of nll_func at the fit (central differences, all parameters
    free): sigma = sqrt([H^-1]_idx,idx). Unlike _param_std, which holds the
    other parameters fixed, this accounts for their correlations -- for a
    biexponential the two lifetimes and amplitudes are strongly correlated,
    and the fixed-parameter shortcut underestimates the error about 3-4x
    (tested against the scatter of repeated fits). Costs ~n^2 evaluations,
    fine for a one-off pooled fit; too slow to run per pixel. Returns NaN if
    the Hessian isn't usable.

    free: optional list of the parameter indices that were actually fitted.
    Parameters held fixed (e.g. the background C when fit_background=False)
    must be left out, otherwise their curvature would inflate the error of
    the free ones as if they had been fitted too.

    bounds: optional list of (low, high) per parameter. Parameters sitting ON
    a bound are treated as fixed (the usual constrained-optimum convention).
    If the requested parameter itself sits on a bound, its curvature isn't
    meaningful and the fixed-parameter estimate (_param_std) is returned
    instead."""
    p = np.asarray(p_hat, dtype=float)
    free = list(range(len(p))) if free is None else list(free)
    if bounds is not None:
        on_bound = [i for i, (lo, hi) in enumerate(bounds)
                    if np.isclose(p[i], lo, rtol=1e-6, atol=1e-9) or np.isclose(p[i], hi, rtol=1e-6, atol=1e-9)]
        if idx in on_bound:
            return _param_std(nll_func, p_hat, idx, args=args)
        free = [i for i in free if i not in on_bound]
    n = len(free)
    h = np.maximum(np.abs(p) * 0.01, 1e-4)
    H = np.zeros((n, n))

    def f(i, a, j, b):
        q = p.copy()
        q[i] += a * h[i]
        q[j] += b * h[j]
        return nll_func(q, *args)

    try:
        f0 = nll_func(p, *args)
        for ii, i in enumerate(free):
            H[ii, ii] = (f(i, 1, i, 0) - 2 * f0 + f(i, -1, i, 0)) / h[i] ** 2
            for jj in range(ii + 1, n):
                j = free[jj]
                H[ii, jj] = H[jj, ii] = (f(i, 1, j, 1) - f(i, 1, j, -1)
                                         - f(i, -1, j, 1) + f(i, -1, j, -1)) / (4 * h[i] * h[j])
        var = _inv_diag_entry(H, free.index(idx))
    except Exception:
        return np.nan
    return float(np.sqrt(var)) if np.isfinite(var) and var > 0 else np.nan


def _curve_fit_tail(model, tn, yv, p0, lo, hi, fit_background=True):
    """curve_fit of a tail model whose LAST parameter is the background C,
    weighted by sqrt(max(counts, 1)) like the original fits. With
    fit_background=False, C is held at 0: the fit runs on the model without
    C, and popt/pcov are padded back to full size (C = 0, zero covariance)
    so callers see the same parameter layout either way."""
    sigma = np.sqrt(np.maximum(yv, 1))
    if fit_background:
        return curve_fit(model, tn, yv, p0=p0, sigma=sigma, bounds=(lo, hi), maxfev=20000)
    popt, pcov = curve_fit(lambda t, *q: model(t, *q, 0.0), tn, yv, p0=list(p0)[:-1], sigma=sigma,
                           bounds=(list(lo)[:-1], list(hi)[:-1]), maxfev=20000)
    n = len(popt) + 1
    pcov_full = np.zeros((n, n))
    pcov_full[:-1, :-1] = pcov
    return np.append(popt, 0.0), pcov_full


def _inv_diag_entry(M, k):
    """[M^-1]_kk for a symmetric information/Hessian matrix M, or NaN if
    parameter k itself carries no information.

    Parameters with (numerically) zero curvature are dropped first. That
    happens in practice: when a biexponential pixel collapses to one
    component (that component's amplitude pinned at 0), its lifetime has no
    effect on the model at all, M is exactly singular along that direction,
    and the plain inverse turns the std of EVERY parameter into NaN -- about
    a third of dim biexponential pixels. Dropping the dead parameter gives
    the remaining ones their proper (single-component) uncertainty, while
    the dead lifetime itself is reported as NaN (unidentifiable), not as a
    confident-looking number.

    The inverse is computed on the diagonally rescaled matrix
    D^-1/2 M D^-1/2, so parameters whose curvature differs by many orders of
    magnitude do not spoil its accuracy."""
    d = np.diag(M)
    alive = d > 1e-14 * np.max(d)
    if not alive[k]:
        return np.nan
    M = M[np.ix_(alive, alive)]
    k = int(np.sum(alive[:k]))
    s = 1.0 / np.sqrt(np.diag(M))
    return np.linalg.inv(M * s[:, None] * s[None, :])[k, k] * s[k] ** 2


def _param_std_fisher(model, p_hat, idx, free=None, bounds=None, fallback=None):
    """1-sigma uncertainty on parameter p_hat[idx] from the Poisson FISHER
    INFORMATION, F_ij = sum_k (dm_k/dp_i)(dm_k/dp_j) / m_k, with the model
    derivatives taken by central differences; sigma = sqrt([F^-1]_idx,idx).
    This is the expected value of the Hessian that _param_std_full computes
    by second differences, but needs only ~2n model evaluations instead of
    ~2n^2 -- the difference between +5 % and +40 % run time per pixel for a
    reconvolution fit -- and it accounts for the correlations between
    parameters that the fixed-parameter shortcut (_param_std) ignores, which
    made biexponential pixel uncertainties ~4x too small.

    model(p) -> expected counts per bin. free / bounds: as in
    _param_std_full (parameters on a bound are treated as fixed; if the
    requested parameter itself is on a bound, fallback() is returned if
    given, else NaN)."""
    p = np.asarray(p_hat, dtype=float)
    free = list(range(len(p))) if free is None else list(free)
    if bounds is not None:
        on_bound = [i for i, (lo, hi) in enumerate(bounds)
                    if np.isclose(p[i], lo, rtol=1e-6, atol=1e-9) or np.isclose(p[i], hi, rtol=1e-6, atol=1e-9)]
        if idx in on_bound:
            return fallback() if fallback is not None else np.nan
        free = [i for i in free if i not in on_bound]
    try:
        m = np.maximum(model(p), 1e-9)
        J = np.empty((len(free), m.size))
        for a, i in enumerate(free):
            h = max(abs(p[i]) * 1e-3, 1e-4)
            q = p.copy(); q[i] = p[i] + h; up = model(q)
            q[i] = p[i] - h; dn = model(q)
            J[a] = (up - dn) / (2 * h)
        var = _inv_diag_entry((J / m) @ J.T, free.index(idx))
    except Exception:
        return np.nan
    return float(np.sqrt(var)) if np.isfinite(var) and var > 0 else np.nan


def _tail_axis(s, cube, offset=None):
    """Determine the peak, tail-start bin and the ns time axis for the tail.
    offset: bins after the peak where the tail fit starts (default:
    TAIL_START_OFFSET). Exposed as a parameter so callers -- including the
    GUI -- can override it per run instead of only via the module constant."""
    if offset is None:
        offset = TAIL_START_OFFSET
    summed = cube.sum(axis=(0, 1))                       # sum decay over all pixels -> one global histogram
    peak_bin = s["B0"] + int(np.argmax(summed))          # bin (absolute index) with the most counts = decay peak
    start_bin = peak_bin + offset                         # skip a few bins past the peak to avoid IRF/rise distortion
    s_idx = start_bin - s["B0"]                           # convert to an index relative to the cube's first bin
    tn = (np.arange(start_bin, s["B1"] + 1) - start_bin) * s["tpb"] * 1e9  # tail time axis in ns, t=0 at start_bin
    return summed, peak_bin, s_idx, tn


def global_fit(summed, s, s_idx, tn, mono=False,
              tau_short_bounds=None, tau_long_bounds=None, tau_mono_bounds=None,
              estimator="poisson", fit_background=True):
    """Robust tail fit of the summed decay (all pixels). mono=False (default)
    fits a biexponential; mono=True fits a single exponential -- appropriate
    when a mono-exponential model is a better description of the sample (a
    single well-behaved fluorophore), or as a quick reference/sanity check
    against the biexponential result.

    tau_short_bounds/tau_long_bounds/tau_mono_bounds: (min, max) in ns,
    the optimizer's search range for each lifetime -- pass None (default)
    to use the config.py default, or override per run/per dye.

    estimator: "poisson" (default) fits the Poisson likelihood, the same
    statistical model the per-pixel fits use. "leastsq" is the previous
    behaviour: curve_fit weighted by sqrt(max(counts, 1)). That weighting
    is estimated from the noisy data itself, which biases the lifetimes
    LOW when the pooled curve has few photons (e.g. 6 ns read 5.7 ns at
    20k photons, and a 2 + 6 ns biexponential came out 1.6 / 5.2 ns); the
    bias vanishes at millions of photons. For the biexponential the
    curve_fit result is kept as the SEED (robust at finding the right
    neighbourhood) and then refined with the Poisson fit; if that fails,
    the curve_fit result is returned with a warning.

    fit_background: True (default) fits a constant background C together with
    the decay. False holds C = 0 -- only correct for data that really contain
    no background (e.g. a simulation, or time-gated data); on data WITH
    background it biases the lifetimes (a missing constant is absorbed into a
    too-long tau). Do not pre-subtract the background and then fit with
    False: that leaves negative or non-integer counts, which the Poisson
    likelihood cannot handle (the free C is the statistically sound way).

    The returned dict includes tau_std (mono) or tau_short_std/tau_long_std
    (bi): the full-Hessian estimate for "poisson", curve_fit's own
    covariance matrix for "leastsq"."""
    if estimator not in ("poisson", "leastsq"):
        raise ValueError(f"estimator must be 'poisson' or 'leastsq', got {estimator!r}")
    tau_short_bounds, tau_long_bounds, tau_mono_bounds = _resolve_bounds(
        tau_short_bounds, tau_long_bounds, tau_mono_bounds)
    yv = summed[s_idx:].astype(float)      # tail portion of the summed (whole-image) decay

    if mono:
        def model(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        p0 = [yv.max(), sum(tau_mono_bounds) / 2, (yv.min() + 1) if fit_background else 0.0]
        c_max = np.inf if fit_background else 0.0
        free = [0, 1, 2] if fit_background else [0, 1]
        if estimator == "poisson":
            refined = _poisson_refine(model, tn, yv, p0,
                                      [(0, np.inf), tuple(tau_mono_bounds), (0, c_max)])
            if refined is not None:
                popt, dev = refined
                return dict(tau=popt[1], tau_std=_param_std_full(dev, popt, 1, free=free),
                            popt=popt, model=model, yv=yv, mono=True, estimator="poisson",
                            fit_background=fit_background)
            warnings.warn("global_fit (mono): Poisson fit failed, using curve_fit result instead.",
                          RuntimeWarning)
        popt, pcov = _curve_fit_tail(model, tn, yv, p0,
                                     [0, tau_mono_bounds[0], 0], [np.inf, tau_mono_bounds[1], np.inf],
                                     fit_background)
        A, tau, C = popt
        return dict(tau=tau, tau_std=_curve_fit_std(pcov, 1),
                    popt=popt, model=model, yv=yv, mono=True, estimator="leastsq",
                    fit_background=fit_background)

    def model(t, A1, t1, A2, t2, C):
        # Biexponential decay + constant background offset C.
        return A1 * np.exp(-t / t1) + A2 * np.exp(-t / t2) + C

    # Initial guess: split the peak amplitude ~60/40 between a short and long
    # component, seed lifetimes in the middle of their expected ranges.
    p0 = [yv.max() * 0.6, sum(tau_short_bounds) / 2, yv.max() * 0.4, sum(tau_long_bounds) / 2,
          (yv.min() + 1) if fit_background else 0.0]
    c_max = np.inf if fit_background else 0.0
    free = [0, 1, 2, 3, 4] if fit_background else [0, 1, 2, 3]
    # Bounds pin t1 to the "short" range and t2 to the "long" range so the fit
    # can't swap/confuse the two components.
    lo = [0, tau_short_bounds[0], 0, tau_long_bounds[0], 0]
    hi = [np.inf, tau_short_bounds[1], np.inf, tau_long_bounds[1], np.inf]
    try:
        # sqrt(max(counts, 1)) weighting: approximate Poisson (shot-noise) weighting
        popt, pcov = _curve_fit_tail(model, tn, yv, p0, lo, hi, fit_background)
    except (RuntimeError, ValueError):
        if estimator != "poisson":
            raise
        popt, pcov = np.array(p0, dtype=float), None      # curve_fit failed: seed the Poisson fit from p0
    std_of = lambda i: _curve_fit_std(pcov, i)
    used = "leastsq"
    if estimator == "poisson":
        refined = _poisson_refine(model, tn, yv, popt, list(zip(lo, hi[:-1] + [c_max])))
        if refined is not None:
            popt, dev = refined
            std_of = lambda i: _param_std_full(dev, popt, i, free=free)
            used = "poisson"
        elif pcov is None:
            raise RuntimeError("global_fit: both curve_fit and the Poisson fit failed.")
        else:
            warnings.warn("global_fit: Poisson refinement failed, using curve_fit result instead.",
                          RuntimeWarning)
    A1, t1, A2, t2, C = popt
    # t1 is always the "short" range and t2 the "long" range (enforced by
    # the bounds above), so unlike the per-pixel free fit, no index-tracking
    # is needed to know which covariance entry belongs to which lifetime.
    ts, tl = sorted([t1, t2])   # tau_short = smaller lifetime, tau_long = larger, regardless of fit order
    return dict(tau_short=ts, tau_long=tl,
               tau_short_std=std_of(1), tau_long_std=std_of(3),
               popt=popt, model=model, yv=yv, mono=False, estimator=used,
               fit_background=fit_background)


class AnalysisCancelled(Exception):
    """Raised from inside a per-pixel fitting loop when the caller's
    cancel_check() reports the user asked to stop -- caught at the top
    level (the GUI's worker thread, or a script's own try/except) to end
    the run cleanly partway through instead of as an error."""
    pass


def fit_pixels_mono(cube, s_idx, tn, mask, tau_mono_bounds=None, cancel_check=None,
                    fit_background=True):
    """Per-pixel mono-exponential tail fit on masked pixels (A, tau, C -- 3
    free parameters, half of the biexponential's 5).

    Returns a dict of [H, W] maps (NaN/False where not fitted):
      tau, tau_std, n_photons, converged, nll
    See the module docstring for what each one means.

    cancel_check: optional zero-arg callable, checked once per pixel;
    raises AnalysisCancelled (with maps filled in so far discarded by the
    caller) if it returns True -- lets a GUI Stop button interrupt a long
    per-pixel loop promptly instead of only between whole plugin runs.

    fit_background: True (default) fits a constant background C per pixel.
    False holds C = 0 (only for data without background -- see global_fit)."""
    _, _, tau_mono_bounds = _resolve_bounds(None, None, tau_mono_bounds)
    H, W, _ = cube.shape
    tail = cube[:, :, s_idx:]
    tau_map = np.full((H, W), np.nan)
    tau_std_map = np.full((H, W), np.nan)
    n_photons_map = np.zeros((H, W))
    converged_map = np.zeros((H, W), dtype=bool)
    nll_map = np.full((H, W), np.nan)
    tau_seed = sum(tau_mono_bounds) / 2

    def model(p):
        A, tau, C = p
        return A * np.exp(-tn / tau) + C

    def nll(p, k):
        m = np.maximum(model(p), 1e-9)
        return np.sum(m - k * np.log(m))

    ys, xs = np.where(mask)
    total = len(ys)
    print(f"  fitting {total} pixels (mono-exponential) ...", flush=True)
    for n, (i, j) in enumerate(zip(ys, xs)):
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelled(f"Cancelled after {n}/{total} pixels.")
        k = tail[i, j]
        pk = max(k[:5].mean(), 1.0)
        bg = max(k[-20:].mean(), 0.5) if fit_background else 0.0
        p0 = [pk, tau_seed, bg]
        bnds = [(0, pk * 6), tau_mono_bounds, (0, pk if fit_background else 0.0)]
        r = minimize(nll, p0, args=(k,), method="L-BFGS-B", bounds=bnds)
        tau_map[i, j] = r.x[1]
        tau_std_map[i, j] = _param_std_fisher(model, r.x, 1, bounds=bnds,
                                               fallback=lambda: _param_std(nll, r.x, 1, args=(k,)))
        n_photons_map[i, j] = k.sum()
        converged_map[i, j] = bool(r.success)
        nll_map[i, j] = r.fun
        if total > 500 and (n + 1) % 500 == 0:
            print(f"    {n + 1}/{total}", flush=True)
    return dict(tau=tau_map, tau_std=tau_std_map, n_photons=n_photons_map,
               converged=converged_map, nll=nll_map)


def fit_pixels(cube, s_idx, tn, mask, tau_short_fixed=None,
               tau_short_bounds=None, tau_long_bounds=None, cancel_check=None,
               fit_background=True):
    """
    Per-pixel biexponential tail fit on masked pixels.
      tau_short_fixed=None -> both lifetimes float (free biexponential)
      tau_short_fixed=value -> tau_short fixed, only tau_long + amplitudes float
        (tau_short_std is NaN in this mode -- it was pinned, not fitted)

    Returns a dict of [H, W] maps (NaN/False where not fitted):
      tau_long, tau_short, tau_long_std, tau_short_std, n_photons, converged, nll

    cancel_check: see fit_pixels_mono -- same optional per-pixel interrupt.
    fit_background: see fit_pixels_mono -- False holds the per-pixel C = 0.
    """
    tau_short_bounds, tau_long_bounds, _ = _resolve_bounds(tau_short_bounds, tau_long_bounds, None)
    H, W, _ = cube.shape
    tail = cube[:, :, s_idx:]
    tau_long  = np.full((H, W), np.nan)
    tau_short = np.full((H, W), np.nan)
    tau_long_std  = np.full((H, W), np.nan)
    tau_short_std = np.full((H, W), np.nan)
    n_photons_map = np.zeros((H, W))
    converged_map = np.zeros((H, W), dtype=bool)
    nll_map = np.full((H, W), np.nan)
    ts_seed = sum(tau_short_bounds) / 2
    tl_seed = sum(tau_long_bounds) / 2

    def model_free(p):
        A1, t1, A2, t2, C = p
        return A1 * np.exp(-tn / t1) + A2 * np.exp(-tn / t2) + C

    def model_fixed(p):
        A1, A2, t2, C = p
        return A1 * np.exp(-tn / tau_short_fixed) + A2 * np.exp(-tn / t2) + C

    def nll_free(p, k):
        m = np.maximum(model_free(p), 1e-9)
        return np.sum(m - k * np.log(m))

    def nll_fixed(p, k):
        m = np.maximum(model_fixed(p), 1e-9)
        return np.sum(m - k * np.log(m))

    ys, xs = np.where(mask)
    total = len(ys)
    print(f"  fitting {total} pixels ...", flush=True)
    for n, (i, j) in enumerate(zip(ys, xs)):
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelled(f"Cancelled after {n}/{total} pixels.")
        k = tail[i, j]
        pk = max(k[:5].mean(), 1.0)
        bg = max(k[-20:].mean(), 0.5) if fit_background else 0.0
        c_max = pk if fit_background else 0.0
        if tau_short_fixed is None:
            p0 = [pk * 0.6, ts_seed, pk * 0.4, tl_seed, bg]
            bnds = [(0, pk * 6), tau_short_bounds, (0, pk * 6), tau_long_bounds, (0, c_max)]
            r = minimize(nll_free, p0, args=(k,), method="L-BFGS-B", bounds=bnds)
            t1, t2 = r.x[1], r.x[3]
            # Track which parameter INDEX ended up holding the short vs long
            # lifetime (the optimizer isn't required to keep t1 <= t2), so
            # the right covariance-index gets the right std.
            idx_short, idx_long = (1, 3) if t1 <= t2 else (3, 1)
            tau_short[i, j], tau_long[i, j] = sorted([t1, t2])
            tau_short_std[i, j] = _param_std_fisher(model_free, r.x, idx_short, bounds=bnds,
                                                    fallback=lambda: _param_std(nll_free, r.x, idx_short, args=(k,)))
            tau_long_std[i, j] = _param_std_fisher(model_free, r.x, idx_long, bounds=bnds,
                                                   fallback=lambda: _param_std(nll_free, r.x, idx_long, args=(k,)))
        else:
            p0 = [pk * 0.6, pk * 0.4, tl_seed, bg]
            bnds = [(0, pk * 6), (0, pk * 6), tau_long_bounds, (0, c_max)]
            r = minimize(nll_fixed, p0, args=(k,), method="L-BFGS-B", bounds=bnds)
            tau_short[i, j] = tau_short_fixed
            tau_long[i, j]  = r.x[2]
            tau_long_std[i, j] = _param_std_fisher(model_fixed, r.x, 2, bounds=bnds,
                                                   fallback=lambda: _param_std(nll_fixed, r.x, 2, args=(k,)))
        n_photons_map[i, j] = k.sum()
        converged_map[i, j] = bool(r.success)
        nll_map[i, j] = r.fun
        if total > 500 and (n + 1) % 500 == 0:
            print(f"    {n + 1}/{total}", flush=True)
    return dict(tau_long=tau_long, tau_short=tau_short,
               tau_long_std=tau_long_std, tau_short_std=tau_short_std,
               n_photons=n_photons_map, converged=converged_map, nll=nll_map)


def mean_arrival_time(cube, s_idx, tn, mask, subtract_background=True, correct_bias=True):
    """Photon-weighted mean arrival time (the decay's first moment) of
    every masked pixel's tail -- a fast, non-iterative alternative to
    exponential fitting:

        tau_mean = sum(t_i * n_i) / sum(n_i)

    For a true mono-exponential decay this equals tau exactly (in the
    noise-free limit) -- no fitting needed to get the same number. For a
    biexponential decay it is NOT tau_short or tau_long (it's closer to
    their intensity-weighted average), so it can't replace a biexponential
    fit when you need the two separate components -- but because it's
    cheap, robust at low photon counts (no optimizer to fail to converge,
    no bounds to hit), and monotonic with the true lifetime(s), it's a good
    fast trend indicator: e.g. scanning many settings quickly, or watching
    a timelapse for a lifetime shift, before committing to a full fit.

    subtract_background=True (default) removes a flat background level
    (estimated from the tail's own last ~20 bins, same estimate the
    exponential fits use) before computing the moment -- background counts
    that aren't part of the real decay would otherwise bias the mean toward
    later times.

    correct_bias=True (default) removes two small systematic errors, both of
    which pull the result LOW (about 3-4 % for a 6 ns decay, 1-2 % for 2 ns):
      * the cutoff below discards the last ~0.5 % of the pooled signal. That
        is the LATEST light, so it carries a large time-weight and its loss
        lowers the mean (5.75 instead of 5.95 ns for a 6 ns decay). The
        missing part is extrapolated from the pooled curve's own tail
        (_tail_extrapolation) and scaled for each pixel by that pixel's share
        of the late signal, so a short-lived pixel in the same mask is not
        over-corrected. Skipped when the tail cannot be extrapolated
        reliably (too few counts, or the decay is far from finished).
      * times are taken at bin CENTRES instead of left edges (+half a bin,
        0.025 ns at 50 ps).
    False reproduces the previous numbers.

    Also returns the standard error of the mean -- exact, not an
    approximation, for a weighted-mean estimator:

        SEM = sqrt(Var(t) / N),  Var(t) = sum(n_i * (t_i - tau_mean)^2) / N

    where Var(t) is the photon-weighted variance of arrival times and N is
    the (background-subtracted) photon count. This is a genuinely different
    (and cheaper/exact, vs. the fit-curvature approximation) kind of
    uncertainty estimate than tau_std on the exponential-fit maps, but
    plays the same role: large where a pixel's estimate isn't well
    constrained, almost always because of low photon counts.

    IMPORTANT -- the tail window is truncated before computing the moment,
    NOT used all the way to the end of the available bin range. Mean
    arrival time is a TIME-WEIGHTED statistic, so a bin far out in a long
    acquisition window carries a large time-weight even though, once the
    real decay has died away, it carries essentially zero real signal --
    just background-subtraction noise. On real low-photon-count data (a
    dim HEK cell acquisition), this produced wildly wrong per-pixel values
    (tens of ns, some negative) even for pixels with 200+ total photons,
    traced to a single noisy bin ~20 ns out in an otherwise-empty tail
    that happened to extend to ~40 ns. So before the per-pixel calculation,
    the POOLED (masked-region-summed) tail is used to find where the real,
    background-subtracted signal has already been captured (99.5% by
    default -- see _tail_signal_cutoff), and every pixel's own calculation
    is truncated to that same window. Pooling first keeps this decision
    itself statistically stable (using far more photons than any one
    pixel has) while still applying one consistent window to every pixel.

    Fully vectorized (no per-pixel Python loop) -- this is the main
    practical advantage over the exponential fits for a quick look at many
    settings or a large timelapse: computing it for every pixel is a few
    numpy calls, not thousands of individual optimizer runs.

    Returns a dict of [H, W] maps (NaN/0 where not fitted):
      tau_mean, tau_mean_std, n_photons
    """
    H, W, _ = cube.shape
    tail = cube[:, :, s_idx:].astype(float)   # [H, W, n_tail_bins]

    if subtract_background:
        n_bg_bins = min(20, tail.shape[2])
        bg = tail[:, :, -n_bg_bins:].mean(axis=2, keepdims=True)
        tail = tail - bg
        # Deliberately NOT clipped to >=0 here. Clipping negative
        # (pure-noise) residuals to zero after subtraction looks like the
        # safe thing to do, but it is not: it removes only the negative
        # half of the shot noise around zero, and it does so specifically
        # in the late-tail bins where the real decay signal has died out
        # and only background-level counts remain -- exactly the bins that
        # carry the LARGEST time weight in a first-moment sum. That
        # asymmetric removal systematically drags tau_mean later (empirically,
        # roughly a factor of ~2 worse bias than leaving negatives in, on a
        # controlled test against known ground truth). Leaving negative
        # residuals in the sum is what makes the background-subtracted
        # estimator actually unbiased: E[k_i - bg] = true_signal_i exactly
        # when bg is an unbiased background estimate, whether that
        # difference happens to land above or below zero in a given bin.
        bg_level = bg[:, :, 0]
    else:
        bg_level = np.zeros((H, W))

    pooled = tail[mask].sum(axis=0) if mask.any() else tail.sum(axis=(0, 1))
    cutoff = _tail_signal_cutoff(pooled)
    ext = _tail_extrapolation(pooled, cutoff) if correct_bias else None
    dt = float(tn[1] - tn[0]) if len(tn) > 1 else 0.0
    tail = tail[:, :, :cutoff]
    tn = tn[:cutoff]
    if correct_bias:
        tn = tn + 0.5 * dt             # bin centres, not left edges

    n_photons_raw = tail.sum(axis=2)   # [H, W] -- may be slightly negative for a
                                        # pixel with ~no real signal above an
                                        # over-subtracted background estimate.

    with np.errstate(invalid="ignore", divide="ignore"):
        weighted_sum = np.tensordot(tail, tn, axes=([2], [0]))
        tau_trunc = weighted_sum / n_photons_raw     # mean over the kept window
        if ext is not None:
            # Each pixel's own share of the late (pre-cutoff) signal scales the
            # extrapolated missing tail: no late light -> no correction.
            share = tail[:, :, cutoff - ext["share_len"]:cutoff].sum(axis=2) / ext["s_share"]   # NOT clipped: noise stays zero-mean
            missing_n = share * ext["n_ext"]
            missing_t = share * dt * (ext["m_ext"] + 0.5 * ext["n_ext"])    # sum(time * counts), ns
            tau_mean_map = (weighted_sum + missing_t) / (n_photons_raw + missing_n)
        else:
            tau_mean_map = tau_trunc
        # the SEM describes the spread of the photons actually observed, i.e.
        # around the mean of the kept window
        deviation_sq = (tn[None, None, :] - tau_trunc[:, :, None]) ** 2
        var_t = np.sum(tail * deviation_sq, axis=2) / n_photons_raw
        # var_t is a sum of SIGNED terms (unclipped tail * a squared, always
        # non-negative, distance), not a true sum-of-squares, so it can come
        # out negative -- and on real data this isn't always the tiny
        # floating-point noise it looks like at first: confirmed cases with
        # var_t in the hundreds to thousands (bins^2) for pixels that
        # otherwise looked fine (decent net count, comfortably past the
        # noise-floor check below). Flooring at 0 there and reporting that as
        # tau_mean_std=0 is actively misleading -- it displays as the pixel
        # with the HIGHEST confidence on a std map, when what actually
        # happened is the opposite: a single noisy bin far from tau_mean (the
        # squared-distance weighting makes far bins count a lot) dominated
        # the sum. So var_t < 0 is folded into the validity mask below
        # instead of being floored -- an unreliable pixel is marked NaN, not
        # silently displayed as suspiciously precise.
        tau_mean_std_map = np.sqrt(np.maximum(var_t, 0) / n_photons_raw)

    valid = mask & (n_photons_raw > 0) & (var_t >= 0)
    # A positive net count isn't enough on its own: with the tail summed
    # over `cutoff` bins of background at bg_level/bin, pure noise alone
    # has a standard deviation of about sqrt(cutoff * bg_level) -- a net
    # count only a little above 0 is consistent with "no real signal",
    # and the RATIO estimator (mean = weighted_sum / n) is unstable
    # exactly when the denominator is close to that noise floor (confirmed
    # on real data: a pixel with a net count of 3.3 photons gave
    # tau_mean = -147 ns). Require the net count to clear that noise floor
    # by a real margin before trusting the pixel at all.
    noise_floor = np.sqrt(np.maximum(cutoff * bg_level, 1.0))
    valid &= n_photons_raw > 3 * noise_floor
    tau_mean_map = np.where(valid, tau_mean_map, np.nan)
    tau_mean_std_map = np.where(valid, tau_mean_std_map, np.nan)
    # n_photons is reported/plotted (e.g. the noise-floor scatter in
    # plot_fit_quality), where a small negative number from background
    # over-subtraction noise would be a confusing thing to show -- clip to
    # >=0 for that display value only; it plays no further role in the
    # tau_mean/tau_mean_std computation above, which already used the
    # unclipped n_photons_raw.
    n_photons_map = np.where(mask, np.clip(n_photons_raw, 0, None), 0.0)

    return dict(tau_mean=tau_mean_map, tau_mean_std=tau_mean_std_map, n_photons=n_photons_map)


def _tail_signal_cutoff(pooled_bs, min_fraction=0.995, min_bins=20):
    """Bin index (relative to the tail start) beyond which the pooled,
    background-subtracted tail curve has already captured min_fraction of
    its own total (clipped-non-negative) integral -- see mean_arrival_time
    for why this matters. Monotonic and robust to a bit of far-tail noise,
    since it's based on cumulative captured signal rather than a per-bin
    significance test that a single noisy bin could trip. Falls back to
    the full window if the pooled curve has no real net signal at all
    (nothing to truncate against), and never returns fewer than min_bins
    even for a very fast-decaying/low-signal curve."""
    clipped = np.clip(pooled_bs, 0, None)
    total = clipped.sum()
    n = len(pooled_bs)
    if total <= 0:
        return n
    cum = np.cumsum(clipped)
    idx = int(np.searchsorted(cum, min_fraction * total)) + 1
    return max(min(idx, n), min(min_bins, n))


def _tail_extrapolation(pooled_bs, cutoff, min_block_counts=100.0, max_missing=0.05):
    """Estimate the signal that the cutoff throws away, by extrapolating the
    pooled background-subtracted curve beyond `cutoff` as a single
    exponential. Its decay constant comes from the ratio of the summed signal
    in the last two equal blocks (h bins each) before the cutoff, which is
    robust to the negative residuals the background subtraction leaves.

    Returns a dict, in units of counts and bin index counted from the start
    of `pooled_bs`:
      n_ext    the missing counts (sum from the cutoff to infinity)
      m_ext    their first moment, sum(index * counts)
      share_len, s_share  length of the window just before the cutoff and the
               pooled signal in it; each pixel scales the correction by ITS
               OWN share of that window, so a pixel that has no late light
               is not corrected at all
    or None when no reliable extrapolation exists: fewer than
    min_block_counts counts in the block (the decay constant would be noise;
    100 counts keeps its relative error near 15 %, and the correction itself
    is only ~3 % of tau), a curve that is not decaying there, or an
    estimated missing share above max_missing (a normal 99.5 % cutoff
    misses ~0.5 %; much more means the decay is far from finished in the
    window, and a longer window is the fix, not an extrapolation)."""
    c = int(cutoff)
    h = max(10, c // 5)
    if c < 2 * h:
        return None
    s1 = float(pooled_bs[c - 2 * h:c - h].sum())
    s2 = float(pooled_bs[c - h:c].sum())
    if s2 < min_block_counts or s1 <= s2:
        return None
    q = (s2 / s1) ** (1.0 / h)                      # per-bin decay ratio, 0 < q < 1
    if not (0.0 < q < 1.0):
        return None
    peak_at_c = s2 / np.sum(q ** -np.arange(1, h + 1))   # fitted signal in the first excluded bin
    n_ext = peak_at_c / (1.0 - q)
    m_ext = peak_at_c * (c / (1.0 - q) + q / (1.0 - q) ** 2)
    n_in = float(np.clip(pooled_bs[:c], 0, None).sum())
    if n_in <= 0 or n_ext / n_in > max_missing:
        return None
    # Window over which each pixel's share of the late signal is measured: 3
    # blocks, because a pixel's share from one short block is dominated by
    # shot noise, and that noise is correlated with the pixel's own moment
    # (the same late photons feed both), which inflated the per-pixel scatter
    # by ~25 %; the wider window brings that down to ~7 %.
    share_len = min(3 * h, c)
    return dict(n_ext=n_ext, m_ext=m_ext, share_len=share_len,
                s_share=float(pooled_bs[c - share_len:c].sum()))
