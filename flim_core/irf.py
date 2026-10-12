"""
IRF reconvolution fitting: load/align/background-subtract a measured
instrument response function, build convolved decay models, and fit the
FULL decay curve (not just the tail) against it.

This is a fit METHOD (an alternative to the plain tail fit in fitting.py),
not an analysis plugin of its own -- every plugin in analyses/ that declares
IRF_CAPABLE = True calls into this module when the user picks fit_method="irf".
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import fftconvolve

from .config import IRF_SHIFT_SEARCH, TAU_SHORT_BOUNDS, TAU_LONG_BOUNDS, TAU_MONO_BOUNDS
from .io import load_flim_data, build_decay_cube
from .fitting import _resolve_bounds, _param_std, AnalysisCancelled, _tail_signal_cutoff


def load_irf(path, window=True, window_end_ns=15.0, time_bin=1):
    """Load an IRF measurement file (any of the three Flimera HDF5 formats,
    same loader as the main script) and reduce it to a single normalized
    instrument-response curve, summed over every pixel -- the IRF shape is a
    property of the optics/detector, not the sample, so pooling every photon
    across the whole sensor gives the cleanest possible estimate of it.

    IMPORTANT: real IRF measurements (e.g. a foil/scatter shot) carry a flat
    detector dark-count floor spread roughly UNIFORMLY across the whole
    recorded window -- present both before and after the true pulse, at
    basically the same level. That floor is detector noise, not part of the
    optical pulse shape, and must NOT be baked into the convolution kernel:
    doing so smears a large, spurious flat component into every fitted
    pixel's model (in this file's actual IRF, background turned out to be
    ~64% of the RAW IRF's total counts), which was found to collapse
    per-pixel lifetime fits onto their lower bounds. So the background level
    (estimated from the flat tail, far past the peak) is subtracted before
    normalizing, regardless of the window setting below.

    window=True (default): after background subtraction, keep only the
    single CONTIGUOUS run of bins around the main peak that stays
    statistically above the background floor (3 sigma of the background's
    own Poisson noise), zeroing out everything else. This is the right
    choice when any secondary bump in the IRF trace is a measurement
    artifact you don't want in the kernel -- e.g. a genuine reflection off
    an unrelated optical surface that happens to be specific to how THIS
    particular IRF shot was set up, not something that would also affect
    the real sample light path the same way. Since a secondary bump has to
    cross back down through the noise floor to reach the main peak, it can
    never survive this contiguous walk, without needing to know in advance
    whether one is present or assume anything about its cause (in one real
    IRF file, an echo peak ~7 ns after the main pulse carried 3.6% of the
    kernel's total normalized mass after background subtraction alone, and
    is exactly the kind of thing this removes).

    window=False: use instead when you have reason to believe secondary
    peaks in the IRF ARE part of the true instrument response for this
    setup (e.g. reproducible laser cavity or SPAD afterpulsing echoes that
    the real sample decay is convolved with just as much as the main pulse
    is) -- keeping them in the kernel is then the more physically correct
    choice, not a contamination to strip out. Keeps everything from the
    same peak-relative starting point as the windowed version out to
    `window_end_ns` after the peak (default 15 ns) rather than stopping at
    the first dip below the noise floor, so real secondary peaks separated
    from the main one by a noise-level valley survive; background is
    re-estimated from bins PAST that extended window (not the fixed
    peak+200-bin margin windowed mode uses, which can sit inside the region
    window=False is trying to keep) so the two settings' background
    estimates don't contaminate each other. Pick window_end_ns by looking
    at the raw IRF trace yourself: it should sit past every real secondary
    peak but still well before the point the trace visibly flattens into
    pure background -- confirmed on one real IRF file with peaks at ~2.9 ns
    and ~5.9-6.1 ns after the main pulse and a flat background from ~8 ns
    on, where the default 15 ns comfortably clears both.

    time_bin: group every N adjacent microtime bins into one BEFORE any of
    the background/windowing logic runs -- must match the time_bin applied
    to the decay cube (flim_core.masking.bin_time_cube), or the
    reconvolution model's two time grids no longer line up. The returned
    dict's `tpb` is the binned bin width (original tpb * time_bin), and
    nb/peak_bin/window are all in binned-bin units, so callers can treat
    the result exactly like an IRF loaded from natively-coarser data."""
    s = load_flim_data(path)
    cube = build_decay_cube(s)
    raw = cube.sum(axis=(0, 1)).astype(float)          # summed over all pixels
    if time_bin and time_bin > 1:
        nbc = (len(raw) // time_bin) * time_bin
        raw = raw[:nbc].reshape(-1, time_bin).sum(axis=1)
        s = dict(s)
        s["tpb"] = s["tpb"] * time_bin
    peak_bin = int(np.argmax(raw))
    half = raw[peak_bin] / 2
    above = np.where(raw >= half)[0]
    fwhm_bins = int(above.max() - above.min() + 1) if above.size else 0
    tpb_ns = s["tpb"] * 1e9

    # Background: median of the far tail (well past the peak, where the true
    # pulse response has fully died away), representative of the flat dark
    # count floor present everywhere in the window. Where that tail starts
    # depends on the window mode -- see the window=False docs above for why.
    if window:
        # 200 bins is ~7.5 ns at the native 37.4 ps/bin; rescaled so the real-ns
        # margin stays the same under time binning
        tail_start = min(peak_bin + max(1, round(200 / max(time_bin, 1))), len(raw) - 50)
    else:
        window_end_bins = int(round(window_end_ns / tpb_ns))
        tail_start = min(peak_bin + window_end_bins, len(raw) - 50)
    bg_level = float(np.median(raw[tail_start:])) if tail_start < len(raw) else 0.0
    bg_fraction = bg_level * len(raw) / raw.sum() if raw.sum() > 0 else 0.0
    raw_bs = raw - bg_level   # not clipped yet -- windowing (below) needs to see the
                              # true sign to find where the pulse drops back to noise

    noise_sigma = np.sqrt(max(bg_level, 1.0))
    significant = raw_bs > 3 * noise_sigma
    left = peak_bin
    while left > 0 and significant[left - 1]:
        left -= 1
    if window:
        # Window out anything not contiguously connected to the main peak: a
        # secondary bump elsewhere in the window has to cross back down
        # through the noise floor to get there, so it can never survive
        # this walk.
        right = peak_bin
        while right < len(raw) - 1 and significant[right + 1]:
            right += 1
    else:
        # Fixed extent instead of a contiguous walk -- deliberately does NOT
        # stop at the first noise-level dip after the main peak, so real
        # secondary peaks separated by a valley are kept (see the
        # window=False docs above).
        right = min(peak_bin + window_end_bins, len(raw) - 1)
    windowed = np.zeros_like(raw_bs)
    windowed[left:right + 1] = raw_bs[left:right + 1]
    echo_fraction = 1.0 - (np.clip(windowed, 0, None).sum() / max(np.clip(raw_bs, 0, None).sum(), 1e-9))

    raw_bs = np.clip(windowed, 0, None)
    irf_norm = raw_bs / raw_bs.sum()                    # unit-area kernel for convolution

    return dict(raw=raw, irf_norm=irf_norm, tpb=s["tpb"], nb=len(raw),
                peak_bin=peak_bin, fwhm_bins=fwhm_bins, bg_level=bg_level,
                bg_fraction=bg_fraction, window=(left, right), echo_fraction=echo_fraction,
                n_photons=s["n_photons"], source=s["source"], windowed=window)


def shift_array(a, shift):
    """Shift a 1-D array by an integer number of samples, filling the
    vacated end with zeros (NOT a circular roll) -- appropriate here since
    the recorded window is a single laser period, not a periodic signal.
    Integer-only -- kept for the diagnostic plot overlay. For the actual
    fit models, use shift_signal_subbin below instead: the true IRF/data
    timing offset is not generally an integer number of bins, and forcing
    it to one leaves a systematic residual that the amplitude/tau
    parameters can't compensate for (confirmed on real data: this alone
    was enough to pin tau_short to its lower bound in every test)."""
    n = len(a)
    out = np.zeros_like(a)
    if shift == 0:
        return a.copy()
    elif shift > 0:
        out[shift:] = a[: n - shift]
    else:
        out[: n + shift] = a[-shift:]
    return out


def shift_signal_subbin(a, shift_bins):
    """Shift a 1-D array by a continuous (non-integer) number of samples,
    via an FFT phase ramp -- a fractional-bin generalization of
    shift_array. Used INSIDE the reconvolution models so the fit's own
    optimizer can refine the IRF/data timing alignment to sub-bin
    precision as part of the same least-squares/MLE search that fits
    A/tau/C, rather than relying on a separate integer-only pre-alignment
    step. Circular (wraps at the array boundary) -- negligible effect here
    since the real decay has died back to baseline well before either
    array edge."""
    n = len(a)
    freqs = np.fft.fftfreq(n, d=1.0)
    return np.real(np.fft.ifft(np.fft.fft(a) * np.exp(-1j * 2 * np.pi * freqs * shift_bins)))


def estimate_shift(data_summed, irf_norm, max_shift=IRF_SHIFT_SEARCH):
    """Find the integer bin shift that best aligns the IRF's timing to the
    data's, by directly cross-correlating the two SUMMED (whole-image) curves.
    Restricted to a search window so noise can't produce a wild, unphysical
    shift.

    IMPORTANT: this is a COARSE estimate only, used as the initial guess
    (p0) for the continuous `shift` parameter that the reconvolution models
    below fit for real. On real data this dot-product cross-correlation
    turned out to be noisy and unstable on its own -- it returned a
    different shift (1 to 11 bins) for different photon-count subsets of
    the SAME underlying acquisition, which should have one consistent
    answer. Do not treat its return value as the final alignment; it exists
    only to keep the per-pixel optimizer's starting point in the right
    neighborhood so it converges quickly and doesn't wander to a
    physically implausible shift on a noisy pixel."""
    d = data_summed.astype(float) - data_summed.mean()
    g = irf_norm.astype(float) - irf_norm.mean()
    best_shift, best_score = 0, -np.inf
    for shift in range(-max_shift, max_shift + 1):
        score = np.dot(d, shift_array(g, shift))
        if score > best_score:
            best_score, best_shift = score, shift
    return best_shift


def make_conv_model_mono(irf_norm, tpb_ns, fix_shift=None):
    """Mono-exponential reconvolution model(params) -> full decay curve.
    Normally params are (A, tau, C, shift), with `shift` (bins, continuous)
    a genuine free fit parameter, not a pre-alignment step: the convolution
    is built at zero offset and then shifted to sub-bin precision via
    shift_signal_subbin, so the same optimizer that fits A/tau/C also
    resolves the IRF/data timing alignment as part of one consistent
    likelihood -- see shift_signal_subbin's docstring for why a separate
    pre-alignment step (the old estimate_shift+shift_array path) wasn't
    reliable enough to use directly.

    fix_shift=None (default): shift floats, params has 4 entries as above.
    fix_shift=<a number>: shift is fixed at that value (bins) instead of
    fitted -- params then has only 3 entries, (A, tau, C). Meant for
    fix_shift=0.0 specifically: if the IRF and the data were acquired at
    the same laser settings in the same session, there's no physical
    timing-walk reason to expect a real offset between them, and removing
    shift as a free parameter gives the optimizer one fewer degree of
    freedom to chase noise with -- worth it exactly when you're confident
    the premise (matched settings) actually holds."""
    nb = len(irf_norm)
    t = np.arange(nb) * tpb_ns

    def model(params):
        if fix_shift is None:
            A, tau, C, shift = params
        else:
            A, tau, C = params
            shift = fix_shift
        raw = A * np.exp(-t / tau)
        conv = fftconvolve(raw, irf_norm, mode="full")[:nb]
        return shift_signal_subbin(conv, shift) + C

    return model


def make_conv_model(irf_norm, tpb_ns, fix_shift=None):
    """Build model(params) -> full decay curve [nb]. Normally params are the
    free biexponential (A1, tau1, A2, tau2, C, shift). The raw biexponential
    decay is defined causally on the same bin grid as the IRF (starting at
    t=0 = bin 0), convolved with the IRF, truncated back to the original
    length (linear, non-circular reconvolution -- valid here because the
    decay fully returns to baseline well within the recorded window), and
    then shifted by the free `shift` parameter (bins, continuous -- see
    shift_signal_subbin) before the background C is added.

    fix_shift: see make_conv_model_mono -- same meaning here; params then
    has 5 entries, (A1, tau1, A2, tau2, C), with no shift term."""
    nb = len(irf_norm)
    t = np.arange(nb) * tpb_ns

    def model(params):
        if fix_shift is None:
            A1, t1, A2, t2, C, shift = params
        else:
            A1, t1, A2, t2, C = params
            shift = fix_shift
        raw = A1 * np.exp(-t / t1) + A2 * np.exp(-t / t2)
        conv = fftconvolve(raw, irf_norm, mode="full")[:nb]
        return shift_signal_subbin(conv, shift) + C

    return model


def make_conv_model_fixed(irf_norm, tpb_ns, tau_short_fixed, fix_shift=None):
    """Same as make_conv_model, but tau1 is pinned to tau_short_fixed --
    params are normally (A1, A2, tau2, C, shift). Mirrors the main script's
    --fix-tau-short option: fewer free parameters per pixel gives a much
    more stable tau_long estimate when individual (super-)pixels are
    photon-limited. `shift` still floats per fit by default (see
    make_conv_model); fix_shift: see make_conv_model_mono -- params then
    has 4 entries, (A1, A2, tau2, C), with no shift term."""
    nb = len(irf_norm)
    t = np.arange(nb) * tpb_ns

    def model(params):
        if fix_shift is None:
            A1, A2, t2, C, shift = params
        else:
            A1, A2, t2, C = params
            shift = fix_shift
        raw = A1 * np.exp(-t / tau_short_fixed) + A2 * np.exp(-t / t2)
        conv = fftconvolve(raw, irf_norm, mode="full")[:nb]
        return shift_signal_subbin(conv, shift) + C

    return model


def global_reconv_fit_mono(data_summed, irf_norm, tpb_ns, tau_mono_bounds=None, no_shift=False):
    """Mono-exponential reconvolution fit of the whole-image (or masked-only)
    summed decay -- reference/sanity-check fit, and available for --mono.

    no_shift=True: fix the IRF/data timing shift at 0 instead of fitting it
    (see make_conv_model_mono) -- use only when IRF and data were acquired
    at matched laser settings in the same session."""
    _, _, tau_mono_bounds = _resolve_bounds(None, None, tau_mono_bounds)
    yv = data_summed.astype(float)
    model = make_conv_model_mono(irf_norm, tpb_ns, fix_shift=0.0 if no_shift else None)

    def nll(p):
        m = np.maximum(model(p), 1e-9)
        return np.sum(m - yv * np.log(m))

    pk = yv.max()
    bg0 = max(np.percentile(yv, 2), 0.5)
    if no_shift:
        p0 = [pk, sum(tau_mono_bounds) / 2, bg0]
        bounds = [(0, pk * 10), tau_mono_bounds, (0, pk)]
        r = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        A, tau, C = r.x
        shift = 0.0
    else:
        shift0 = estimate_shift(yv, irf_norm)   # coarse integer seed only -- refined below
        p0 = [pk, sum(tau_mono_bounds) / 2, bg0, shift0]
        bounds = [(0, pk * 10), tau_mono_bounds, (0, pk), (-IRF_SHIFT_SEARCH, IRF_SHIFT_SEARCH)]
        r = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        A, tau, C, shift = r.x
    return dict(tau=tau, tau_std=_param_std(nll, r.x, 1), popt=r.x, model=model, yv=yv, mono=True, shift=shift)


def fit_pixels_reconv_mono(cube, irf_norm, tpb_ns, mask, tau_mono_bounds=None, cancel_check=None, no_shift=False):
    """Per-(super-)pixel mono-exponential reconvolution fit on masked pixels,
    using the full decay curve.

    Returns a dict of [H, W] maps (NaN/False where not fitted):
      tau, tau_std, n_photons, converged, nll

    cancel_check: optional zero-arg callable, checked once per pixel;
    raises AnalysisCancelled if it returns True (see flim_core.fitting).
    no_shift: see global_reconv_fit_mono -- same meaning, applied per pixel."""
    _, _, tau_mono_bounds = _resolve_bounds(None, None, tau_mono_bounds)
    H, W, nb = cube.shape
    model = make_conv_model_mono(irf_norm, tpb_ns, fix_shift=0.0 if no_shift else None)
    tau_map = np.full((H, W), np.nan)
    tau_std_map = np.full((H, W), np.nan)
    n_photons_map = np.zeros((H, W))
    converged_map = np.zeros((H, W), dtype=bool)
    nll_map = np.full((H, W), np.nan)
    tau_seed = sum(tau_mono_bounds) / 2
    shift_seed = None if no_shift else estimate_shift(cube.sum(axis=(0, 1)).astype(float), irf_norm)

    def nll(p, k):
        m = np.maximum(model(p), 1e-9)
        return np.sum(m - k * np.log(m))

    ys, xs = np.where(mask)
    total = len(ys)
    print(f"  fitting {total} pixels (mono-exponential, full-curve reconvolution"
          f"{', shift fixed at 0' if no_shift else ''}) ...", flush=True)
    for n, (i, j) in enumerate(zip(ys, xs)):
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelled(f"Cancelled after {n}/{total} pixels.")
        k = cube[i, j, :].astype(float)
        pk = max(k.max(), 1.0)
        bg = max(np.median(k[-max(20, nb // 20):]), 0.5)
        if no_shift:
            p0 = [pk, tau_seed, bg]
            bnds = [(0, pk * 6), tau_mono_bounds, (0, pk)]
        else:
            p0 = [pk, tau_seed, bg, shift_seed]
            bnds = [(0, pk * 6), tau_mono_bounds, (0, pk), (-IRF_SHIFT_SEARCH, IRF_SHIFT_SEARCH)]
        r = minimize(nll, p0, args=(k,), method="L-BFGS-B", bounds=bnds)
        tau_map[i, j] = r.x[1]
        tau_std_map[i, j] = _param_std(nll, r.x, 1, args=(k,))
        n_photons_map[i, j] = k.sum()
        converged_map[i, j] = bool(r.success)
        nll_map[i, j] = r.fun
        if total > 500 and (n + 1) % 500 == 0:
            print(f"    {n + 1}/{total}", flush=True)
    return dict(tau=tau_map, tau_std=tau_std_map, n_photons=n_photons_map,
               converged=converged_map, nll=nll_map)


def global_reconv_fit_fixed_short(data_summed, irf_norm, tpb_ns, tau_short_fixed, tau_long_bounds=None,
                                   no_shift=False):
    """Biexponential reconvolution fit of a summed decay curve with
    tau_short pinned to a known value; only tau_long (and the amplitudes/
    background) float. Mirrors global_reconv_fit, just with fewer free
    parameters -- used when a manual or previously-derived tau_short should
    be reused for a specific pooled-curve fit (e.g. one window/group of a
    whole-image timelapse/periodic run).

    no_shift: see global_reconv_fit_mono -- same meaning."""
    _, tau_long_bounds, _ = _resolve_bounds(None, tau_long_bounds, None)
    yv = data_summed.astype(float)
    model = make_conv_model_fixed(irf_norm, tpb_ns, tau_short_fixed, fix_shift=0.0 if no_shift else None)

    def nll(p):
        m = np.maximum(model(p), 1e-9)
        return np.sum(m - yv * np.log(m))

    pk = yv.max()
    bg0 = max(np.percentile(yv, 2), 0.5)
    if no_shift:
        p0 = [pk * 0.6, pk * 0.4, sum(tau_long_bounds) / 2, bg0]
        bounds = [(0, pk * 10), (0, pk * 10), tau_long_bounds, (0, pk)]
        r = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        A1, A2, t2, C = r.x
        shift = 0.0
    else:
        shift0 = estimate_shift(yv, irf_norm)
        p0 = [pk * 0.6, pk * 0.4, sum(tau_long_bounds) / 2, bg0, shift0]
        bounds = [(0, pk * 10), (0, pk * 10), tau_long_bounds, (0, pk), (-IRF_SHIFT_SEARCH, IRF_SHIFT_SEARCH)]
        r = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        A1, A2, t2, C, shift = r.x
    return dict(tau_short=tau_short_fixed, tau_long=t2, tau_long_std=_param_std(nll, r.x, 2),
               popt=r.x, model=model, yv=yv, mono=False, shift=shift)


def global_reconv_fit(data_summed, irf_norm, tpb_ns, tau_short_bounds=None, tau_long_bounds=None,
                       no_shift=False):
    """Robust reconvolution fit of the whole-image summed decay -- used both
    as a sanity-check reference and (optionally) to supply a fixed tau_short
    for the per-pixel fits.

    no_shift: see global_reconv_fit_mono -- same meaning."""
    tau_short_bounds, tau_long_bounds, _ = _resolve_bounds(tau_short_bounds, tau_long_bounds, None)
    yv = data_summed.astype(float)
    model = make_conv_model(irf_norm, tpb_ns, fix_shift=0.0 if no_shift else None)

    def nll(p):
        m = np.maximum(model(p), 1e-9)
        return np.sum(m - yv * np.log(m))

    pk = yv.max()
    bg0 = max(np.percentile(yv, 2), 0.5)
    if no_shift:
        p0 = [pk * 0.6, sum(tau_short_bounds) / 2, pk * 0.4, sum(tau_long_bounds) / 2, bg0]
        bounds = [(0, pk * 10), tau_short_bounds, (0, pk * 10), tau_long_bounds, (0, pk)]
        r = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        A1, t1, A2, t2, C = r.x
        shift = 0.0
    else:
        shift0 = estimate_shift(yv, irf_norm)
        p0 = [pk * 0.6, sum(tau_short_bounds) / 2, pk * 0.4, sum(tau_long_bounds) / 2, bg0, shift0]
        bounds = [(0, pk * 10), tau_short_bounds, (0, pk * 10), tau_long_bounds, (0, pk),
                  (-IRF_SHIFT_SEARCH, IRF_SHIFT_SEARCH)]
        r = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
        A1, t1, A2, t2, C, shift = r.x
    idx_short, idx_long = (1, 3) if t1 <= t2 else (3, 1)
    ts, tl = sorted([t1, t2])
    return dict(tau_short=ts, tau_long=tl,
               tau_short_std=_param_std(nll, r.x, idx_short), tau_long_std=_param_std(nll, r.x, idx_long),
               popt=r.x, model=model, yv=yv, shift=shift)


def fit_pixels_reconv(cube, irf_norm, tpb_ns, mask, tau_short_fixed=None,
                      tau_short_bounds=None, tau_long_bounds=None, cancel_check=None, no_shift=False):
    """
    Per-(super-)pixel reconvolution fit on masked pixels, using the FULL
    decay curve (not just a tail) -- this is what lets the fit use every
    collected photon and properly resolve lifetimes near the IRF's own width.
      tau_short_fixed=None -> both lifetimes float (free biexponential)
      tau_short_fixed=value -> tau_short fixed, only tau_long + amplitudes float
        (tau_short_std is NaN in this mode -- it was pinned, not fitted)

    Returns a dict of [H, W] maps (NaN/False where not fitted):
      tau_long, tau_short, tau_long_std, tau_short_std, n_photons, converged, nll

    cancel_check: see fit_pixels_reconv_mono -- same optional per-pixel interrupt.
    no_shift: see global_reconv_fit_mono -- same meaning, applied per pixel.
    """
    tau_short_bounds, tau_long_bounds, _ = _resolve_bounds(tau_short_bounds, tau_long_bounds, None)
    H, W, nb = cube.shape
    t = np.arange(nb) * tpb_ns
    tau_long = np.full((H, W), np.nan)
    tau_short = np.full((H, W), np.nan)
    tau_long_std = np.full((H, W), np.nan)
    tau_short_std = np.full((H, W), np.nan)
    n_photons_map = np.zeros((H, W))
    converged_map = np.zeros((H, W), dtype=bool)
    nll_map = np.full((H, W), np.nan)
    ts_seed = sum(tau_short_bounds) / 2
    tl_seed = sum(tau_long_bounds) / 2
    fix_shift = 0.0 if no_shift else None
    shift_seed = None if no_shift else estimate_shift(cube.sum(axis=(0, 1)).astype(float), irf_norm)

    model_free = make_conv_model(irf_norm, tpb_ns, fix_shift=fix_shift)
    model_fixed = make_conv_model_fixed(irf_norm, tpb_ns, tau_short_fixed, fix_shift=fix_shift) \
        if tau_short_fixed else None

    def nll_free(p, k):
        m = np.maximum(model_free(p), 1e-9)
        return np.sum(m - k * np.log(m))

    def nll_fixed(p, k):
        m = np.maximum(model_fixed(p), 1e-9)
        return np.sum(m - k * np.log(m))

    ys, xs = np.where(mask)
    total = len(ys)
    print(f"  fitting {total} pixels (full-curve reconvolution"
          f"{', shift fixed at 0' if no_shift else ''}) ...", flush=True)
    for n, (i, j) in enumerate(zip(ys, xs)):
        if cancel_check is not None and cancel_check():
            raise AnalysisCancelled(f"Cancelled after {n}/{total} pixels.")
        k = cube[i, j, :].astype(float)
        pk = max(k.max(), 1.0)
        bg = max(np.median(k[-max(20, nb // 20):]), 0.5)   # baseline from the far tail
        if tau_short_fixed is None:
            if no_shift:
                p0 = [pk * 0.6, ts_seed, pk * 0.4, tl_seed, bg]
                bnds = [(0, pk * 6), tau_short_bounds, (0, pk * 6), tau_long_bounds, (0, pk)]
            else:
                p0 = [pk * 0.6, ts_seed, pk * 0.4, tl_seed, bg, shift_seed]
                bnds = [(0, pk * 6), tau_short_bounds, (0, pk * 6), tau_long_bounds, (0, pk),
                        (-IRF_SHIFT_SEARCH, IRF_SHIFT_SEARCH)]
            r = minimize(nll_free, p0, args=(k,), method="L-BFGS-B", bounds=bnds)
            t1, t2 = r.x[1], r.x[3]
            idx_short, idx_long = (1, 3) if t1 <= t2 else (3, 1)
            tau_short[i, j], tau_long[i, j] = sorted([t1, t2])
            tau_short_std[i, j] = _param_std(nll_free, r.x, idx_short, args=(k,))
            tau_long_std[i, j] = _param_std(nll_free, r.x, idx_long, args=(k,))
        else:
            if no_shift:
                p0 = [pk * 0.6, pk * 0.4, tl_seed, bg]
                bnds = [(0, pk * 6), (0, pk * 6), tau_long_bounds, (0, pk)]
            else:
                p0 = [pk * 0.6, pk * 0.4, tl_seed, bg, shift_seed]
                bnds = [(0, pk * 6), (0, pk * 6), tau_long_bounds, (0, pk),
                        (-IRF_SHIFT_SEARCH, IRF_SHIFT_SEARCH)]
            r = minimize(nll_fixed, p0, args=(k,), method="L-BFGS-B", bounds=bnds)
            tau_short[i, j] = tau_short_fixed
            tau_long[i, j] = r.x[2]
            tau_long_std[i, j] = _param_std(nll_fixed, r.x, 2, args=(k,))
        n_photons_map[i, j] = k.sum()
        converged_map[i, j] = bool(r.success)
        nll_map[i, j] = r.fun
        if total > 500 and (n + 1) % 500 == 0:
            print(f"    {n + 1}/{total}", flush=True)
    return dict(tau_long=tau_long, tau_short=tau_short,
               tau_long_std=tau_long_std, tau_short_std=tau_short_std,
               n_photons=n_photons_map, converged=converged_map, nll=nll_map)


def mean_arrival_time_reconv(cube, irf_norm, tpb_ns, mask, no_shift=False):
    """IRF-corrected intensity-weighted mean arrival time, per masked pixel
    -- a fit-free (no nonlinear optimization) alternative to the biexponential
    reconvolution fit, in the same spirit as fitting.mean_arrival_time but
    corrected for the instrument response instead of just avoiding it via a
    tail-only window.

    Rests on the first-moment identity for a convolution: the mean of a
    convolution equals the SUM of the two inputs' own means,
        mean(data) = mean(true_decay) + mean(irf)
    so the true decay's own mean arrival time is recovered by simple
    subtraction:
        tau_mean = mean(data) - mean(irf)
    computed over the SAME (background-subtracted) time axis for both. This
    holds regardless of where the IRF's own peak happens to sit in the array
    -- verified against a synthetic decay convolved with this codebase's own
    measured IRF, recovering the true intensity-weighted lifetime to better
    than 0.05 ns with no alignment step at all, when decay and IRF share one
    time origin by construction.

    On REAL two-file data (a separately-recorded IRF file), that shared-
    origin assumption doesn't hold automatically: the IRF and decay
    acquisitions can carry a genuine relative timing offset (electronics,
    cabling, or just being separate acquisitions), the same offset the
    reconvolution fits solve for via a free `shift` parameter. A moment
    calculation has no such per-fit freedom to compensate, so this function
    aligns irf_norm to the data ONCE, via the same coarse estimate_shift()
    cross-correlation used to seed the reconvolution fits, before taking its
    moment. This won't be as precise as a fitted sub-bin shift (on real data,
    skipping it entirely left an uncorrected timing bias of several tenths
    of a ns), but keeps this method's core advantage -- no nonlinear fit at
    all, safe even at very low photon counts -- rather than requiring a
    biexponential fit first just to get a precise shift.

    Two more things this borrows directly from fitting.mean_arrival_time,
    after an early version of this function (using the full window and
    clipping negative residuals to zero) produced a per-pixel map that
    visibly tracked the INTENSITY image rather than looking flat -- exactly
    the two things that function's own docstring already warns matter:
      1. Background-subtracted residuals are NOT clipped to >=0. Clipping
         removes only the negative half of pure shot noise, specifically in
         the far-tail bins where only background remains and where a first
         moment weights each bin by its own (large) time position -- an
         asymmetric removal that drags tau_mean later, worse for dimmer
         pixels (where a larger fraction of the window is noise-only),
         which is exactly the dim-vs-bright/edge-vs-center pattern that
         showed up. Leaving negative residuals in keeps
         E[k_i - bg] = true_signal_i unbiased regardless of sign.
      2. The window is truncated to where the POOLED (whole-masked-region)
         background-subtracted curve has already captured 99.5% of its own
         signal (_tail_signal_cutoff, shared with fitting.py), instead of
         summing all the way to the end of the recorded range. Bins well
         past where the real decay has died away carry background-noise-only
         counts but still get a large time-weight in a first moment -- a
         single such pixel can swing tau_mean by tens of ns. Pooling first
         (far more photons than any one pixel) keeps the cutoff itself
         stable, then applies that SAME window to every pixel.
    The IRF's own moment is left over its full (already load_irf-windowed,
    already very compact) array -- truncating the data's far, empty tail
    doesn't need a matching change there, since none of the IRF's real mass
    lives out that far to begin with.

    NOTE this measures a genuinely different quantity than
    fitting.mean_arrival_time's tail-only version: this integrates the FULL
    causal decay (including the fast/short-lived component near t=0), while
    the tail-only version deliberately starts well after the peak and so is
    biased toward the longer-lived component. Don't expect the two numbers
    to agree -- on real data they differed by roughly the amount you'd
    expect from a real short-lifetime population (order 0.5-1 ns here).

    Returns dict with tau_mean [H,W] (ns) and n_photons [H,W], mirroring
    fitting.mean_arrival_time's return shape."""
    H, W, nb = cube.shape
    t_bins = np.arange(nb)

    signal_sum = cube[mask].sum(axis=0).astype(float)
    if no_shift:
        # See make_conv_model_mono's no_shift docs: matched laser settings
        # between IRF and data mean no physical timing-walk offset is
        # expected, so skip the alignment step entirely.
        shift = 0.0
        irf_aligned = irf_norm
    else:
        shift = estimate_shift(signal_sum, irf_norm)
        irf_aligned = shift_array(irf_norm, shift)
    irf_mean_bins = np.sum(t_bins * irf_aligned)   # irf_aligned already sums to 1

    n_bg_bins = min(20, nb)
    pooled_bs = signal_sum - np.mean(signal_sum[-n_bg_bins:])
    cutoff = _tail_signal_cutoff(pooled_bs)
    t_bins_c = t_bins[:cutoff]

    tau_mean = np.full((H, W), np.nan)
    tau_mean_std = np.full((H, W), np.nan)
    n_photons_map = np.zeros((H, W))
    ys, xs = np.where(mask)
    for i, j in zip(ys, xs):
        k = cube[i, j, :].astype(float)
        bg_i = k[-n_bg_bins:].mean()
        y = (k - bg_i)[:cutoff]   # NOT clipped -- see docstring above
        N = y.sum()
        n_photons_map[i, j] = max(N, 0.0)   # net (background-subtracted) count, clipped only
                                             # for this display value -- matches
                                             # fitting.mean_arrival_time's n_photons_map convention,
                                             # and is what the noise-floor scatter plot actually
                                             # means by "photons used in fit"
        # A positive net count isn't enough on its own to trust this pixel: with
        # `cutoff` bins of background at bg_i/bin, pure noise alone has a standard
        # deviation of about sqrt(cutoff * bg_i) -- a net count only a little above
        # 0 is consistent with "no real signal", and the ratio estimator
        # (mean = weighted_sum / N) is unstable exactly when N is close to that
        # noise floor. This is the same check fitting.mean_arrival_time uses
        # (confirmed there on real data: a net count of 3.3 photons gave
        # tau_mean = -147 ns) -- an earlier version of this function only checked
        # N <= 0, which is far too weak a filter and let exactly this kind of
        # pixel through, showing up as spurious std=0 (from var_t's noise-driven
        # negative-then-floored-to-0 swings) and wild/negative tau_mean values.
        noise_floor = np.sqrt(max(cutoff * bg_i, 1.0))
        if N <= 3 * noise_floor:
            continue
        data_mean_bins = np.sum(t_bins_c * y) / N
        tau_mean_i = (data_mean_bins - irf_mean_bins) * tpb_ns
        # Exact SEM of the weighted-mean estimator, same formula as
        # fitting.mean_arrival_time -- the IRF subtraction is a constant
        # shift, so it doesn't add variance; only the data's own spread does.
        var_t = np.sum(y * (t_bins_c - data_mean_bins) ** 2) / N
        # Unlike fitting.mean_arrival_time's tail-only window (short, close
        # to the mean it's centered on), this function's window runs the
        # full pooled-cutoff range from bin 0 -- on real data, up to ~30 ns,
        # hundreds of bins past the mean. The (t - mean)^2 weighting is
        # quadratic, so a single noisy bin far out gets a huge weight; with
        # unclipped (possibly negative) y, that one bin can swing the sum
        # substantially negative -- confirmed on real data (var_t as low as
        # -6654 for a pixel with N=611, no small-N edge case; a first
        # attempt at a lenient magnitude threshold here still let through
        # var_t in the -800 to -5600 range, clearly not float noise). So:
        # ANY negative var_t here means this pixel's variance estimate
        # itself is unreliable, not just imprecise -- reporting std=0 would
        # be actively misleading (implying unusual precision, the opposite
        # of what happened), and the pixel's own tau_mean is suspect for
        # the same reason. Flag both as unreliable (NaN) rather than
        # clamping, so a genuinely bad pixel doesn't look like a good one.
        if var_t < 0:
            continue
        tau_mean[i, j] = tau_mean_i
        tau_mean_std[i, j] = np.sqrt(var_t / N) * tpb_ns
    return dict(tau_mean=tau_mean, tau_mean_std=tau_mean_std, n_photons=n_photons_map, shift=shift)


def plot_irf_diagnostic(irf, signal_sum, shift, g, tpb, out_png, show, n_signal, unit):
    """Signal-only summed data decay (from the masked pixels/super-pixels)
    vs the (aligned, background-subtracted) IRF vs the fitted reconvolution
    model -- a visual check that the IRF genuinely explains the data's
    rising edge and that the fit tracks the measured curve."""
    nb = len(signal_sum)
    t_ns = np.arange(nb) * tpb * 1e9
    irf_aligned = shift_array(irf["irf_norm"], shift) * signal_sum.max() / irf["irf_norm"].max()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.semilogy(t_ns, signal_sum + 1, color="black", lw=1.2,
                label=f"Data (summed over {n_signal} masked {unit})")
    ax.semilogy(t_ns, irf_aligned + 1, color="tab:blue", lw=1.0, alpha=0.7,
                label=f"IRF (background-subtracted, aligned, shift={shift} bins)")
    if g.get("mono"):
        fit_label = f"Reconvolution fit (\u03c4={g['tau']:.2f} ns)"
    else:
        fit_label = f"Reconvolution fit (\u03c4s={g['tau_short']:.2f}, \u03c4l={g['tau_long']:.2f} ns)"
    ax.semilogy(t_ns, g["model"](g["popt"]) + 1, color="tab:red", lw=1.2, ls="--",
                label=fit_label)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Photon counts (+1, log scale)")
    title = (f"IRF check: FWHM {irf['fwhm_bins']*tpb*1e9:.2f} ns, "
             f"{irf['bg_fraction']*100:.0f}% of raw IRF was background (removed), "
             f"source: {irf['source']}")
    if irf.get("echo_fraction", 0) > 0.005:
        title += (f"\n{irf['echo_fraction']*100:.1f}% of the background-subtracted IRF was a "
                  f"disconnected secondary peak (windowed out, kept bins {irf['window'][0]}-{irf['window'][1]})")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"  saved {out_png}")
    if show:
        plt.show(block=False); plt.pause(0.1)
