"""
Shared matplotlib figures: intensity/mask overview panels and lifetime-map
+ distribution result panels. Used by every analysis plugin so results look
consistent regardless of which plugin produced them.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .config import DISPLAY_ASPECT, TAU_LONG_VLIM


def _auto_vlim(finite_values, pad_frac=0.05):
    """Robust auto display range for a lifetime map/histogram: the 1st-99th
    percentile of the actual fitted values, with a little padding so the
    median line and outermost bars aren't jammed against the plot edge.
    Percentile-based (not min/max) so a handful of bad/noisy outlier pixels
    -- there are usually a few in any real map -- can't stretch the whole
    color scale/histogram out and wash out the real distribution.

    This replaces a previously-hardcoded display window: the old code fixed
    both the colorbar range and the histogram's bin range to the same
    literal values (from config.TAU_LONG_VLIM, or worse, a second,
    completely disconnected hardcoded histogram range) tuned for one
    specific dye/sample. Any run whose lifetimes fell outside that window
    -- a different dye, or just a different sample -- produced a colorbar
    that clipped to one flat color and a histogram that silently dropped
    almost every pixel (matplotlib's hist() excludes values outside its
    explicit range rather than erroring), which looked like a bug even
    though the fit itself was fine."""
    if finite_values.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(finite_values, [1, 99])
    if hi <= lo:   # degenerate (e.g. only one distinct value) -- fall back to true min/max
        lo, hi = float(finite_values.min()), float(finite_values.max())
    if hi <= lo:   # still degenerate (a single finite pixel) -- open up a small fixed window
        lo, hi = lo - 0.5, hi + 0.5
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def plot_overview(img, out_png, show, unit="pixel", title_suffix="", close_after=False):
    """Intensity (log) + per-pixel (or per-super-pixel, if binned) photon count
    map (linear, hot-pixel-clipped). `unit` and `title_suffix` let this same
    function be reused for the post-binning overview window.

    close_after: pass True when calling this repeatedly in a loop (e.g. once
    per group/window) -- otherwise figures pile up and matplotlib eventually
    warns "more than 20 figures have been opened". Leave False (default) for
    a one-off overview shown right before an interactive prompt (e.g.
    choosing a threshold), where the window should stay visible on screen
    while the person reads it and answers in the terminal."""
    nz = img[img > 0]
    vmax = np.percentile(nz, 99) if nz.size else 1
    fig, ax = plt.subplots(1, 2, figsize=(9, 7))
    im0 = ax[0].imshow(np.log10(np.maximum(img, 0.5)), cmap="inferno",
                       aspect=DISPLAY_ASPECT, origin="upper")
    ax[0].set_title(f"Intensity (log\u2081\u2080 photons){title_suffix}")
    im1 = ax[1].imshow(img, cmap="inferno", vmin=0, vmax=vmax,
                       aspect=DISPLAY_ASPECT, origin="upper")
    ax[1].set_title(f"Photons per {unit} (linear)\nclipped at 99th pct = {int(vmax)}")
    for a, im, lbl in ((ax[0], im0, "log\u2081\u2080 N"), (ax[1], im1, "photons")):
        a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.02, label=lbl)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"  saved {out_png}")
    if show:
        plt.show(block=False); plt.pause(0.1)
    if close_after:
        plt.close(fig)


def plot_overview_and_mask(img_orig, img_masked, mask, removed, threshold, min_neighbors,
                           bin_size, out_png, show, close_after=True):
    """Two-panel figure matching exactly what the GUI shows: the raw
    (unbinned) whole-image photon count on the left, and the mask preview
    (post-binning, thresholded, with the neighbour filter applied) on the
    right. Replaces the older separate overview/binned_overview files with
    one file that also shows the actual mask used for the fit."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 6.5))

    vmax = np.percentile(img_orig[img_orig > 0], 99) if (img_orig > 0).any() else 1
    im0 = ax[0].imshow(img_orig, cmap="inferno", vmin=0, vmax=vmax,
                       aspect=DISPLAY_ASPECT, origin="upper")
    ax[0].set_title("Photons per pixel")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.02, label="photons")

    im1 = ax[1].imshow(np.log10(np.maximum(img_masked, 0.5)), cmap="gray",
                       aspect=DISPLAY_ASPECT, origin="upper")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.02, label="log\u2081\u2080 photons")
    ax[1].contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    ys, xs = np.where(removed)
    if xs.size:
        ax[1].scatter(xs, ys, s=10, c="red", marker="x", linewidths=0.6)
    unit = "super-pixel" if bin_size and bin_size > 1 else "pixel"
    mask_desc = "external mask" if threshold is None else f"threshold={threshold}, bin={bin_size}"
    ax[1].set_title(f"Mask preview: {int(mask.sum())} {unit}s kept ({mask_desc})", fontsize=10)

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"  saved {out_png}")
    if show:
        plt.show(block=False); plt.pause(0.1)
    if close_after:
        plt.close(fig)


def save_lifetime_map(tau_map, out_prefix, label="tau_long", symbol="\u03c4_long", vlim=None):
    """Save the lifetime map as three separate files, each useful for a
    different purpose:
      - PNG: a standalone, publication-ready image (colorbar, no other panels)
      - TIFF: 32-bit float, single channel -- the actual numeric values,
        openable in ImageJ/Fiji or any scientific image tool, NOT just a
        colored picture. NaN marks pixels that weren't fitted.
      - CSV: the same numeric grid as plain text, for anything that just
        wants to read the numbers directly (Excel, pandas, a quick script).

    vlim: (min, max) colorbar range in ns. Pass None (default) to
    auto-scale from this map's own fitted values (see _auto_vlim) -- the
    right choice for a single file. Pass an explicit range if you want
    several files' maps to share one color scale for a fair side-by-side
    comparison (e.g. config.TAU_LONG_VLIM, or the same bounds used to fit).
    Returns the three paths written."""
    png_path = f"{out_prefix}.png"
    tif_path = f"{out_prefix}.tif"
    csv_path = f"{out_prefix}.csv"

    finite = tau_map[np.isfinite(tau_map)]
    if vlim is None:
        vlim = _auto_vlim(finite)

    # -- PNG: standalone map, own colorbar, own figure --
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    cmap = plt.get_cmap("turbo").copy(); cmap.set_bad("#1a1a1a")
    im = ax.imshow(tau_map, vmin=vlim[0], vmax=vlim[1],
                   cmap=cmap, aspect=DISPLAY_ASPECT, origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{symbol} (ns)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=f"{symbol} (ns)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # -- TIFF: raw float values, no color mapping --
    Image.fromarray(tau_map.astype(np.float32), mode="F").save(tif_path)

    # -- CSV: same raw float values as plain text (NaN round-trips via "nan") --
    np.savetxt(csv_path, tau_map, delimiter=",", fmt="%.4f")

    print(f"  saved {png_path}, {tif_path}, {csv_path}")
    return png_path, tif_path, csv_path


def plot_results(img, mask, removed, tau_map, threshold, min_neighbors, out_png, show,
                 bin_size=1, label="tau_long", symbol="\u03c4_long", close_after=True, vlim=None):
    """Lifetime map + lifetime distribution (the mask itself is shown
    separately, in plot_overview_and_mask -- this figure focuses on the fit
    result). `label`/`symbol` let this same function serve both the
    biexponential tau_long map (default) and a mono-exponential single-tau map.

    vlim: (min, max) in ns, used for BOTH the map's colorbar and the
    histogram's bin range, so the two panels always agree with each other
    and with what the fit actually produced. Pass None (default) to
    auto-scale from this run's own fitted values (see _auto_vlim) -- this
    is what makes the plot correct for any dye/sample without needing a
    hand-tuned constant. Pass an explicit range only if you specifically
    want several files' plots to share one color/axis scale.

    img, mask, removed, threshold, min_neighbors, bin_size are accepted for
    call-site compatibility (several scripts already pass them) but are no
    longer rendered here."""
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 6.5))

    tlv = tau_map[np.isfinite(tau_map)]
    if vlim is None:
        vlim = _auto_vlim(tlv)

    map_title = "LONG lifetime" if label == "tau_long" else "Lifetime"
    cmap = plt.get_cmap("turbo").copy(); cmap.set_bad("#1a1a1a")
    im0 = ax[0].imshow(tau_map, vmin=vlim[0], vmax=vlim[1],
                       cmap=cmap, aspect=DISPLAY_ASPECT, origin="upper")
    ax[0].set_title(f"{map_title}  {symbol}  (per pixel)")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.02, label=f"{symbol} (ns)")

    ax[1].hist(tlv, bins=45, range=vlim, color="#c44")
    if tlv.size:
        ax[1].axvline(np.median(tlv), color="k", ls="--",
                      label=f"median {np.median(tlv):.2f} ns")
        ax[1].legend(fontsize=9)
    ax[1].set_title(f"{symbol} distribution ({tlv.size} pixels)")
    ax[1].set_xlabel(f"{symbol} (ns)"); ax[1].set_ylabel("pixels")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"  saved {out_png}")
    if show:
        plt.show()
    if close_after:
        plt.close(fig)

def plot_fit_quality(tau_std_map, n_photons_map, mask, out_png, show,
                     symbol="\u03c4_long", converged_map=None, close_after=True):
    """Fit-quality diagnostic for a per-pixel lifetime map -- how much to
    trust each pixel's fitted lifetime, and why. Three panels:
      1. spatial map of the lifetime's 1-sigma uncertainty (tau_std)
      2. histogram of that uncertainty across every fitted pixel
      3. uncertainty vs. photon count (log-x): the noise-floor curve. This
         is the one that actually answers "is setting A or B better" --
         it shows directly where (in photon count) the uncertainty starts
         blowing up, so you can judge whether a lower threshold or coarser
         binning trades away more precision than it's worth, using the
         SAME acquisition rather than guessing from the map alone.

    Restricted to `mask` and, if given, pixels where converged_map is True
    -- unmasked or non-converged pixels have no meaningful uncertainty to
    show. Deliberately does NOT auto-exclude high-std pixels from the
    lifetime results themselves anywhere else -- that's a judgment call
    left to you, informed by this plot, not a hardcoded cutoff."""
    valid = mask & np.isfinite(tau_std_map)
    if converged_map is not None:
        valid = valid & converged_map
    std_vals = tau_std_map[valid]
    n_vals = n_photons_map[valid] if n_photons_map is not None else np.array([])

    fig, ax = plt.subplots(1, 3, figsize=(14, 5))

    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title(f"{symbol} uncertainty (per pixel)")
    if std_vals.size:
        vlim = _auto_vlim(std_vals)
        std_display = np.where(valid, tau_std_map, np.nan)
        cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#1a1a1a")
        im = ax[0].imshow(std_display, vmin=vlim[0], vmax=vlim[1], cmap=cmap,
                          aspect=DISPLAY_ASPECT, origin="upper")
        plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02, label=f"{symbol} std (ns)")
    else:
        ax[0].text(0.5, 0.5, "no usable\nuncertainty values", ha="center", va="center",
                  transform=ax[0].transAxes, color="#888")

    ax[1].set_title(f"{symbol} std distribution ({std_vals.size} pixels)")
    ax[1].set_xlabel(f"{symbol} std (ns)"); ax[1].set_ylabel("pixels")
    if std_vals.size:
        ax[1].hist(std_vals, bins=45, range=vlim, color="#4477aa")
        ax[1].axvline(np.median(std_vals), color="k", ls="--",
                      label=f"median {np.median(std_vals):.2f} ns")
        ax[1].legend(fontsize=9)

    ax[2].set_title("uncertainty vs. photon count (noise floor)")
    ax[2].set_xlabel("photons used in fit")
    ax[2].set_ylabel(f"{symbol} std (ns)")
    ax[2].grid(alpha=0.3)
    if std_vals.size and n_vals.size:
        ax[2].scatter(n_vals, std_vals, s=6, alpha=0.4, color="#c44")
        ax[2].set_xscale("log")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"  saved {out_png}")
    if show:
        plt.show()
    if close_after:
        plt.close(fig)
