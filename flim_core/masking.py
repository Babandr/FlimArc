"""
Spatial filtering: photon-count thresholding, isolated-pixel removal,
spatial binning, and ROI cropping. Applies equally regardless of which
analysis plugin is running.
"""

import numpy as np
from scipy.ndimage import convolve, uniform_filter


def clean_mask(mask, min_neighbors):
    """Spatial filter: keep a selected pixel only if at least `min_neighbors` of its
    8 neighbours are also selected. Removes isolated background pixels/specks that
    pass the photon threshold but are surrounded by eliminated pixels."""
    if min_neighbors <= 0:
        return mask.copy()                 # filter disabled: keep the raw threshold mask as-is
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0                       # count neighbours only, not the pixel itself
    # For every pixel, count how many of its 8 neighbours are also selected
    # (convolution with the 3x3 ring kernel above). cval=0 treats out-of-frame
    # neighbours (at the image edges) as "not selected".
    n_sel = convolve(mask.astype(int), kernel, mode="constant", cval=0)
    # Keep a pixel only if it passed the threshold AND has enough selected neighbours.
    return mask & (n_sel >= min_neighbors)


def report_dead_pixels(img):
    """Flag pixels with exactly 0 photons over the whole acquisition. A handful
    of these scattered across an otherwise-illuminated area is typically a
    camera/sensor defect (dead pixels) rather than "no sample there". Returns
    a boolean mask so callers can visualize or otherwise account for them."""
    dead = img == 0
    n_dead = int(dead.sum())
    pct = 100 * n_dead / img.size
    print(f"  dead/no-signal pixels (0 photons): {n_dead} ({pct:.1f}% of sensor)")
    return dead


def bin_cube(img, cube, bin_size):
    """Spatially bin the intensity image and decay cube by summing bin_size x
    bin_size blocks of pixels. This pools photons from neighbouring pixels
    into one "super-pixel" before fitting, trading spatial resolution for
    photon budget -- and as a side effect, a dead pixel (0 photons) inside a
    block just contributes nothing to that block's sum instead of leaving a
    hole in the map. bin_size=1 returns the inputs unchanged. If H or W isn't
    evenly divisible by bin_size, the sensor is cropped (from the high-index
    edge) to the nearest multiple first."""
    if bin_size <= 1:
        return img, cube
    H, W = img.shape
    Hc = (H // bin_size) * bin_size
    Wc = (W // bin_size) * bin_size
    if Hc != H or Wc != W:
        print(f"  binning: cropping sensor from {H}x{W} to {Hc}x{Wc} "
              f"to fit bin size {bin_size} (dropped edge rows/cols)")
    img_c, cube_c = img[:Hc, :Wc], cube[:Hc, :Wc, :]
    Hb, Wb, nb = Hc // bin_size, Wc // bin_size, cube_c.shape[2]
    # Reshape each spatial axis into (n_blocks, bin_size) and sum over the
    # bin_size axes -- a standard vectorized "block-sum downsampling" trick.
    img_binned = img_c.reshape(Hb, bin_size, Wb, bin_size).sum(axis=(1, 3))
    cube_binned = cube_c.reshape(Hb, bin_size, Wb, bin_size, nb).sum(axis=(1, 3))
    print(f"  binning: {bin_size}x{bin_size} blocks -> {Hb}x{Wb} super-pixels "
          f"(median photons/super-pixel: {int(np.median(img_binned))}, "
          f"was {int(np.median(img_c[img_c>=0]))} before binning)")
    return img_binned, cube_binned


def bin_time_cube(cube, n, tpb):
    """Group every n adjacent MICROTIME bins into one, summing photon counts
    (not averaging -- these feed a Poisson NLL fit downstream, so counts
    have to stay counts). Orthogonal to bin_cube's SPATIAL binning: this
    reduces nb and widens each bin, trading time resolution for less photon
    noise per bin and less sensitivity to any single bin's own nonlinearity
    (e.g. the Flimera SPAD's per-channel response isn't perfectly uniform;
    summing several adjacent channels together averages that out). n=1
    returns the cube and tpb unchanged. If nb isn't evenly divisible by n,
    the far (high-index) end of the cube is cropped to the nearest multiple
    first -- that end is deep into the decay tail, well past where real
    signal lives (see TAIL_START_OFFSET and the IRF's own window_end_ns
    default elsewhere in this codebase), so a few dropped bins there cost
    at most a little background-level signal, never real decay shape.

    Returns (cube_binned, tpb_binned) -- tpb_binned = tpb * n, so every
    downstream ns-based calculation (the time axis itself, TAIL_START_OFFSET,
    IRF_SHIFT_SEARCH bounds, etc.) that already takes tpb as a parameter
    keeps working in real ns without any further change; nothing downstream
    needs to know binning happened at all."""
    if n <= 1:
        return cube, tpb
    H, W, nb = cube.shape
    nbc = (nb // n) * n
    if nbc != nb:
        print(f"  time binning: cropping {nb} microtime bins to {nbc} "
              f"to fit group size {n} (dropped {nb - nbc} bins from the far tail)")
    cube_c = cube[:, :, :nbc]
    nb_binned = nbc // n
    cube_binned = cube_c.reshape(H, W, nb_binned, n).sum(axis=3)
    tpb_binned = tpb * n
    print(f"  time binning: grouping {n} microtime bins -> {nb_binned} bins "
          f"({tpb_binned*1e9:.4f} ns/bin, was {tpb*1e9:.4f} ns/bin)")
    return cube_binned, tpb_binned


def moving_average_cube(img, cube, window):
    """Spatial smoothing by a centered NxN box-SUM at every pixel (not a
    block downsample like bin_cube -- output stays at the full original
    H x W resolution, and neighbouring output pixels share most of their
    source photons rather than being independent). For pixel (x, y), pools
    photons from the window x window square centered on it: window=3 means
    (x-1,y-1) to (x+1,y+1), matching the standard "3x3 moving average"
    meaning. window must be odd (1, 3, 5, ...) so there's a single well-
    defined center pixel; window=1 returns the inputs unchanged.

    Pixels nearer the sensor edge than window//2 have a smaller window
    actually available on the sensor -- those missing neighbours are
    treated as 0 (not wrapped or mirrored), so an edge pixel's value is a
    true sum over however much of its window actually exists on the
    sensor, not a number inflated or distorted by fabricated data.

    IMPORTANT, unlike bin_cube: the resulting "super-pixels" are NOT
    statistically independent of each other -- adjacent output pixels are
    built from overlapping sets of real photons. Fine (good, even) for a
    visually smoother map or for following a spatial trend, but treat any
    per-pixel statistic computed across a moving-averaged image (e.g. a
    histogram of fitted tau across all pixels) with that in mind -- it is
    not a histogram of independent measurements the way a bin_cube'd one
    is.

    Returns (img_smoothed, cube_smoothed), same shape as the inputs."""
    if window <= 1:
        return img, cube
    if window % 2 == 0:
        raise ValueError(f"moving_average_cube window must be odd (got {window}) -- "
                         f"an even window has no single center pixel.")
    # uniform_filter computes a MEAN over the window (zero-padded at the
    # edges, so the mean there is already correctly diluted by the missing
    # neighbours) -- multiplying back by window**2 (or window for the cube's
    # extra time axis, where size=1 leaves that axis untouched) converts
    # that mean into the SUM this function promises.
    img_avg = uniform_filter(img.astype(float), size=window, mode="constant", cval=0.0)
    cube_avg = uniform_filter(cube.astype(float), size=(window, window, 1), mode="constant", cval=0.0)
    img_smoothed = img_avg * (window ** 2)
    cube_smoothed = cube_avg * (window ** 2)
    print(f"  moving average: {window}x{window} centered window -> {img.shape[1]}x{img.shape[0]} "
          f"pixels (unchanged resolution; median photons/pixel: {int(np.median(img_smoothed))}, "
          f"was {int(np.median(img[img>=0]))} before smoothing)")
    return img_smoothed, cube_smoothed


def crop_to_roi(img, cube, roi):
    """Crop the intensity image and decay cube to a rectangular region of
    interest (x0, y0, x1, y1), pixel coordinates, x1/y1 exclusive (so a
    100x100 ROI at the origin is (0, 0, 100, 100)). Cropping this early
    shrinks every downstream array, which is where the real speedup comes
    from: binning, masking, and (especially) the per-pixel fit loop all scale
    with the number of pixels, so working on a smaller sensor region makes
    the whole pipeline faster, not just the plots. Bounds are clamped to the
    image, and an empty result after clamping raises a clear error rather
    than silently producing a 0-pixel image."""
    x0, y0, x1, y1 = roi
    H, W = img.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI ({roi}) is empty after clamping to the image bounds ({W}x{H}).")
    print(f"  ROI crop: ({x0},{y0})-({x1},{y1})  ->  {x1-x0}x{y1-y0} pixels "
          f"(was {W}x{H})")
    return img[y0:y1, x0:x1], cube[y0:y1, x0:x1, :], (x0, y0, x1, y1)


def prompt_roi(W, H):
    """Ask the person for an optional rectangular region of interest, given
    the full sensor's width/height. Returns (x0, y0, x1, y1) or None (use
    the whole image). Meant to be called right after an overview plot of the
    FULL image is shown (non-blocking), so hovering the mouse over that
    still-open window shows pixel coordinates in the toolbar to read off."""
    print(f"\nOptionally crop to a rectangular region of interest, to speed up the "
          f"analysis by skipping pixels you don't care about.")
    print(f"Full sensor is {W}x{H} (x: 0-{W-1}, y: 0-{H-1}). Hover the mouse over "
          f"the image above to read off pixel coordinates.")
    resp = input("ROI as x0,y0,x1,y1 (top-left to bottom-right), or blank for full image: ").strip()
    if not resp:
        return None
    try:
        parts = tuple(int(p.strip()) for p in resp.split(","))
    except ValueError:
        print("  couldn't parse that as 4 numbers; using the full image.")
        return None
    if len(parts) != 4:
        print("  need exactly 4 numbers (x0,y0,x1,y1); using the full image.")
        return None
    return parts


def save_mask(mask, path):
    """Save a boolean pixel-selection mask to disk (.npy) so it can be
    reused across other files/runs -- e.g. one physical sample imaged at
    several laser powers, where each power needs its own photon-count
    threshold but you want the SAME pixels analyzed every time. Saves at
    whatever resolution `mask` currently is (unbinned or binned) -- it's
    the caller's job to make sure later runs use matching binning."""
    mask = np.asarray(mask, dtype=bool)
    np.save(path, mask)


def load_mask(path):
    """Load a previously-saved boolean mask (see save_mask). Raises a
    plain ValueError (not a numpy-internal error) if the file doesn't
    contain a 2-D boolean-like array, so callers can show a clear message
    instead of a confusing stack trace."""
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2:
        raise ValueError(f"'{path}' does not contain a 2-D mask array (got shape {arr.shape}).")
    return arr.astype(bool)
