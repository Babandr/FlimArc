"""
File loading: turns a Flimera/EzTime HDF5 file (raw photon stream, flat
export, or native .ds3d project) into an intensity image + decay cube.

Moved out of flimera_flim_analysis.py unchanged -- every analysis plugin now
imports this instead of importing "the normal analysis script".
"""

import numpy as np
import h5py


def load_flim_data(path):
    """Return dict with the per-pixel intensity map 'img' [H,W], the decay cube
    'cube' [H,W,nb] and metadata (tpb, W, H, B0, B1, n_photons, source).

    The image is rotated 90 degrees counter-clockwise from the raw sensor
    readout (native Width=126, Height=192) to match the orientation
    requested for display."""
    with h5py.File(path, "r") as f:
        # A pre-computed-histogram file (flat export OR native project) has a
        # group somewhere in the tree that holds BOTH a "Histogram" dataset
        # (per-pixel decays) and an "Intensity" dataset, with Width/Height/
        # StartBin/EndBin/TimePerBin/NumberOfBins as that group's own
        # attributes. Searching the whole tree for this pattern -- rather
        # than assuming a fixed path like "FLIM/..." -- covers both the flat
        # export layout AND the native project layout (.ds3d and similar),
        # which nests the same pair of datasets under a randomly-generated
        # UUID group name instead. If nothing matches, fall back to treating
        # it as a raw, undecoded photon stream.
        g = _find_histogram_group(f)
        out = _load_exported(g) if g is not None else _load_raw(f)
    return _rotate_ccw(out)


def _rotate_ccw(out):
    """Rotate every spatial [H,W](,...) array in a loaded-data dict 90
    degrees counter-clockwise (np.rot90, k=1) and update W/H to match."""
    out["img"] = np.ascontiguousarray(np.rot90(out["img"], k=1))
    out["cube"] = np.ascontiguousarray(np.rot90(out["cube"], k=1, axes=(0, 1)))
    out["H"], out["W"] = out["img"].shape
    for key in ("lifetime_eztime", "fwhm_eztime"):
        if key in out:
            out[key] = np.ascontiguousarray(np.rot90(out[key], k=1))
    return out


def _find_histogram_group(f):
    """Walk the whole HDF5 tree and return the first group that looks like a
    pre-computed FLIM histogram group (has "Histogram" and "Intensity" child
    datasets, plus the geometry/timing attributes on itself), or None."""
    found = {"group": None}
    required_attrs = ("Width", "Height", "StartBin", "EndBin",
                       "TimePerBin", "NumberOfBins")

    def _check(name, obj):
        if found["group"] is not None:
            return  # already found one; stop bothering with the rest
        if not isinstance(obj, h5py.Group):
            return
        if "Histogram" not in obj or "Intensity" not in obj:
            return
        if not all(k in obj.attrs for k in required_attrs):
            return
        found["group"] = obj

    f.visititems(_check)
    return found["group"]


def _load_exported(g):
    """Pre-computed-histogram group (flat export OR native .ds3d project):
    decay histograms and total counts are already stored per pixel."""
    a = g.attrs
    # Sensor geometry and microtime-bin range/count, all stored as HDF5 attributes.
    # np.ravel(...)[0] guards against attrs that come back as 1-element arrays.
    W  = int(np.ravel(a["Width"])[0]);   H  = int(np.ravel(a["Height"])[0])
    B0 = int(np.ravel(a["StartBin"])[0]); B1 = int(np.ravel(a["EndBin"])[0])
    nb = int(np.ravel(a["NumberOfBins"])[0])
    tpb = float(np.ravel(a["TimePerBin"])[0])   # seconds per microtime bin
    # Flat intensity image (total photons per pixel) and the full decay cube
    # (per-pixel histogram of arrival times), reshaped from EzTime's flattened layout.
    img  = g["Intensity"][:].reshape(H, W).astype(float)
    cube = g["Histogram"][:].reshape(H, W, nb).astype(np.float32)
    out = dict(img=img, cube=cube, tpb=tpb, W=W, H=H, B0=B0, B1=B1,
               n_photons=int(img.sum()), source=f"pre-computed histogram (group: {g.name})")
    # Native project files often also carry EzTime's OWN per-pixel lifetime
    # and IRF-FWHM estimates alongside the raw histograms -- not needed for
    # this script's own fit, but handy to have available for a sanity check
    # against it, so pull them in if present rather than discarding them.
    if "Lifetime" in g and g["Lifetime"].shape == (H * W,):
        out["lifetime_eztime"] = g["Lifetime"][:].reshape(H, W)
    if "FWHM" in g and g["FWHM"].shape == (H * W,):
        out["fwhm_eztime"] = g["FWHM"][:].reshape(H, W)
    return out


def _load_raw(f):
    """Raw photon-stream file: decode 32-bit records into x, y, microtime."""
    # Locate the 1-D uint32 photon-stream dataset by walking the whole file tree,
    # since its exact path can vary; fall back to the conventional name.
    dset = None
    def _find(name, obj):
        nonlocal dset
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1 and obj.dtype == np.uint32:
            dset = obj.name
    f.visititems(_find)
    if dset is None:
        dset = "Streams/Photon"
    d = f[dset]; data = d[:]; a = d.attrs
    tpb = float(np.ravel(a["TimePerBin"])[0])
    W = int(np.ravel(a["CameraWidth"])[0]); H = int(np.ravel(a["CameraHeight"])[0])
    # Some files don't store an explicit microtime bin range; detect it later
    # from the data itself in that case.
    try:
        B0 = int(np.ravel(a["BinRangeLowerBounds"])[0])
        B1 = int(np.ravel(a["BinRangeUpperBounds"])[0])
    except KeyError:
        B0 = B1 = None

    # Decode each 32-bit record per the Flimera "Photon streamed" format:
    # bits 28-31 = record type (0x2 = photon, 0xF = marker/other -> discarded).
    top = (data >> 28) & 0xF
    ph = data[top == 0x2]                    # photon records only
    #x  = (ph & 0xFF).astype(np.int32) - 1    # bits 0-7:  column (file is 1-indexed -> shift to 0-indexed) was before not sure
    x  = (ph & 0xFF).astype(np.int32)
    y  = ((ph >> 8) & 0xFF).astype(np.int32) # bits 8-15: row
    micro = (((ph >> 24) & 0xF) << 8) | ((ph >> 16) & 0xFF)  # bits 16-27: 12-bit microtime bin

    if B0 is None:
        B0, B1 = int(micro.min()), int(micro.max())

    # Drop any out-of-bounds pixel coordinates or out-of-range microtime bins
    # (corrupt/edge records) before building the images.
    keep = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (micro >= B0) & (micro <= B1)
    x, y, micro = x[keep], y[keep], micro[keep]
    nb = B1 - B0 + 1

    # Photon-count image: histogram photons by flattened pixel index.
    img = np.bincount(y * W + x, minlength=H * W).reshape(H, W).astype(float)
    # Per-pixel decay cube: histogram photons by (pixel, microtime bin) jointly,
    # using a single flattened index so np.bincount can build the whole cube in one pass.
    flat = (y * W + x) * nb + (micro - B0)
    cube = np.bincount(flat, minlength=H * W * nb).reshape(H, W, nb).astype(np.float32)
    return dict(img=img, cube=cube, tpb=tpb, W=W, H=H, B0=B0, B1=B1,
                n_photons=int(keep.sum()), source="raw photon stream")


def photon_count_map(s):
    """Per-pixel photon count image [H, W]. Thin accessor kept as its own
    function so the main() pipeline reads as a clear sequence of named steps."""
    return s["img"]


def build_decay_cube(s):
    """Per-pixel decay histograms [H, W, nb]. Both loaders already build this
    cube, so this is just a named pass-through for readability in main()."""
    return s["cube"]
