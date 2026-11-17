# FlimArc — Claude Code Guide

Fluorescence Lifetime Imaging Microscopy (FLIM) analysis pipeline for
Flimera/EzTime camera data. Desktop Tkinter GUI + CLI toolkit + JSON batch runner.

---

## Repository layout

```
flim_gui.py          Entry point — creates FlimMasterApp and starts the event loop
batch_run.py         JSON-driven headless batch runner (uses Agg backend, no windows)
launch_ui.bat        Windows launcher shortcut

analyses/
  _registry.py       Auto-discovers plugins; exposes list_plugins() and get(id)
  normal.py          Per-pixel lifetime map (tail / IRF / mean) — the reference plugin
  phasor.py          Per-pixel phasor (G, S) — model-free, no per-pixel fitting
  <others>           Whole-image, phase-resolved, timelapse, … (follow same contract)

flim_core/
  config.py          Shared numeric constants and defaults
  io.py              HDF5 loader (histogram + raw photon-stream)
  masking.py         Threshold + min-neighbours mask
  fitting.py         Tail fitting, IRF reconvolution, mean arrival time
  irf.py             IRF loading, shifting, normalisation
  plotting.py        Result maps, overview PNGs, fit-quality figures
  <other modules>    Additional shared utilities

gui/
  mainwindow.py      Plugin-agnostic Tkinter app (FlimMasterApp)
```

---

## Running the app

```bash
pip install -r requirements.txt       # numpy scipy h5py matplotlib Pillow
python flim_gui.py                    # GUI
python -m analyses.normal data.H5 --threshold 400   # CLI, one plugin
python batch_run.py jobs.json         # headless batch
```

---

## Plugin contract — **never break this**

Every file in `analyses/` that does **not** start with `_` is treated as a plugin
candidate. `_registry.py` keeps only files that expose all of the following.

### Required module-level attributes

| Name | Type | Meaning |
|---|---|---|
| `NAME` | `str` | Display name shown in the GUI. Must be unique across plugins. |
| `DESCRIPTION` | `str` | One-line description shown in the GUI. |
| `REQUIRES_RAW` | `bool` | `True` → plugin is disabled unless the file is a raw photon stream. |
| `IRF_CAPABLE` | `bool` | `True` → plugin stays enabled when an IRF fit method is selected. |
| `EXTRA_ARGS` | `list[str]` | Plugin-specific field names the GUI will render as input widgets. |

### Optional `GUI_OPTIONS` (dict)

Tells the GUI which common settings apply, so unused sections are hidden.
Any key left out keeps its default (`analyses/_registry.py`,
`GUI_OPTION_DEFAULTS`); plugins without `GUI_OPTIONS` show everything.

| Key | Default | Meaning |
|---|---|---|
| `fit_methods` | `None` (all) | `{token: label}` offered in the fit-method dropdown |
| `fit_method_title` | `"Fit method"` | Title of that section |
| `tail_offset_with_irf` | `False` | Keep "Tail start offset" visible in IRF modes |
| `uses_model` | `True` | `False` hides Model / preset / bounds (and the model tag in the folder name) |
| `uses_lifetime_range` | `True` | `False` hides Lifetime display range |
| `uses_bin_pattern` | `True` | `False` hides 8-bin pattern correction |
| `outputs` | all four `save_*` | Which output toggles the plugin honours |

The GUI must stay plugin-agnostic: hide sections from this metadata, never
by checking a plugin id.

**Naming convention:** a module whose name ends in `_wholeimage` is treated as
a whole-image plugin by the GUI (disables moving-average smoothing, affects
output folder name). Use this suffix for any plugin that fits the full FOV
rather than individual pixels.

### `run(args)`

Called by the GUI in a **background thread** with the working directory already
set to the output folder and stdout redirected to the log panel.

- `args` is a plain attribute-holder (not argparse.Namespace from the CLI, but
  compatible). Always read attributes with `getattr(args, name, default)` —
  not every attribute is guaranteed present.
- **One file per call:** the input is `args.h5file` (singular). A CLI that
  accepts several files must loop in `main()` and call `run()` once per file.
- Let exceptions propagate — don't catch-and-print inside `run()`, or the GUI
  never shows the error.
- Call `args.cancel_check()` periodically in tight loops and raise
  `AnalysisCancelled` (from `flim_core.fitting`) if it returns `True`.
  The GUI treats `AnalysisCancelled` as a clean user-initiated stop.
  Any other exception is shown as an error.
- Return value is ignored.

### `preview_slice(loaded_data, settings)`

Called on the **main thread** for the mask preview — must be fast, must not fit.

- `loaded_data`: dict returned by `flim_core.io.load_flim_data()`.
- `settings`: dict with an `"extra"` key containing the current values of the
  plugin's `EXTRA_ARGS` fields.
- Must return `(img, cube, description)`:
  - `img`: `[H, W]` photon-count image
  - `cube`: `[H, W, nb]` decay histogram cube
  - `description`: short string shown in the preview label

### `args` attributes supplied by the GUI

The GUI always passes these; plugins read what they need and ignore the rest.

```
h5file             Path to the input HDF5 file
prefix             Always None (GUI sets output names itself)
no_show            Always True  (never open interactive plot windows)
roi_prompt         Always False

bin                Spatial binning factor
moving_average     Moving-average kernel size (disabled for _wholeimage plugins)
time_bin           Temporal rebinning factor
roi                ROI coords or None

threshold          Minimum photon count to include a pixel
min_neighbors      Minimum neighbouring pixels above threshold
external_mask      Path to .npy mask or None

mono               bool — fit monoexponential instead of biexponential
fix_tau_short      bool — pin tau_short to tau_short_value
tau_short_value    float
tail_start_offset  int — bins after the peak where the tail starts
fit_method         "tail" | "irf" | "mean" | "mean_irf"
tau_short_bounds   (lo, hi) in ns, or None → config default
tau_long_bounds    (lo, hi) in ns, or None → config default
tau_mono_bounds    (lo, hi) in ns, or None → config default

irf                Path to IRF HDF5 file, or None
no_shift           bool — skip IRF time-shift search
irf_full           bool — use full IRF reconvolution (vs tail-only)
irf_window_end_ns  float
pixel_offset       int
bin_pattern        list[int] or None

lifetime_vlim      (vmin, vmax) or None
save_overview      bool
save_results       bool
save_quality_plots bool
save_irf_check     bool

cancel_check       callable() → bool  ← MUST be called in long loops
```

Plus one attribute per name in `EXTRA_ARGS`. The GUI renders known names
(`step`, `period`, `groups`, `rolling_window`) with predefined types and
defaults. Unknown names get a plain text entry; the value arrives as a `str`
(`""` when left blank), so the plugin must parse it and treat `""` as "use the
default". `batch_run.py` **requires every** `EXTRA_ARGS` key in each job
(`""` or `null` is accepted).

Note the external-mask attribute is `external_mask` (not `mask`), and the
`flim_core.masking.load_mask(path)` helper takes only the path.

---

## `flim_core` public API

### `flim_core.config`

```python
DEFAULT_THRESHOLD   = 400
TAIL_START_OFFSET   = 50          # bins after peak
TAU_SHORT_BOUNDS    = (0.5, 2.0)  # ns
TAU_LONG_BOUNDS     = (2.8, 8.0)  # ns
TAU_MONO_BOUNDS     = (0.5, 8.0)  # ns
TAU_LONG_VLIM       = (2.8, 3.8)  # optional fixed colour range
MIN_NEIGHBORS       = 2
BIN_SIZE            = 1
DISPLAY_ASPECT      = 2
IRF_SHIFT_SEARCH    = 80          # max bins searched each side when aligning IRF
```

### `flim_core.io`

```python
load_flim_data(path) -> dict
  # Returns: img [H,W], cube [H,W,nb], tpb, W, H, B0, B1, n_photons, source
  # Optional: lifetime_eztime, fwhm_eztime
  # Arrays rotated 90° CCW after loading; W/H updated accordingly
  # Handles: pre-computed histogram HDF5, .ds3d project files, raw photon streams

photon_count_map(s) -> np.ndarray [H,W]   # pass-through to s["img"]
build_decay_cube(s)  -> np.ndarray [H,W,nb] # pass-through to s["cube"]
```

### `flim_core.fitting`

```python
# Shared exception — raise this from run() on cancel_check() == True
class AnalysisCancelled(Exception): ...

# Whole-image tail fit
global_fit(summed, s, s_idx, tn,
           mono=False,
           tau_short_bounds=None, tau_long_bounds=None, tau_mono_bounds=None,
           estimator="poisson",   # "poisson" | "leastsq"
           fit_background=True) -> dict
  # Mono returns:   tau, tau_std, popt, model, yv, mono, estimator, fit_background
  # Bi   returns:   tau_short, tau_long, tau_short_std, tau_long_std, + above

# Per-pixel monoexponential tail fit
fit_pixels_mono(cube, s_idx, tn, mask,
                tau_mono_bounds=None,
                cancel_check=None,
                fit_background=True) -> dict
  # Maps [H,W]: tau, tau_std, n_photons, converged, nll

# Per-pixel biexponential tail fit
fit_pixels(cube, s_idx, tn, mask,
           tau_short_fixed=None,   # pin short lifetime; tau_short_std → NaN
           tau_short_bounds=None, tau_long_bounds=None,
           cancel_check=None,
           fit_background=True) -> dict
  # Maps [H,W]: tau_long, tau_short, tau_long_std, tau_short_std, n_photons, converged, nll

# Non-iterative mean arrival time (vectorised, fast)
mean_arrival_time(cube, s_idx, tn, mask,
                  subtract_background=True,
                  correct_bias=True) -> dict
  # Maps [H,W]: tau_mean, tau_mean_std, n_photons
  # For biexponential decays this is an intensity-weighted average, NOT a component lifetime

# Tail start helper
_tail_axis(s, cube, offset=None) -> (summed, peak_bin, s_idx, tn)
  # Finds decay peak, sets tail start TAIL_START_OFFSET bins after it
  # Returns summed histogram and index arrays for fitting
```

---

## Scientific accuracy requirements

These are non-negotiable — do not simplify:

- **Tail fitting** starts after the IRF-dominated rise; the offset is in bins
  (`TAIL_START_OFFSET`). Never fit the full decay curve unless IRF reconvolution
  is explicitly selected (`fit_method="irf"`).
- **IRF reconvolution** requires an IRF file. Never approximate it as a Gaussian.
- **Mean arrival time** (`fit_method="mean"`) is the photon-weighted first moment,
  not a lifetime component. For biexponential decays it is an intensity-weighted
  average of both components.
- **Poisson estimator** (default) minimises Poisson deviance, not chi-squared.
  The `"leastsq"` branch uses weighted curve_fit and is a fallback only.
- **`tau_short` is always the shorter lifetime.** The fitter enforces this;
  never swap or relabel the two components.
- **Uncertainties** are propagated from the Hessian of the Poisson likelihood,
  not from the covariance of a least-squares fit (unless `estimator="leastsq"`).
- **Never clip background-subtracted counts or levels at 0** — clipping keeps
  only the positive half of the noise and biases moments and phasors.
- **IRF/data timing:** IRF files and data have a real timing offset. Measure it
  with a reconvolution fit (`global_reconv_fit`, free `shift`); never align on
  the decay *peak* (`estimate_shift`), which lags time zero by ~0.3 ns. The
  fitted `shift` includes a +0.5-bin model convention
  (`flim_core.irf._BIN_CONVENTION_SHIFT`); subtract it for the physical offset.

---

## Phasor plugin (`analyses/phasor.py`)

Per-pixel DFT of each histogram at the laser repetition frequency:
`G + iS = Σ (n_k − B) e^{iωt_k} / Σ (n_k − B)`. Fully vectorised, with no
per-pixel optimiser. Uses only two small pooled fits: one for the background
shapes and one for the IRF offset.

`EXTRA_ARGS` (str/int/bool accepted; blank = default):

| Name | Default | Meaning |
|---|---|---|
| `harmonic` | 1 | DFT harmonic (GUI dropdown) |
| `rep_rate_mhz` | none | Laser rep. rate; if blank, period is **assumed** = histogram window (logged) |
| `subtract_background` | yes | Flat background subtraction (GUI checkbox) |
| `decay_model` | `bi` | `bi` (two lifetimes: tie-line, `f_long`) or `single` (single-lifetime curve: `tau_single`, `off_curve`) |
| `tau_ref` | none | `single` only: reference lifetime (ns), marked on the plot |
| `tau_ref1`, `tau_ref2` | none | `bi` only: reference lifetimes (ns) → tie-line + `f_long` |
| `per_pixel_irf` | no | IRF mode only: divide each pixel by its own IRF phasor (GUI checkbox) |
| `hot_pixels` | `off` | `exclude` / `subtract` hot (high dark-count) pixels -- see below |
| `show_semicircle` | no | Plot only: also draw the universal semicircle next to the single-lifetime curve (always drawn when there is no curve, i.e. two-lifetime mode) |

In the GUI, `tau_ref` / `tau_ref1` / `tau_ref2` / `show_semicircle` show only for their decay model
(`"show_if"` in `EXTRA_ARG_SPECS`; hidden fields are also left out of
`metadata.txt`).

The time window follows the common `fit_method` (no separate setting):
`"tail"` → tail window; `"irf"` → full window + IRF correction (needs `irf`).
Other fit methods raise. The old `window_mode` arg/`--window-mode` flag is
gone; a `window_mode` attribute raises with a pointer to `fit_method`.
`GUI_OPTIONS` hides Model, preset, bounds and Lifetime display range, and
keeps the tail offset visible in IRF mode (the background fit uses it).

Science conventions — keep these when editing:

- **f_long vs f_long_window.** The tie-line projection gives the long
  component's share of photons *inside the window* (`f_long_window`). For a
  tail window that overestimates; `f_long` converts it to the whole-decay
  intensity fraction using each reference lifetime's in-window share.
- **Reference phasors** are computed numerically through the same window,
  from a *periodic* decay (repeats every laser period) starting at the decay
  peak — not from the analytic semicircle. Tail mode: point samples at left
  bin edges (exact, since a bin integral of an exponential is proportional
  to its left-edge value). IRF-corrected mode: the decay starts at `t0 = 0`
  and each bin is integrated exactly *with its phase* (`continuous=True` in
  `_mono_phasors`) -- dividing by the IRF phasor removes the binning, so the
  corrected phasor is the continuous decay's; point samples lag it by half
  a bin (was +0.025 ns on tau at 50 ps bins, +0.11 ns at 400 ps).
- **In-window photon share** `w_window` uses exact bin integrals
  (`1 - e^{-tpb/tau}`), not value x bin width (~10 % high for 1.4 ns at
  300 ps bins, biasing `f_long`).
- **Background** can't be read from the last bins: at 80 MHz a 4.3 ns
  component is still decaying there (~60 % of those counts are signal, varying
  per pixel). Instead each pixel's tail is solved linearly as
  `a1·p1 + a2·p2 + B` with fixed lifetimes (refs, else a pooled `global_fit`);
  only `B` is used. Skipped with a warning if ill-conditioned.
- **IRF correction** = `z_data / (z_irf · e^{iω·offset})`, with `offset` the
  physical offset from one pooled `global_reconv_fit`, **fitted at native
  time resolution** (`irf_offset_binned`) and divided by `time_bin`. The
  reconvolution model is inaccurate when bins approach the IRF width (on
  synthetic data: -199 ps at 400 ps bins, true 0), so never fit the offset
  on time-binned data. Check: IRF-mode results must not depend on `time_bin`. Valid only for
  the full window (`fit_method="irf"`), exact only when the histogram
  window spans one full laser period (warns otherwise). If the offset can't be
  measured, the correction is skipped rather than applied with a wrong offset.
- **Per-pixel IRF** (`per_pixel_irf`): each pixel's IRF comes from the same
  pixel of the IRF file, put through the data's own pixel-offset / ROI /
  binning / time-binning steps (`irf_cube_like_data`), so the pixel's TDC
  offset cancels with or without a calibration map. The pooled offset fit
  already contains the masked pixels' mean IRF timing vs the whole sensor
  (+15-20 ps here); `reference_per_pixel_irf` removes it from the per-pixel
  IRFs -- don't drop that step (it shifted tau_phase by -0.03 ns). Timelapse:
  the pooled trace divides by the signal-weighted mean of the pixels' IRF
  phasors (exact for a sum of pixels). Check: per-pixel and pooled IRF with a
  correct map give the same median (within ~0.01 ns).
- In tail mode `tau_phase` / `tau_mod` are **not** lifetimes (semicircle
  formulas don't hold for a truncated window); use `f_long`, `tau_single` or G/S.
- **Single-lifetime mode** projects each pixel onto the single-lifetime curve
  (`single_lifetime_curve`: periodic monoexponentials through the *same*
  window and `t0` as the reference phasors; the semicircle only in IRF mode).
  `tau_single` = lifetime of the nearest curve point (valid in tail mode too);
  `off_curve` = signed distance, + = inner side where mixtures and
  zero-lifetime scatter fall. NaN where the nearest point is a curve end.
- **Single-mode background is one image-wide level** (constant of a pooled
  biexponential + constant fit / number of pixels), and the log says so.
  Do not "improve" it to per-pixel: a fixed decay shape biases B wherever the
  lifetime differs from the pooled one; tail bins can't separate per-pixel B
  from per-pixel tau; and iterating B with each pixel's `tau_single` runs
  away on non-single-exponential pixels (tested on Flipper-TR: B doubled,
  points left the curve). Two-lifetime mode keeps its per-pixel tail fit.
  Exception, opt-in: `hot_pixels="subtract"` (below).
- **Hot pixels** (`flim_core/hot_pixels.py`): ~11 % of the sensor is switched
  off by the camera (0 photons, same pixels in every file); of the rest
  ~6-11 % still have several-fold dark counts (fixed sensor property, 90-98 %
  the same pixels across files). Their extra flat counts streak the cloud
  toward the flat-background phasor. Detected from the **pre-rise bins**
  (lifetime-independent, unlike the tail): pre-rise level vs
  `b0 + k * photons` (k = previous-pulse remnant), 3 sigma one-sided Poisson,
  on the raw sensor BEFORE the pixel-offset shift (its circular FFT shift
  wraps late bins into the pre-rise stretch). `exclude` zeroes them before
  threshold/binning; `subtract` (single mode) adds each one's excess to the
  image-wide level, which is lowered by their share
  (`add_hot_pixel_background`), and drops hot pixels whose *signal* photons
  fail the threshold (`drop_weak_hot_pixels`) -- without that, pixels that
  passed on dark counts alone gave NaN / wild values. Fluorescein 40 MHz:
  tau_single IQR 0.14 -> 0.12 ns, off_curve sd 0.013 -> 0.006, median
  3.955 -> 4.005 ns (the image-wide level had absorbed the hot pixels'
  counts and over-subtracted the normal ones).

Outputs: `{prefix}_maps.npz` (G, S, modulation, tau_phase, tau_mod, n_photons,
mask, decay_model, t0_ns, f_long, f_long_window, background_per_bin, irf_shift;
single mode: tau_single, off_curve, curve_tau_ns/curve_G/curve_S, …),
`_overview.png`, `_phasor_scatter.png`, `_phasor_maps.png`.

Scatter plots draw the pixel cloud with `_draw_pixel_cloud` (bin size set by
the axes, not the cloud -- a fixed 100x100 grid over a tight single-lifetime
cloud made it invisible) and add a zoom inset (`_add_zoom_inset`) when the
cloud is small. `phasor_timelapse` reuses the same helper.

---

## Phasor timelapse (`analyses/phasor_timelapse.py`)

Raw files only. Same windows as `timelapse.py` (`step`, `rolling_window`) plus
all phasor `EXTRA_ARGS`; reuses the maths in `analyses/phasor.py` (import it,
don't copy it). Per window: pixel maps (`f_long`, `tau_single`/`off_curve`, or
`tau_phase`) and one pooled whole-mask phasor.

Measured **once on the whole acquisition** and reused for every window: the
fixed pixel mask (threshold = photons per pixel per output window, averaged),
time axes, tail start, `t0`, reference phasors / single-lifetime curve, the
IRF offset, and the background (scaled to each window by its duration --
assumes a constant background rate). Pixel offset is applied to photons
(integer bins), so the 8-bin pattern correction is unsupported and hidden.
Check: one window spanning the whole run reproduces `phasor.py` (G/S within
~2e-4).

Outputs: `_maps.npz` (stacks `[n_windows, H, W]`, `pooled_*` traces,
`t_start_s`/`t_end_s`/`t_mid_s`, …), `_overview.png`,
`_phasor_trajectory.png`, `_phasor_timecourse.png`, `_phasor_frames.png`.

## Pixel-offset calibration (`flim_core/pixel_offset.py`, `calibration/`)

Per-pixel SPAD timing offsets (~100-180 ps rms). **The map depends on the
laser rep. rate / sync delay**: 20 MHz and 40 MHz/3560 maps correlate at
r = -0.6, and the wrong map makes timing worse (pooled IRF 490 -> 670 ps
instead of -> 190 ps). One map per setting in `calibration/`, named with the
setting, `rep_rate_mhz` / `sync_delay` in the `_meta.json`. Build from a foil
IRF taken at the data's settings:
`python -m flim_core.pixel_offset IRF.ds3d --out calibration/NAME.npy --rep-rate-mhz 40 --sync-delay 3560`.
`load_irf` checks the map against the IRF (`check_against_irf`, warns if it
doesn't remove the IRF's timing spread); phasor plugins also compare the
map's rate with `rep_rate_mhz` (`check_settings`). The GUI pre-selects a map
only when `calibration/` holds exactly one. `apply()` is vectorised
(scipy.fft, ~0.6 s per cube); `integer_bins=True` matches the raw-photon path.

## Output conventions

Every plugin produces at minimum `{prefix}_maps.npz`. The `.npz` is **always**
saved regardless of the output-toggle flags. Optional outputs (controlled by
`args.save_*`):

| Flag | Output |
|---|---|
| `save_overview` | Overview PNG of intensity and lifetime maps |
| `save_results` | Per-component result PNGs |
| `save_quality_plots` | Convergence, NLL, uncertainty maps |
| `save_irf_check` | IRF alignment and fit quality figure |

Relative paths in batch JSON jobs resolve against the job file's directory,
not the current working directory.

---

## Adding a new plugin (checklist)

1. Create `analyses/<name>.py` (no leading underscore).
2. Declare all five metadata attributes (`NAME`, `DESCRIPTION`, `REQUIRES_RAW`,
   `IRF_CAPABLE`, `EXTRA_ARGS`).
3. Implement `run(args)` — use `getattr(args, key, default)` for all args,
   call `args.cancel_check()` in loops, raise `AnalysisCancelled` on stop.
4. Implement `preview_slice(loaded_data, settings)` — fast, no fitting,
   returns `(img, cube, description)`.
5. If the plugin fits the full FOV rather than per pixel, name the module
   `analyses/<name>_wholeimage.py`.
6. No new `requirements.txt` entries without flagging it first.
7. The GUI and registry need **no changes** — restart the app to pick it up.

---

## Dependencies (`requirements.txt`)

```
numpy
scipy
h5py
matplotlib
Pillow
```

Do not add dependencies without explicit discussion.

---

## Common tasks for Claude Code

```bash
# Run one plugin on a file
python -m analyses.normal path/to/data.H5 --threshold 400 --out-dir ./out

# Run the full batch
python batch_run.py path/to/jobs.json

# Launch the GUI
python flim_gui.py

# Run tests (if present)
python -m pytest

# Check the registry discovers your plugin
python -c "from analyses._registry import list_plugins; [print(p['id'], p['module'].NAME) for p in list_plugins()]"
```

---

## Style notes

- Prefer `getattr(args, name, default)` over direct attribute access in plugins.
- Keep physical units in variable names where ambiguous (`_ns`, `_bins`).
- Do not open interactive Matplotlib windows in `run()` — always use
  `plt.savefig()` and `plt.close()`.
- Keep `print()` text ASCII (`tau`, `->`, not `τ`, `→`): the Windows console
  (cp1252) raises `UnicodeEncodeError` on CLI/batch runs. Plot labels may use
  Unicode.
- Module-level docstrings should state the analysis type and fit method.
- Private helpers get a leading underscore; only the plugin contract functions
  and `flim_core` public API are considered stable.
