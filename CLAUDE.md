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

`EXTRA_ARGS` (all GUI text fields; blank = default):

| Name | Default | Meaning |
|---|---|---|
| `harmonic` | 1 | DFT harmonic |
| `window_mode` | `tail` | `tail` (from `_tail_axis` start) or `full` (whole histogram) |
| `tau_ref1`, `tau_ref2` | none | Reference lifetimes (ns) → tie-line + `f_long` |
| `rep_rate_mhz` | none | Laser rep. rate; if blank, period is **assumed** = histogram window (logged) |
| `subtract_background` | yes | Per-pixel flat background subtraction |

IRF correction uses the common `irf` / `no_shift` / `irf_full` args (the GUI
only passes `irf` when an IRF fit method is selected).

Science conventions — keep these when editing:

- **f_long vs f_long_window.** The tie-line projection gives the long
  component's share of photons *inside the window* (`f_long_window`). For a
  tail window that overestimates; `f_long` converts it to the whole-decay
  intensity fraction using each reference lifetime's in-window share.
- **Reference phasors** are computed numerically through the same window,
  from a *periodic* decay (repeats every laser period) starting at the decay
  peak — not from the analytic semicircle. In IRF-corrected mode they start
  at the IRF origin, sampled at bin centres (`t0 = −½ bin`).
- **Background** can't be read from the last bins: at 80 MHz a 4.3 ns
  component is still decaying there (~60 % of those counts are signal, varying
  per pixel). Instead each pixel's tail is solved linearly as
  `a1·p1 + a2·p2 + B` with fixed lifetimes (refs, else a pooled `global_fit`);
  only `B` is used. Skipped with a warning if ill-conditioned.
- **IRF correction** = `z_data / (z_irf · e^{iω·offset})`, with `offset` the
  physical offset from one pooled `global_reconv_fit`. Valid only for
  `window_mode="full"` (skipped for `tail`), exact only when the histogram
  window spans one full laser period (warns otherwise). If the offset can't be
  measured, the correction is skipped rather than applied with a wrong offset.
- In tail mode `tau_phase` / `tau_mod` are **not** lifetimes (semicircle
  formulas don't hold for a truncated window); use `f_long` or G/S.

Outputs: `{prefix}_maps.npz` (G, S, modulation, tau_phase, tau_mod, n_photons,
mask, f_long, f_long_window, background_per_bin, irf_shift, …),
`_overview.png`, `_phasor_scatter.png`, `_phasor_maps.png`.

---

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
