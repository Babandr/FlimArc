# Flimera FLIM Analysis

A desktop app and set of command-line scripts for FLIM (fluorescence
lifetime imaging) analysis of Flimera/EzTime camera data: load a decay
cube, mask by photon count, and compute a lifetime -- via tail fit, IRF
reconvolution, or fast mean arrival time -- and export the result.

## Layout

```
flim_analysis/
├── flim_core/          # shared library: no analysis-specific logic here
│   ├── config.py         tunable constants (thresholds, bounds, defaults)
│   ├── io.py              load a Flimera HDF5 file (raw / exported / .ds3d)
│   ├── masking.py        threshold, neighbour filter, binning, ROI crop
│   ├── fitting.py        tail-fit lifetime fitting (global + per-pixel)
│   ├── irf.py             IRF reconvolution fitting (an alternative fit
│   │                      method, not an analysis type of its own)
│   └── plotting.py       shared overview / results figures
│
├── analyses/            # one file per analysis plugin
│   ├── _registry.py       auto-discovers every plugin in this folder
│   ├── _common.py         shared by the periodic/timelapse family only
│   ├── normal.py          per-pixel lifetime map, single acquisition
│   ├── normal_wholeimage.py    same, pooled into one whole-image fit
│   ├── periodic.py             phase-resolved (repeating signal)
│   ├── periodic_wholeimage.py       same, pooled per phase group
│   ├── timelapse.py            fixed time windows
│   └── timelapse_wholeimage.py      same, pooled per time window
│
├── gui/
│   └── mainwindow.py     the Tkinter app -- reads the plugin registry,
│                         never imports a plugin by name
│
├── flim_gui.py           entry point: `python flim_gui.py`
├── batch_run.py          run many jobs (files/plugins/settings/output folders)
│                         from one JSON job list -- see "Batch runs" below
├── launch_ui.bat
└── requirements.txt
```

## Running

```bash
pip install -r requirements.txt
python flim_gui.py
```

Each analysis is also runnable standalone from the command line, e.g.:

```bash
python -m analyses.normal data.H5 --threshold 400
python -m analyses.periodic data.H5 --period 1.0 --groups 5
```

Run any of them with `--help` for the full option list.

## Stopping a run

The GUI has a Stop button next to Run Analysis, enabled while a run is in
progress. Python threads can't be safely force-killed, so this is
cooperative: it sets a flag that the running plugin checks -- once per
pixel during the actual fitting loop (`tail`/`irf` fit methods), and once
per window/group for plugins that split the acquisition, so even the fast
`mean` fit method (which has no per-pixel loop to interrupt) still responds
promptly. The run stops at the next checkpoint, not instantly, but that's
at most a moment away for any run worth stopping. Partial output already
written before the stop (e.g. earlier windows in a timelapse) is left on
disk; nothing after the stop point is written.

## Output file toggles

The GUI's Output options panel (below the plugin list) has four independent
checkboxes -- Overview/mask preview, Results plot, Fit-quality diagnostic,
IRF check plot -- so you can skip whichever plot types you don't need for a
given run. The underlying `.npz` data is always saved regardless; these only
control which PNGs get written. On the CLI/in a batch job: `--no-overview-plot`
/ `"save_overview": false`, `--no-results-plot` / `"save_results"`,
`--no-quality-plots` / `"save_quality_plots"`, `--no-irf-check-plot` /
`"save_irf_check"` (all default to saving). Trend/phase-summary plots for the
timelapse/periodic plugins aren't covered by these -- they're always saved.

## Decay viewer

The GUI's "Decay viewer..." button (next to Open File) opens a separate
window for inspecting individual pixels' raw decay curves -- click any
pixel/super-pixel in the intensity image to see its decay on a log scale,
optionally compare several named time windows within the acquisition
(raw photon-stream files only -- disabled for pre-summed histogram/.ds3d
files, which have no per-photon timing left), and export exactly what's
plotted to CSV. It reuses whatever file is already loaded in the main
window -- no separate file dialog, and it closes automatically if you open
a different file. Its own spatial binning is independent of the main
window's Masking settings, so you can inspect at a different granularity
than an actual analysis run. See `gui/decay_viewer.py`.

## External masks

The GUI's "Selection mode" has a third option, "External mask", alongside
Threshold and ROI -- it uses a previously-saved set of pixels directly,
with no threshold or neighbour filtering applied. Useful for analyzing the
exact same pixels across acquisitions that each need a different photon
threshold -- e.g. the same sample imaged at several laser powers.

- **Save a mask**: pick Threshold or ROI as usual, click Apply to build a
  preview, then click "Save mask..." to write it to a `.npy` file.
- **Reuse it**: switch Selection mode to "External mask", click "Load mask
  file...", and pick that file. Every pixel it contains is used as-is;
  the file's shape must match the current image *after* binning/ROI are
  applied, or you'll get a clear shape-mismatch error rather than silently
  wrong results.

On the CLI/in a batch job, the same thing is `--external-mask <path.npy>`
/ `"external_mask": "path.npy"` on any plugin -- it takes priority over
`--threshold`/`--roi` entirely (both are ignored if given alongside it).
For the pixel-wise timelapse/periodic plugins, the same external mask is
applied to every window/group, the same fixed-mask idea the whole-image
plugins already use. See `flim_core/masking.py`'s `save_mask`/`load_mask`
if you want to build or inspect a mask file from a script instead.

## Lifetime bounds and presets

The optimizer's search range for each lifetime (tau_short/tau_long for a
biexponential fit, or a single tau range for mono-exponential) lives in
`flim_core/config.py` as the fallback default, but every fitting function in
`flim_core/fitting.py` and `flim_core/irf.py` takes it as a real parameter
(`tau_short_bounds`, `tau_long_bounds`, `tau_mono_bounds`) -- so it can be
overridden per run instead of only by editing that file.

- **CLI**: `--tau-short-bounds MIN MAX`, `--tau-long-bounds MIN MAX`,
  `--tau-mono-bounds MIN MAX` on every plugin.
- **GUI**: a "Lifetime preset" dropdown in the Model settings panel. Picking
  a preset fills in the model (mono/bi) and bounds fields; the fields stay
  editable afterward, so a preset is a starting point, not a locked mode.
  Built-in presets (`gui/mainwindow.py`, `LIFETIME_PRESETS`): Flipper-TR
  (biexponential, the flim_core/config.py defaults) and Fluorescein
  (mono-exponential, 2-5 ns). Add your own dye by adding an entry to that
  dict.

## Fit quality: uncertainty, photon count, convergence

Every per-pixel fitting function (`fit_pixels`, `fit_pixels_mono` in
`flim_core/fitting.py`, and their reconvolution equivalents in
`flim_core/irf.py`) returns a dict of `[H, W]` maps, not just the
lifetime(s) -- so a photon-starved pixel's unreliable fit is visible
instead of looking identical to a good one:

- `tau[_short/_long]_std` -- 1-sigma uncertainty on the lifetime, from the
  local curvature of the negative log-likelihood at the fit (the standard
  asymptotic maximum-likelihood variance estimate). Large where the fit is
  poorly constrained -- typically, too few photons.
- `n_photons` -- photons actually used in that pixel's fit.
- `converged` -- `scipy.optimize.minimize`'s own success flag.
- `nll` -- the negative log-likelihood at the optimum (goodness of fit;
  only comparable between pixels with similar photon counts).

The whole-image/pooled fits (`global_fit` and friends, which use
`scipy.curve_fit`) get the equivalent single-number `tau_std` for free, from
`curve_fit`'s own covariance matrix -- previously computed and discarded.

All of this is saved into every `*_maps.npz` alongside the lifetime maps
(`tau_long_std_ns`, `n_photons_fit`, `converged`, `nll`, etc.) and,
for the four per-pixel plugins (`normal`, `periodic`, `timelapse`, and their
whole-image siblings' per-group/per-window maps), rendered as a companion
`*_quality.png`: the uncertainty map, its histogram, and uncertainty vs.
photon count on a log axis -- the noise-floor curve, useful for comparing
different thresholds/binning against each other on the same acquisition.

Nothing is auto-masked based on these values -- high-uncertainty or
non-converged pixels still appear in the lifetime maps and results. That's
a deliberate choice: what counts as "too uncertain to trust" is a judgment
call best made by looking at the quality plot for your own data, not a
hardcoded cutoff baked into the pipeline.

**Output options**: the GUI has a "Save fit-quality plots" checkbox
(below the plugin list) that skips writing the `*_quality.png` for that run
-- useful once you already know what your acquisition's noise floor looks
like and don't need it re-plotted every time. The underlying `tau_std`/
`n_photons`/etc. values are still saved in the `.npz` either way; only the
extra PNG is skipped. On the CLI/in a batch job, the equivalent is
`--no-quality-plots` / `"save_quality_plots": false`.

## Fit methods: tail, IRF, or mean arrival time

Every plugin's `--fit-method` accepts three values -- this is a choice
about *how a lifetime gets computed from the decay data*, orthogonal to
which plugin/windowing scheme is in use, so it applies uniformly across
`normal`, `periodic`, `timelapse`, and their whole-image siblings:

- **`tail`** (default): fit the decay tail only (mono- or biexponential,
  see "Lifetime bounds and presets" above).
- **`irf`**: fit the FULL decay curve by reconvolution against a measured
  instrument response -- requires `--irf <file>`. Not yet available for
  `normal_wholeimage` (see that plugin's docstring).
- **`mean`**: no fit at all -- compute the photon-weighted mean arrival
  time of the decay tail instead, `tau_mean = sum(t_i * n_i) / sum(n_i)`
  (`flim_core/fitting.py`'s `mean_arrival_time`). For a true
  mono-exponential decay this equals tau exactly, with no optimizer
  involved; for a biexponential decay it isn't tau_short or tau_long, but
  tracks their intensity-weighted trend. It's fully vectorized -- no
  per-pixel optimizer loop, nothing to fail to converge, no lifetime bounds
  to set -- and reports an exact (not fit-curvature-approximated) standard
  error of the mean, `tau_mean_std`. `--mono`/`--fix-tau-short`/the
  lifetime-bounds fields have no effect with this method (there's only one
  number to compute), and the GUI greys out the whole Model section when
  it's selected.

Because `mean` is this cheap and robust at low photon counts, it's a good
first pass when scanning several threshold/binning settings quickly, or a
quick look at how a lifetime is trending across a timelapse/periodic scan,
before committing to a full (and slower, and less robust at low counts)
biexponential fit.

## Batch runs

To run several jobs -- different files, plugins, settings, output folders
-- from one command instead of one CLI invocation per job, use
`batch_run.py` with a JSON job list:

```bash
python batch_run.py jobs.json
```

```json
[
  {"plugin": "timelapse", "h5file": "file1.H5", "out_dir": "results/1sec",
   "step": 1, "threshold": 400, "bin": 2},
  {"plugin": "timelapse", "h5file": "file1.H5", "out_dir": "results/2sec",
   "step": 2, "threshold": 400, "bin": 2}
]
```

Only `plugin`, `h5file`, and `out_dir` are required per job; every other
setting (threshold, bin, mono, fit_method, tau_short_bounds, a plugin's own
EXTRA_ARGS like step/period/groups, ...) falls back to the same default the
CLI uses, or can be set per job. `out_dir` is created automatically.

To avoid repeating settings shared by many jobs (a long `h5file` path, a
common threshold) use the `"defaults"` + `"jobs"` object format instead --
every key in `defaults` applies to every job, and a job's own keys always
win over a matching default:

```json
{
  "defaults": {
    "h5file": "20x_beads_high10p_5s_timelapse_pixel_1s_thr500_irf.H5",
    "threshold": 500,
    "tail_start_offset": 50
  },
  "jobs": [
    {"plugin": "timelapse", "out_dir": "results/pixel_mono", "step": 1, "mono": true},
    {"plugin": "timelapse", "out_dir": "results/pixel_bi", "step": 1, "mono": false},
    {"plugin": "timelapse_wholeimage", "out_dir": "results/whole_mono", "step": 1, "mono": true}
  ]
}
```

Relative `h5file`/`out_dir`/`irf` paths are resolved against the job
file's own directory, not wherever you happen to run `batch_run.py` from,
so the same jobs.json behaves the same regardless of your current
directory. One job failing doesn't stop the rest of the batch -- failures
are collected and reported at the end, with a nonzero exit code. See
`batch_run.py`'s module docstring for the full option list.

## How a plugin works

Every file in `analyses/` (except the underscore-prefixed shared/template
ones) is a self-contained plugin that declares:

```python
NAME          = "Human-readable label"
DESCRIPTION   = "One-line explanation"
REQUIRES_RAW  = True/False   # needs a raw photon-stream file, or works on
                              # exported/.ds3d histogram files too
IRF_CAPABLE   = True/False   # accepts fit_method="irf"
EXTRA_ARGS    = [...]        # setting names beyond the common set, e.g.
                              # ["step"] or ["period", "groups"]

def run(args):
    ...

def preview_slice(loaded_data, settings):
    # returns (img, cube, description) for one representative slice --
    # e.g. "whole acquisition", "first time window", "first phase group"
    ...
```

`analyses/_registry.py` scans the folder and picks up every module with
this shape -- the GUI (`gui/mainwindow.py`) only ever talks to plugins
through `registry.list_plugins()`, `plugin["module"].run(ns)`, and
`plugin["module"].preview_slice(data, settings)`. It never hardcodes a
plugin's name, id, settings, or what kind of data-splitting (if any) it
does -- that's why the mask-preview panel works for any plugin without the
GUI knowing whether it splits by time, by phase, or not at all.

**To add a new analysis type:** drop a new file in `analyses/` with the
metadata above, a `run(args)` function, and a `preview_slice(loaded_data,
settings)` function. It shows up in the GUI's plugin list and gets a
settings panel automatically -- built from `EXTRA_ARGS` against the small
lookup table `EXTRA_ARG_SPECS` at the top of `gui/mainwindow.py` (a field
name it already recognises, like `"step"`, needs no GUI changes at all; a
brand new field name gets a plain labeled text box until someone adds a
nicer widget spec for it). A plugin missing any required attribute
(including `preview_slice`) is silently skipped by the registry rather
than crashing the GUI -- check the console for a `[plugin loader] skipping
...` message if a new plugin doesn't appear in the list.

**To add a new file format:** add a loader in `flim_core/io.py` following
the existing `_load_raw` / `_load_exported` pattern.

## Where things came from

This package is a reorganization of an earlier set of standalone scripts
(one per analysis type, plus a master GUI that imported all of them by
name). The underlying algorithms are unchanged; the split into
`flim_core/` (shared logic), `analyses/` (interchangeable plugins), and
`gui/` (plugin-agnostic UI) is new, and follows the same plugin-registry
pattern used by NanoPrepare (`protocols/screening/` + `protocols/importer.py`).
