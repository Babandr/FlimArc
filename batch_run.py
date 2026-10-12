#!/usr/bin/env python3
"""
Batch runner: run many analysis jobs -- different files, plugins, settings,
output folders -- from one job-list file, instead of one CLI invocation per
job. Each job goes through the exact same plugin interface (analyses/
_registry.py + plugin.run(ns)) the GUI drives, so this stays in sync with
whatever plugins exist automatically -- nothing here names a specific
plugin.

Usage:
    python batch_run.py jobs.json

Job list format: a JSON array of job objects. Only "plugin", "h5file", and
"out_dir" are required; everything else falls back to a sensible batch-safe
default (see JOB_DEFAULTS below), so a job only needs to specify what it's
actually varying.

Example jobs.json -- the same file analyzed with two different timelapse
step sizes, into two different output folders:

[
  {"plugin": "timelapse", "h5file": "file1.H5", "out_dir": "results/1sec",
   "step": 1, "threshold": 400, "bin": 2},
  {"plugin": "timelapse", "h5file": "file1.H5", "out_dir": "results/2sec",
   "step": 2, "threshold": 400, "bin": 2}
]

To avoid repeating settings shared by every job (e.g. a long h5file path,
a common threshold) across many jobs that only differ in one or two
things, use the "defaults" + "jobs" object format instead -- every key in
"defaults" is applied to every job, and a job's own keys always override
the matching default:

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

Any setting a plugin's CLI (--threshold, --bin, --mono, --fit-method, --irf,
--tau-short-bounds, etc.) accepts can be set as a job key, using the same
name minus the leading "--" and with "-" replaced by "_" (e.g. --fit-method
-> "fit_method", --tau-mono-bounds -> "tau_mono_bounds": [2.0, 5.0]). A
plugin's own EXTRA_ARGS fields (e.g. "step" for timelapse, "period"/"groups"
for periodic) are set the same way and are REQUIRED if that plugin declares
them -- run() will refuse the job with a clear error rather than silently
using a wrong default.

out_dir is created automatically if it doesn't exist. Each job's plugin
runs with that folder as its working directory (matching how the GUI runs
plugins), so output filenames don't need any folder logic of their own.

Relative h5file/out_dir/irf paths in the job file are resolved against the
JOB FILE'S OWN DIRECTORY, not against wherever you happen to run batch_run.py
from -- so the same jobs.json behaves identically regardless of your
current directory when you invoke it. Keep a jobs.json next to its data and
paths like "file1.H5" or "results/1sec" just work.

One job failing (bad settings, missing file, plugin exception) is reported
and does NOT stop the rest of the batch; the exit code is nonzero only if
at least one job failed, so this is safe to use in a script that should
notice failures without losing already-completed results.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # batch runs are always headless -- see gui/mainwindow.py's
                        # module docstring for why this matters: it keeps every
                        # plugin's fitting/plotting code from ever touching a
                        # GUI backend, regardless of how it's invoked.

from flim_core.config import DEFAULT_THRESHOLD, MIN_NEIGHBORS, BIN_SIZE
import analyses._registry as registry


class ArgsStub:
    """Blank object to hang attributes on -- equivalent to
    argparse.Namespace(), built directly from a job dict instead of the
    command line or the GUI. Every plugin's run(args) only reads attributes
    via getattr with its own default, so a job doesn't need to specify
    every possible setting -- only what it's using."""
    pass


# Batch-safe defaults for every COMMON setting (i.e. not a per-plugin
# EXTRA_ARGS field like "step"). A job's own dict overrides these.
JOB_DEFAULTS = dict(
    threshold=DEFAULT_THRESHOLD,   # never None in batch mode: None makes the
                                    # plugin call input() interactively, which
                                    # would hang a batch run waiting on stdin
    min_neighbors=MIN_NEIGHBORS,
    bin=BIN_SIZE,
    mono=False,
    fix_tau_short=False,
    tau_short_value=None,
    tail_start_offset=None,
    fit_method="tail",
    irf=None,
    tau_short_bounds=None,
    tau_long_bounds=None,
    tau_mono_bounds=None,
    roi=None,
    save_quality_plots=True,
    save_overview=True,
    save_results=True,
    save_irf_check=True,
    external_mask=None,
)


def _as_bounds_tuple(value):
    """job JSON gives bounds as a [min, max] list; every fitting function
    wants a (min, max) tuple. None (not overriding this bound) passes through."""
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"bounds must be a 2-element [min, max] list, got {value!r}")
    return (float(value[0]), float(value[1]))


def build_namespace(job, plugin):
    """Turn one job dict into the argparse.Namespace-like object
    plugin['module'].run(ns) expects -- the batch-mode equivalent of the
    GUI's _make_namespace(). Raises ValueError with a clear message for
    anything the job is missing or has wrong."""
    ns = ArgsStub()
    for key, default in JOB_DEFAULTS.items():
        setattr(ns, key, job.get(key, default))
    for key in ("tau_short_bounds", "tau_long_bounds", "tau_mono_bounds"):
        setattr(ns, key, _as_bounds_tuple(getattr(ns, key)))
    if ns.roi is not None:
        ns.roi = tuple(ns.roi)
    ns.roi_prompt = False   # batch mode is never interactive
    ns.no_show = True       # never pop up a plot window
    ns.prefix = job.get("prefix")   # None -> plugin uses the file's own stem

    for name in plugin["extra_args"]:
        if name not in job:
            raise ValueError(
                f"plugin '{plugin['id']}' requires setting '{name}' "
                f"(declared in its EXTRA_ARGS) but the job doesn't provide it")
        setattr(ns, name, job[name])

    if ns.fit_method == "irf" and not ns.irf:
        raise ValueError("fit_method is 'irf' but no 'irf' file was given in the job")

    return ns


def run_job(job_index, job, plugins_by_id, jobs_base_dir):
    plugin_id = job.get("plugin")
    if plugin_id not in plugins_by_id:
        available = ", ".join(sorted(plugins_by_id))
        raise ValueError(f"unknown plugin '{plugin_id}' -- available: {available}")
    plugin = plugins_by_id[plugin_id]

    if "h5file" not in job:
        raise ValueError("job is missing required key 'h5file'")
    if "out_dir" not in job:
        raise ValueError("job is missing required key 'out_dir'")

    # Relative h5file/out_dir/irf paths are resolved against the JOBS FILE's
    # own directory, not against wherever batch_run.py happens to be
    # invoked from -- otherwise the same jobs.json would silently point at
    # different files depending on your current directory when you run it.
    # (jobs_base_dir / p) leaves an already-absolute p untouched, so
    # absolute paths in the job file still work exactly as given.
    h5file = str((jobs_base_dir / job["h5file"]).resolve())
    out_dir = (jobs_base_dir / job["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ns = build_namespace(job, plugin)
    ns.h5file = h5file
    if ns.irf:
        ns.irf = str((jobs_base_dir / ns.irf).resolve())
    if ns.external_mask:
        ns.external_mask = str((jobs_base_dir / ns.external_mask).resolve())

    print(f"\n{'='*70}\n[job {job_index}] plugin={plugin_id}  file={job['h5file']}  "
         f"-> {job['out_dir']}\n{'='*70}")

    original_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        plugin["module"].run(ns)
    finally:
        os.chdir(original_cwd)


def load_jobs(jobs_file):
    """Load a job file in either supported format:
      - a plain JSON array of job objects (original format, still works
        exactly as before), or
      - a JSON object with a top-level "defaults" dict and a "jobs" array --
        every key in "defaults" is merged into each job BEFORE that job's
        own keys, so a job only needs to state what's different from the
        defaults (its own keys always win if both set the same key). This
        is the natural way to avoid retyping a long h5file path, a shared
        threshold, etc. across many jobs that only differ in one or two
        settings.
    Returns a plain list of (already-merged) job dicts, so the rest of the
    script never needs to know which format was used."""
    with open(jobs_file) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "jobs" in data:
        defaults = data.get("defaults", {})
        if not isinstance(defaults, dict):
            raise ValueError('"defaults" must be a JSON object (key: value pairs).')
        if not isinstance(data["jobs"], list):
            raise ValueError('"jobs" must be a JSON array of job objects.')
        merged = []
        for job in data["jobs"]:
            combined = dict(defaults)
            combined.update(job)   # job's own keys override defaults
            merged.append(combined)
        return merged

    raise ValueError(
        'The jobs file must be either a JSON array of job objects, or a JSON '
        'object with a "jobs" array (and optionally a "defaults" object).')


def main():
    ap = argparse.ArgumentParser(
        description="Run a batch of FLIM analysis jobs from a JSON job list.")
    ap.add_argument("jobs_file", help="path to a JSON file containing a list of job objects, "
                    "or an object with \"defaults\"/\"jobs\"")
    args = ap.parse_args()

    try:
        jobs = load_jobs(args.jobs_file)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    jobs_base_dir = Path(args.jobs_file).resolve().parent

    plugins = registry.list_plugins()
    plugins_by_id = {p["id"]: p for p in plugins}
    print(f"Loaded {len(jobs)} job(s) from {args.jobs_file}. "
         f"Available plugins: {', '.join(sorted(plugins_by_id))}")

    results = []
    for i, job in enumerate(jobs, start=1):
        try:
            run_job(i, job, plugins_by_id, jobs_base_dir)
            results.append((i, job, True, None))
        except Exception:
            err = traceback.format_exc()
            print(f"\n[job {i}] FAILED:\n{err}")
            results.append((i, job, False, err))

    n_ok = sum(1 for _, _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    print(f"\n{'='*70}\nBatch complete: {n_ok}/{len(results)} job(s) succeeded" +
         (f", {n_fail} FAILED" if n_fail else "") + f"\n{'='*70}")
    if n_fail:
        print("Failed jobs:")
        for i, job, ok, err in results:
            if not ok:
                print(f"  [job {i}] plugin={job.get('plugin')!r} h5file={job.get('h5file')!r} "
                     f"out_dir={job.get('out_dir')!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
