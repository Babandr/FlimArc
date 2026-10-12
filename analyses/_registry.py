"""
Auto-discovers every analysis plugin in this package, the same way
NanoPrepare's protocols/importer.py discovers screening plugins: import every
module in the folder whose filename doesn't start with "_", and collect the
ones that expose the plugin interface (a NAME attribute).

This is the one piece of code the GUI (or any other caller) needs to touch
to find out what plugins exist -- it never imports an analysis module by a
hardcoded name. Adding a 7th analysis type later means dropping a new file
in this folder; nothing here or in the GUI needs to change.

Plugin interface a module must expose to be picked up:
    NAME          -- str, shown in the GUI's dropdown
    DESCRIPTION   -- str, one-line explanation
    REQUIRES_RAW  -- bool, True if this plugin only works on raw photon-stream
                     files (needs per-photon timing), False if it also works
                     on exported/.ds3d histogram files
    IRF_CAPABLE   -- bool, True if this plugin accepts fit_method="irf"
    EXTRA_ARGS    -- list[str], names of settings (beyond the common set)
                     this plugin needs -- e.g. ["step"] or ["period", "groups"]
    run(args)     -- callable, args is an argparse.Namespace-like object
                     (or anything with the right attributes, e.g. the GUI's
                     ArgsStub); runs the plugin end to end
    preview_slice(loaded_data, settings)
                  -- callable, returns (img, cube, description) for one
                     representative slice of the loaded data (e.g. "the
                     whole acquisition" or "the first time window"), used
                     by the GUI's mask-preview panel BEFORE a full run.
                     settings is a dict with an "extra" sub-dict holding
                     this plugin's EXTRA_ARGS values. This is what lets the
                     GUI preview any plugin's data without knowing what
                     kind of splitting (if any) that plugin does.
"""

import importlib
import pkgutil

REQUIRED_ATTRS = ("NAME", "DESCRIPTION", "REQUIRES_RAW", "IRF_CAPABLE", "EXTRA_ARGS",
                   "run", "preview_slice")


def _iter_plugin_modules():
    pkg = importlib.import_module(__package__)  # the analyses package itself
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue  # template/shared-helper modules, not plugins (e.g. _common.py)
        try:
            mod = importlib.import_module(f".{info.name}", package=__package__)
        except ImportError as e:
            print(f"  [plugin loader] skipping '{info.name}': {e}")
            continue
        if all(hasattr(mod, attr) for attr in REQUIRED_ATTRS):
            yield info.name, mod


def list_plugins():
    """Returns a list of dicts, one per discovered plugin, each with an
    'id' (the module name -- stable, used to remember the user's last
    selection) plus every REQUIRED_ATTRS field and the module itself."""
    plugins = []
    seen_names = {}   # NAME -> module that first claimed it
    for module_name, mod in _iter_plugin_modules():
        # Two modules declaring the same NAME would show up as indistinguishable
        # duplicate entries in the GUI dropdown -- typically a stale backup copy
        # (e.g. "normal_old.py") left next to the real plugin. Modules are visited
        # in alphabetical order, so the first one wins; say so on the console
        # rather than silently hiding it.
        if mod.NAME in seen_names:
            print(f"  [plugin loader] '{module_name}' declares the same NAME "
                  f"('{mod.NAME}') as '{seen_names[mod.NAME]}' -- ignoring it. "
                  f"If it's an old backup, delete it from the analyses folder.")
            continue
        seen_names[mod.NAME] = module_name
        plugins.append(dict(
            id=module_name,
            module=mod,
            name=mod.NAME,
            description=mod.DESCRIPTION,
            requires_raw=mod.REQUIRES_RAW,
            irf_capable=mod.IRF_CAPABLE,
            extra_args=list(mod.EXTRA_ARGS),
        ))
    plugins.sort(key=lambda p: p["name"])
    return plugins


def get(plugin_id):
    """Look up one plugin's module by its id (as returned by list_plugins)."""
    mod = importlib.import_module(f".{plugin_id}", package=__package__)
    if not all(hasattr(mod, attr) for attr in REQUIRED_ATTRS):
        raise ValueError(f"'{plugin_id}' does not implement the plugin interface")
    return mod
