#!/usr/bin/env python3
"""
Flimera FLIM Analysis -- GUI

Drives whatever analysis plugins analyses/_registry.py discovers -- it never
imports a plugin module by name, never lists plugin ids, and never branches
on "which plugin is this". Adding, removing, or renaming a plugin in
analyses/ changes what shows up here automatically; nothing in this file
needs to change.

What the GUI DOES still need to know, to build a sensible settings panel and
a mask preview, is the *shape* of a plugin's needs -- and that shape comes
from the plugin's own declared metadata (REQUIRES_RAW, IRF_CAPABLE,
EXTRA_ARGS), not from its identity:
  - REQUIRES_RAW tells it whether the plugin can accept a histogram file,
    and (together with EXTRA_ARGS) which kind of preview slice to show.
  - EXTRA_ARGS names the settings fields a plugin needs beyond the common
    set; EXTRA_ARG_SPECS below is the small, open-ended catalogue of what a
    field named "step" or "period" means as a widget. A brand new plugin
    that reuses an existing field name (e.g. another "step"-based windowing
    scheme) needs zero GUI changes; a plugin introducing a genuinely new
    field name gets a generic labeled text entry automatically.

Two different threading strategies are used deliberately:
  - The full analysis (potentially a long per-pixel fitting loop) always
    runs in a background thread, with matplotlib's pyplot-level backend
    fixed to "Agg" so none of that code ever touches Tkinter.
  - The two on-screen preview panels (raw overview, mask preview) are fast
    (no fitting -- just binning and thresholding) and run on the main
    thread instead, using LIVE embedded matplotlib canvases (Figure +
    FigureCanvasTkAgg, built directly rather than through pyplot) so they
    can resize with the window and respond to mouse hover for a live
    photon-count readout.

Requires: numpy, scipy, h5py, matplotlib, tkinter (standard library).
"""

import queue
import subprocess
import sys
import threading
import traceback
import os
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # protects the background-thread analysis calls (see
                        # module docstring); the GUI's own live preview
                        # panels build Figure/FigureCanvasTkAgg directly,
                        # which works independently of this setting.

import h5py
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

from flim_core.config import (
    DISPLAY_ASPECT, MIN_NEIGHBORS, BIN_SIZE, DEFAULT_THRESHOLD, TAIL_START_OFFSET,
    TAU_SHORT_BOUNDS, TAU_LONG_BOUNDS, TAU_MONO_BOUNDS,
)
from flim_core.pixel_offset import PixelOffsetCorrection
from flim_core.io import load_flim_data, _find_histogram_group
from flim_core.masking import clean_mask, bin_cube, moving_average_cube, crop_to_roi, save_mask, load_mask
from flim_core.fitting import AnalysisCancelled
from analyses._common import load_raw_with_time, build_group_cube
import analyses._registry as registry
from gui.decay_viewer import DecayViewer


# ----------------------------------------------------------------------------- #
# The only place a "new kind of plugin setting" needs to be taught anything.
# Maps a field name (as it appears in a plugin's EXTRA_ARGS) to how to show
# it: (label text, tk Var class, default value). A field name not listed
# here still gets a working (generic) text-entry widget -- see
# _build_plugin_settings_widgets -- so a brand new plugin never breaks the
# GUI, it just gets a slightly less polished widget until someone adds an
# entry here.
# ----------------------------------------------------------------------------- #
EXTRA_ARG_SPECS = {
    "step":   ("Step size (s):", tk.DoubleVar, 5.0),
    "period": ("Period (s):", tk.DoubleVar, 1.0),
    "groups": ("Number of groups:", tk.IntVar, 5),
}

# Human-readable labels for the two other places in the GUI that pick
# between a couple of fixed options (selection mode, fit method) -- shown
# as a single-line dropdown, same compact look as the Model dropdown,
# instead of a 2-3 line stack of radio buttons. The GUI still stores and
# passes around the short internal token ("threshold"/"roi",
# "tail"/"irf") everywhere else in the code; only the widget itself
# translates to/from the friendly label -- see _build_mapped_combobox.
SELECTION_MODE_LABELS = {"threshold": "Threshold mask", "roi": "ROI", "external": "External mask"}
SPATIAL_MODE_LABELS = {"bin": "Block binning", "moving_average": "Moving average"}
FIT_METHOD_LABELS = {"tail": "Tail fit", "irf": "IRF fit (reconvolution)", "mean": "Mean arrival time",
                     "mean_irf": "Mean arrival time (IRF-corrected)"}


# ----------------------------------------------------------------------------- #
# Named lifetime-bounds presets, since re-typing four bound numbers every time
# you switch dye/sample types is exactly the kind of thing a preset should
# save you from. Add an entry here for any dye you use regularly; "Custom"
# is the no-op entry that leaves whatever is currently in the fields alone.
# ----------------------------------------------------------------------------- #
DEFAULT_PRESET_NAME = "Flipper-TR (biexponential)"
LIFETIME_PRESETS = {
    DEFAULT_PRESET_NAME: dict(mono=False, tau_short=TAU_SHORT_BOUNDS, tau_long=TAU_LONG_BOUNDS),
    "Fluorescein (mono-exponential)": dict(mono=True, tau_mono=(2.0, 5.0)),
    "Custom": None,
}


# ----------------------------------------------------------------------------- #
# File type detection
# ----------------------------------------------------------------------------- #
def detect_file_type(path):
    """Returns 'raw', 'histogram', or None (unrecognized)."""
    with h5py.File(path, "r") as f:
        if _find_histogram_group(f) is not None:
            return "histogram"
        found = {"raw": False}
        def check(name, obj):
            if found["raw"]:
                return
            if isinstance(obj, h5py.Dataset) and obj.ndim == 1 and obj.dtype == np.uint32:
                found["raw"] = True
        f.visititems(check)
        if found["raw"]:
            return "raw"
    return None


# ----------------------------------------------------------------------------- #
# Thread-safe stdout -> Tkinter log redirection
# ----------------------------------------------------------------------------- #
class QueueWriter:
    """print() calls from the background analysis thread get captured here
    and pushed onto a queue; the Tkinter main thread polls the queue and
    appends to the log widget -- Tkinter widgets must only ever be touched
    from the main thread, so this indirection is required."""
    def __init__(self, q):
        self.q = q
    def write(self, text):
        if text:
            self.q.put(("log", text))
    def flush(self):
        pass


class ArgsStub:
    """A blank object to hang attributes on -- equivalent to
    argparse.Namespace(), built directly from the GUI's settings instead of
    the command line. Every plugin's run(args) only ever reads attributes
    off this via getattr with a default, so it's fine if a given plugin
    doesn't use every attribute set here."""
    pass


# ----------------------------------------------------------------------------- #
# A live, resizable, hoverable image panel embedded in Tkinter
# ----------------------------------------------------------------------------- #
class ImagePanel:
    """One embedded matplotlib figure showing a photon-count image, with an
    optional mask overlay, that resizes with its container and shows a
    live photon-count readout under the mouse cursor."""

    def __init__(self, parent, title, hover_label):
        self.hover_label = hover_label
        self.current_img = None

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        widget = self.canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)
        widget.bind("<Configure>", self._on_resize)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("figure_leave_event", lambda e: self.hover_label.config(text=""))
        self._im = None
        self._colorbar = None
        self._roi_selector = None
        self._roi_callback = None

    def enable_roi_selector(self, on_select):
        """Turn on click-and-drag rectangle selection on this panel.
        on_select(x0, y0, x1, y1) is called (with integer pixel bounds, x1/y1
        exclusive) once the user finishes dragging a box."""
        self._roi_callback = on_select
        if self._roi_selector is None:
            self._roi_selector = RectangleSelector(
                self.ax, self._on_rectangle_selected,
                useblit=True, button=[1], minspanx=1, minspany=1,
                spancoords="data", interactive=True,
            )
        self._roi_selector.set_active(True)

    def disable_roi_selector(self):
        if self._roi_selector is not None:
            self._roi_selector.set_active(False)

    def _on_rectangle_selected(self, eclick, erelease):
        if self.current_img is None or self._roi_callback is None:
            return
        h, w = self.current_img.shape
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        x0 = max(0, int(round(x0)))
        y0 = max(0, int(round(y0)))
        x1 = min(w, int(round(x1)) + 1)
        y1 = min(h, int(round(y1)) + 1)
        if x1 > x0 and y1 > y0:
            self._roi_callback(x0, y0, x1, y1)

    def _on_resize(self, event):
        if event.width < 50 or event.height < 50:
            return
        dpi = self.fig.get_dpi()
        self.fig.set_size_inches(event.width / dpi, event.height / dpi)
        self.canvas.draw_idle()

    def _on_hover(self, event):
        if self.current_img is None or event.inaxes != self.ax:
            self.hover_label.config(text="")
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        h, w = self.current_img.shape
        if 0 <= x < w and 0 <= y < h:
            val = self.current_img[y, x]
            self.hover_label.config(text=f"pixel ({x}, {y}): {int(val)} photons")
        else:
            self.hover_label.config(text="")

    def show_photon_image(self, img, title, mask=None, removed=None, unit="pixel"):
        """Render a photon-count image, optionally with a cyan mask contour
        and red x's for isolated pixels removed by the neighbour filter."""
        self.current_img = img
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self._roi_selector = None
        if mask is None:
            vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1
            self._im = self.ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax,
                                      aspect=DISPLAY_ASPECT, origin="upper")
            self.fig.colorbar(self._im, ax=self.ax, fraction=0.046, pad=0.02, label="photons")
        else:
            self._im = self.ax.imshow(np.log10(np.maximum(img, 0.5)), cmap="gray",
                                      aspect=DISPLAY_ASPECT, origin="upper")
            self.fig.colorbar(self._im, ax=self.ax, fraction=0.046, pad=0.02, label="log\u2081\u2080 photons")
            self.ax.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
            if removed is not None:
                ys, xs = np.where(removed)
                if xs.size:
                    self.ax.scatter(xs, ys, s=10, c="red", marker="x", linewidths=0.6)
        self.ax.set_title(title, fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def clear(self, placeholder_text=""):
        self.current_img = None
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        if placeholder_text:
            self.ax.text(0.5, 0.5, placeholder_text, ha="center", va="center",
                         transform=self.ax.transAxes, color="#888", fontsize=10, wrap=True)
        self.canvas.draw_idle()


# ----------------------------------------------------------------------------- #
# Main application
# ----------------------------------------------------------------------------- #
class FlimMasterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flimera FLIM Analysis")
        self.geometry("1300x800")

        self.plugins = registry.list_plugins()   # discovered once at startup
        self.plugins_by_id = {p["id"]: p for p in self.plugins}
        if not self.plugins:
            messagebox.showerror("No plugins found",
                                 "No analysis plugins were found in analyses/. "
                                 "Nothing to run.")

        self.filepath = None
        self.file_type = None
        self.plugin_id = None
        self.loaded_data = None   # cached load: 's' dict (histogram) or 'd' dict (raw)
        self.external_mask_path = None   # path to a loaded external mask file, if any
        self.last_preview_mask = None    # last mask shown by Apply -- what "Save mask..." saves
        self.cancel_event = None         # threading.Event for the currently-running analysis, if any
        self.decay_viewer_window = None  # the DecayViewer Toplevel, if currently open
        self.q = queue.Queue()
        self.running = False

        self._build_ui()
        self._set_plugin_availability()

    # -- layout ---------------------------------------------------------------
    def _build_mapped_combobox(self, parent, var, labels_dict, on_change):
        """A readonly Combobox that shows human labels but keeps `var`
        holding the same short internal token it always has (e.g.
        "tail"/"irf") -- every other reference to `var` elsewhere in this
        file is completely unaffected. Gives an N-option choice the same
        compact single-line look as the Model dropdown, instead of an
        N-row stack of radio buttons."""
        display_var = tk.StringVar(value=labels_dict[var.get()])
        inverse = {v: k for k, v in labels_dict.items()}

        def _on_select(event=None):
            var.set(inverse[display_var.get()])
            on_change()

        combo = ttk.Combobox(parent, textvariable=display_var, state="readonly",
                             values=list(labels_dict.values()), width=20)
        combo.bind("<<ComboboxSelected>>", _on_select)
        return combo

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.minsize(1150, 700)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        ttk.Button(top, text="Open File...", command=self.open_file).pack(side="left")
        self.decay_viewer_button = ttk.Button(top, text="Decay viewer...", command=self.open_decay_viewer)
        self.decay_viewer_button.pack(side="left", padx=(6, 0))
        self.decay_viewer_button.state(["disabled"])
        self.file_label = ttk.Label(top, text="No file loaded", foreground="#555")
        self.file_label.pack(side="left", padx=12)

        main = ttk.Frame(self)
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=0)

        # -- left: plugin list, built from the registry, not hardcoded --
        # plus, below it, general output-file options -- this column has
        # spare vertical space (it's only as tall as a handful of plugin
        # names) so this is where per-run "how much do you want saved"
        # toggles live, rather than crowding the taller settings column.
        left_container = ttk.Frame(main)
        left_container.grid(row=0, column=0, sticky="ns", padx=(0, 8))

        left = ttk.LabelFrame(left_container, text="Plugin")
        left.pack(fill="x")
        self.plugin_buttons = {}
        self.plugin_var = tk.StringVar(value="")
        for p in self.plugins:
            b = ttk.Radiobutton(left, text=p["name"], value=p["id"], variable=self.plugin_var,
                                command=self._on_plugin_selected, width=24)
            b.pack(anchor="w", padx=8, pady=4)
            self.plugin_buttons[p["id"]] = b

        output_frame = ttk.LabelFrame(left_container, text="Output options")
        output_frame.pack(fill="x", pady=(8, 0))
        self.save_overview_var = tk.BooleanVar(value=True)
        self.save_results_var = tk.BooleanVar(value=True)
        self.save_quality_var = tk.BooleanVar(value=True)
        self.save_irf_check_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, variable=self.save_overview_var,
                       text="Overview / mask preview plot").pack(anchor="w", padx=8, pady=(6, 2))
        ttk.Checkbutton(output_frame, variable=self.save_results_var,
                       text="Results plot (lifetime map)").pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(output_frame, variable=self.save_quality_var,
                       text="Fit-quality diagnostic plot").pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(output_frame, variable=self.save_irf_check_var,
                       text="IRF check plot").pack(anchor="w", padx=8, pady=(2, 6))
        ttk.Label(output_frame, text="The underlying data (.npz) is always saved --\n"
                 "these only control which plots get written.",
                 foreground="#777", font=("", 8), wraplength=220, justify="left").pack(
            anchor="w", padx=8, pady=(0, 6))

        # -- center: two live image panels side by side --
        center = ttk.Frame(main)
        center.grid(row=0, column=1, sticky="nsew", padx=4)
        center.columnconfigure(0, weight=1); center.columnconfigure(1, weight=1)
        center.rowconfigure(0, weight=0)
        center.rowconfigure(1, weight=1)

        left_img_frame = ttk.LabelFrame(center, text="Overview (whole acquisition)")
        left_img_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        right_img_frame = ttk.LabelFrame(center, text="Mask preview (click Apply)")
        right_img_frame.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

        self.hover_label = ttk.Label(center, text="", foreground="#333", font=("Courier", 9))
        self.hover_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=6)

        self.overview_panel = ImagePanel(left_img_frame, "Photons per pixel", self.hover_label)
        self.mask_panel = ImagePanel(right_img_frame, "Mask preview", self.hover_label)
        self.overview_panel.clear("Open a file to see a preview here")
        self.mask_panel.clear("Click Apply (in Masking settings) to preview the mask")

        # -- right: settings, in a scrollable panel -- there are enough
        # settings groups (plugin-specific, masking, fit method, model +
        # lifetime bounds) that on a shorter screen this column can run
        # taller than the window; a mouse-wheel-scrollable canvas means
        # every setting stays reachable regardless of screen height,
        # instead of some controls being cut off with no way to reach them.
        right_outer = ttk.Frame(main)
        right_outer.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        right_outer.rowconfigure(0, weight=1)
        right_outer.columnconfigure(0, weight=1)

        right_canvas = tk.Canvas(right_outer, highlightthickness=0, width=260)
        right_scroll = ttk.Scrollbar(right_outer, orient="vertical", command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right_canvas.grid(row=0, column=0, sticky="nsew")
        right_scroll.grid(row=0, column=1, sticky="ns")

        right = ttk.Frame(right_canvas)
        right_window = right_canvas.create_window((0, 0), window=right, anchor="nw")

        def _on_right_configure(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        right.bind("<Configure>", _on_right_configure)

        def _on_canvas_configure(event):
            right_canvas.itemconfig(right_window, width=event.width)
        right_canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel only scrolls the settings panel while the cursor is
        # actually over it -- bound/unbound on enter/leave so it doesn't
        # hijack scrolling anywhere else in the window (the log box, etc.).
        def _on_mousewheel(event):
            delta = event.delta if event.delta else (-120 if event.num == 4 else 120)
            right_canvas.yview_scroll(int(-delta / 120), "units")
        def _bind_mousewheel(event):
            right_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            right_canvas.bind_all("<Button-4>", _on_mousewheel)
            right_canvas.bind_all("<Button-5>", _on_mousewheel)
        def _unbind_mousewheel(event):
            right_canvas.unbind_all("<MouseWheel>")
            right_canvas.unbind_all("<Button-4>")
            right_canvas.unbind_all("<Button-5>")
        right_canvas.bind("<Enter>", _bind_mousewheel)
        right_canvas.bind("<Leave>", _unbind_mousewheel)

        self.plugin_settings_frame = ttk.Frame(right)
        self.plugin_settings_frame.pack(fill="x", pady=(0, 8))
        self._build_plugin_settings_widgets()
        self._show_plugin_settings(None)

        self._build_masking_settings(right)
        self._build_fit_method_settings(right)
        self._build_model_settings(right)

        run_row = ttk.Frame(right)
        run_row.pack(fill="x", padx=8, pady=(16, 4))
        run_row.columnconfigure(0, weight=1)
        run_row.columnconfigure(1, weight=1)
        self.run_button = ttk.Button(run_row, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.run_button.state(["disabled"])
        self.stop_button = ttk.Button(run_row, text="Stop", command=self.stop_analysis)
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(2, 0))
        self.stop_button.state(["disabled"])
        self.status_label = ttk.Label(right, text="", foreground="#555", wraplength=240)
        self.status_label.pack(fill="x", padx=8)

        bottom = ttk.LabelFrame(self, text="Log")
        bottom.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.log_box = scrolledtext.ScrolledText(bottom, height=9, state="disabled", font=("Courier", 9))
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

    # -- plugin-specific settings: one frame per distinct EXTRA_ARGS shape,
    # built generically from EXTRA_ARG_SPECS (or a plain text entry for an
    # unrecognized field name) rather than one hardcoded frame per plugin --
    def _build_plugin_settings_widgets(self):
        self.extra_vars = {}          # field name -> tk Var, shared across plugins that reuse a name
        self._extra_frames = {}       # frozenset(field names) -> frame, one per distinct shape

        shapes = {frozenset(p["extra_args"]) for p in self.plugins}
        for shape in shapes:
            if not shape:
                continue
            frame = ttk.LabelFrame(self.plugin_settings_frame, text="Plugin settings")
            for row, field in enumerate(sorted(shape)):
                label, var_cls, default = EXTRA_ARG_SPECS.get(field, (f"{field}:", tk.StringVar, ""))
                if field not in self.extra_vars:
                    self.extra_vars[field] = var_cls(value=default)
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
                ttk.Entry(frame, textvariable=self.extra_vars[field], width=10).grid(
                    row=row, column=1, padx=4)
            self._extra_frames[shape] = frame

        self.none_frame = ttk.Label(self.plugin_settings_frame,
                                    text="No extra settings for this plugin.", foreground="#777")

    def _show_plugin_settings(self, plugin_id):
        for w in list(self._extra_frames.values()) + [self.none_frame]:
            w.pack_forget()
        shape = frozenset(self.plugins_by_id[plugin_id]["extra_args"]) if plugin_id else frozenset()
        if shape and shape in self._extra_frames:
            self._extra_frames[shape].pack(fill="x")
        else:
            self.none_frame.pack(fill="x")

    def _build_masking_settings(self, parent):
        f = ttk.LabelFrame(parent, text="Masking")
        f.pack(fill="x", pady=8)

        sm = ttk.Frame(f)
        sm.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(sm, text="Spatial smoothing:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.spatial_mode_var = tk.StringVar(value="bin")
        self._build_mapped_combobox(sm, self.spatial_mode_var, SPATIAL_MODE_LABELS,
                                    self._on_spatial_mode_changed).grid(row=0, column=1, padx=4)
        self.bin_size_label = ttk.Label(sm, text="Binning (NxN):")
        self.bin_size_label.grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.bin_var = tk.IntVar(value=BIN_SIZE)
        ttk.Spinbox(sm, from_=1, to=21, textvariable=self.bin_var, width=8).grid(row=1, column=1, sticky="w", padx=4)
        self.spatial_note_label = ttk.Label(
            sm, text="", foreground="#777", font=("", 8), wraplength=240, justify="left")
        self.spatial_note_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Label(sm, text="Time binning (bins):").grid(row=3, column=0, sticky="w", padx=4, pady=3)
        self.time_bin_var = tk.IntVar(value=1)
        ttk.Spinbox(sm, from_=1, to=16, textvariable=self.time_bin_var, width=8).grid(
            row=3, column=1, sticky="w", padx=4)
        ttk.Label(sm, text="Groups N adjacent time channels (1 = off). Tail offset auto-rescales.",
                  foreground="#777", font=("", 8), wraplength=240, justify="left").grid(
            row=4, column=0, columnspan=2, sticky="w", padx=4)
        self._on_spatial_mode_changed()

        ttk.Separator(f, orient="horizontal").grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)

        ttk.Label(f, text="Selection mode:").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        self.selection_mode_var = tk.StringVar(value="threshold")
        self._build_mapped_combobox(f, self.selection_mode_var, SELECTION_MODE_LABELS,
                                    self._on_selection_mode_changed).grid(row=2, column=1, padx=4)

        self.threshold_settings_frame = ttk.Frame(f)
        ttk.Label(self.threshold_settings_frame, text="Photon threshold:").grid(
            row=0, column=0, sticky="w", padx=4, pady=3)
        self.threshold_var = tk.IntVar(value=DEFAULT_THRESHOLD)
        ttk.Entry(self.threshold_settings_frame, textvariable=self.threshold_var, width=10).grid(
            row=0, column=1, padx=4)
        ttk.Label(self.threshold_settings_frame, text="Min neighbours:").grid(
            row=1, column=0, sticky="w", padx=4, pady=3)
        self.min_neighbors_var = tk.IntVar(value=MIN_NEIGHBORS)
        ttk.Spinbox(self.threshold_settings_frame, from_=0, to=8, textvariable=self.min_neighbors_var, width=8).grid(
            row=1, column=1, padx=4)

        self.roi_settings_frame = ttk.Frame(f)
        ttk.Button(self.roi_settings_frame, text="Draw ROI on Overview",
                  command=self._start_roi_draw).grid(row=0, column=0, columnspan=4, sticky="ew", padx=4, pady=(2, 4))
        ttk.Label(self.roi_settings_frame, text="x0:").grid(row=1, column=0, sticky="e")
        ttk.Label(self.roi_settings_frame, text="y0:").grid(row=1, column=2, sticky="e")
        ttk.Label(self.roi_settings_frame, text="x1:").grid(row=2, column=0, sticky="e")
        ttk.Label(self.roi_settings_frame, text="y1:").grid(row=2, column=2, sticky="e")
        self.roi_x0_var = tk.StringVar(value="")
        self.roi_y0_var = tk.StringVar(value="")
        self.roi_x1_var = tk.StringVar(value="")
        self.roi_y1_var = tk.StringVar(value="")
        ttk.Entry(self.roi_settings_frame, textvariable=self.roi_x0_var, width=6).grid(row=1, column=1, padx=2)
        ttk.Entry(self.roi_settings_frame, textvariable=self.roi_y0_var, width=6).grid(row=1, column=3, padx=2)
        ttk.Entry(self.roi_settings_frame, textvariable=self.roi_x1_var, width=6).grid(row=2, column=1, padx=2)
        ttk.Entry(self.roi_settings_frame, textvariable=self.roi_y1_var, width=6).grid(row=2, column=3, padx=2)
        ttk.Button(self.roi_settings_frame, text="Clear ROI", command=self._clear_roi).grid(
            row=3, column=0, columnspan=4, sticky="ew", padx=4, pady=(4, 2))
        ttk.Label(self.roi_settings_frame,
                 text="Drag a box on the Overview panel, or type coordinates directly. "
                      "Every pixel inside the box is used -- no photon-count filtering.",
                 foreground="#777", font=("", 8), wraplength=220).grid(
            row=4, column=0, columnspan=4, sticky="w", padx=4)

        self.external_mask_frame = ttk.Frame(f)
        ttk.Button(self.external_mask_frame, text="Load mask file...",
                  command=self._select_external_mask).grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(2, 4))
        self.external_mask_label = ttk.Label(self.external_mask_frame, text="No mask file selected",
                                             foreground="#777", font=("", 8), wraplength=220)
        self.external_mask_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Label(self.external_mask_frame,
                 text="Uses exactly the pixels saved in this file -- no threshold or "
                      "neighbour filter applied. Its shape must match the current image "
                      "(after binning/ROI). Useful for analyzing the same pixels across "
                      "acquisitions that need different thresholds.",
                 foreground="#777", font=("", 8), wraplength=220).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0))

        self._show_selection_mode_settings()

        button_row = ttk.Frame(f)
        button_row.grid(row=10, column=0, columnspan=2, sticky="ew", padx=4, pady=(8, 4))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        self.apply_button = ttk.Button(button_row, text="Apply", command=self.apply_mask_preview)
        self.apply_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))
        self.apply_button.state(["disabled"])
        self.save_mask_button = ttk.Button(button_row, text="Save mask...", command=self._save_mask)
        self.save_mask_button.grid(row=0, column=1, sticky="ew", padx=(2, 0))
        self.save_mask_button.state(["disabled"])

    def _show_selection_mode_settings(self):
        self.threshold_settings_frame.grid_forget()
        self.roi_settings_frame.grid_forget()
        self.external_mask_frame.grid_forget()
        mode = self.selection_mode_var.get()
        if mode == "roi":
            self.roi_settings_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        elif mode == "external":
            self.external_mask_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        else:
            self.threshold_settings_frame.grid(row=4, column=0, columnspan=2, sticky="ew")

    def _select_external_mask(self):
        path = filedialog.askopenfilename(
            title="Select a saved mask file",
            filetypes=[("NumPy mask files", "*.npy"), ("All files", "*.*")]
        )
        if not path:
            return
        self.external_mask_path = path
        self.external_mask_label.config(text=Path(path).name, foreground="#333")

    def _save_mask(self):
        if self.last_preview_mask is None:
            messagebox.showinfo("Nothing to save", "Click Apply first to build a mask, then save it.")
            return
        path = filedialog.asksaveasfilename(
            title="Save mask as...", defaultextension=".npy",
            filetypes=[("NumPy mask files", "*.npy"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            save_mask(self.last_preview_mask, path)
        except Exception as e:
            messagebox.showerror("Couldn't save mask", str(e))
            return
        messagebox.showinfo("Mask saved", f"Saved {int(self.last_preview_mask.sum())} pixels to:\n{path}")

    def _on_selection_mode_changed(self):
        self._show_selection_mode_settings()
        if self.selection_mode_var.get() != "roi":
            self.overview_panel.disable_roi_selector()

    def _start_roi_draw(self):
        if self.filepath is None:
            messagebox.showinfo("Open a file first", "Load a file before drawing an ROI.")
            return
        self.overview_panel.enable_roi_selector(self._on_roi_drawn)

    def _on_roi_drawn(self, x0, y0, x1, y1):
        self.roi_x0_var.set(str(x0)); self.roi_y0_var.set(str(y0))
        self.roi_x1_var.set(str(x1)); self.roi_y1_var.set(str(y1))

    def _clear_roi(self):
        self.roi_x0_var.set(""); self.roi_y0_var.set("")
        self.roi_x1_var.set(""); self.roi_y1_var.set("")
        self.overview_panel.disable_roi_selector()

    def _get_roi_bounds(self):
        vals = {}
        for name, var in (("x0", self.roi_x0_var), ("y0", self.roi_y0_var),
                          ("x1", self.roi_x1_var), ("y1", self.roi_y1_var)):
            raw = var.get().strip()
            if not raw:
                raise ValueError("ROI mode is selected but no ROI has been set -- "
                                 "draw one on the Overview panel or type all four coordinates.")
            try:
                vals[name] = int(raw)
            except ValueError:
                raise ValueError(f"ROI {name} value '{raw}' isn't a whole number.")
        if vals["x1"] <= vals["x0"] or vals["y1"] <= vals["y0"]:
            raise ValueError("ROI is empty: x1 must be greater than x0, and y1 greater than y0.")
        return vals["x0"], vals["y0"], vals["x1"], vals["y1"]

    def _build_fit_method_settings(self, parent):
        f = ttk.LabelFrame(parent, text="Fit method")
        f.pack(fill="x", pady=(0, 8))

        ttk.Label(f, text="Fit method:").grid(row=0, column=0, sticky="w", padx=4, pady=(4, 3))
        self.fit_method_var = tk.StringVar(value="tail")
        self._build_mapped_combobox(f, self.fit_method_var, FIT_METHOD_LABELS,
                                    self._on_fit_method_changed).grid(row=0, column=1, padx=4, pady=(4, 3))

        self.tail_settings_frame = ttk.Frame(f)
        ttk.Label(self.tail_settings_frame, text="Tail start offset (bins):").grid(
            row=0, column=0, sticky="w", padx=4, pady=3)
        self.tail_start_offset_var = tk.IntVar(value=TAIL_START_OFFSET)
        ttk.Entry(self.tail_settings_frame, textvariable=self.tail_start_offset_var, width=8).grid(
            row=0, column=1, padx=4)
        ttk.Label(self.tail_settings_frame, text="bins after the decay peak where the tail fit starts",
                 foreground="#777", font=("", 8), wraplength=200).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=4)

        self.irf_settings_frame = ttk.Frame(f)
        self.irf_path = None
        ttk.Button(self.irf_settings_frame, text="Select IRF file...", command=self._select_irf_file).grid(
            row=0, column=0, sticky="w", padx=4, pady=3)
        self.irf_label = ttk.Label(self.irf_settings_frame, text="No IRF file selected",
                                   foreground="#777", font=("", 8), wraplength=200)
        self.irf_label.grid(row=1, column=0, sticky="w", padx=4)

        self.no_shift_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.irf_settings_frame, text="No shift (fix IRF/data timing offset at 0)",
                        variable=self.no_shift_var).grid(row=2, column=0, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(self.irf_settings_frame,
                 text="Only use this if the IRF and data were acquired at matched laser settings "
                      "in the same session -- otherwise a real timing offset can go uncorrected.",
                 foreground="#777", font=("", 8), wraplength=220).grid(
            row=3, column=0, sticky="w", padx=4)

        self.irf_full_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.irf_settings_frame, text="Use full IRF trace (keep secondary peaks)",
                        variable=self.irf_full_var, command=self._update_irf_window_end_state).grid(
            row=4, column=0, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(self.irf_settings_frame,
                 text="Off (default): window to just the main peak, discarding any secondary bump as "
                      "a measurement artifact. On: keep the trace out to the setting below -- use this "
                      "if secondary peaks are believed to be a real, reproducible part of the "
                      "instrument response (e.g. laser/SPAD echoes) rather than an artifact.",
                 foreground="#777", font=("", 8), wraplength=220).grid(
            row=5, column=0, sticky="w", padx=4)

        irf_window_row = ttk.Frame(self.irf_settings_frame)
        irf_window_row.grid(row=6, column=0, sticky="w", padx=4, pady=(4, 0))
        ttk.Label(irf_window_row, text="Keep out to:").grid(row=0, column=0)
        self.irf_window_end_ns_var = tk.DoubleVar(value=15.0)
        self.irf_window_end_entry = ttk.Entry(irf_window_row, textvariable=self.irf_window_end_ns_var, width=6)
        self.irf_window_end_entry.grid(row=0, column=1, padx=4)
        ttk.Label(irf_window_row, text="ns after the main peak").grid(row=0, column=2)
        self._update_irf_window_end_state()

        ttk.Label(self.irf_settings_frame,
                 text="Only plugins that declare IRF_CAPABLE are enabled while IRF fit is selected.",
                 foreground="#a55", font=("", 8), wraplength=200).grid(
            row=7, column=0, sticky="w", padx=4, pady=(8, 0))

        self._show_fit_method_settings()

        # ── Pixel offset correction (always visible, independent of fit method) ──
        f_off = ttk.LabelFrame(parent, text="Pixel offset correction")
        f_off.pack(fill="x", pady=(0, 8))

        self.pixel_offset_var = tk.BooleanVar(value=False)
        self.pixel_offset_path = None
        ttk.Checkbutton(
            f_off, text="Apply per-pixel SPAD timing offset correction",
            variable=self.pixel_offset_var,
            command=self._update_pixel_offset_state,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0))
        ttk.Label(
            f_off,
            text="Corrects the fixed per-pixel TDC offset measured from an IRF "
                 "calibration file (.npy). Apply before fitting to remove systematic "
                 "timing spread (~103 ps rms) across the sensor. "
                 "Uncheck to run without correction for a side-by-side comparison.",
            foreground="#777", font=("", 8), wraplength=260,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=4)

        self._pixel_offset_btn = ttk.Button(
            f_off, text="Select calibration .npy...",
            command=self._select_pixel_offset_file, state="disabled",
        )
        self._pixel_offset_btn.grid(row=2, column=0, sticky="w", padx=4, pady=(6, 0))
        self._pixel_offset_label = ttk.Label(
            f_off, text="No calibration file selected",
            foreground="#777", font=("", 8), wraplength=220,
        )
        self._pixel_offset_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=4)

    def _update_pixel_offset_state(self):
        state = "normal" if self.pixel_offset_var.get() else "disabled"
        self._pixel_offset_btn.config(state=state)
        if not self.pixel_offset_var.get():
            self.pixel_offset_path = None
            self._pixel_offset_label.config(text="No calibration file selected",
                                            foreground="#777")

    def _select_pixel_offset_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("NumPy array", "*.npy"), ("All files", "*.*")],
            title="Select pixel offset calibration .npy file",
        )
        if not path:
            return
        self.pixel_offset_path = path
        self._pixel_offset_label.config(text=Path(path).name, foreground="#333")

    def _update_irf_window_end_state(self):
        self.irf_window_end_entry.config(state="normal" if self.irf_full_var.get() else "disabled")

    def _show_fit_method_settings(self):
        self.tail_settings_frame.grid_forget()
        self.irf_settings_frame.grid_forget()
        if self.fit_method_var.get() in ("irf", "mean_irf"):
            self.irf_settings_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        else:
            self.tail_settings_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

    def _select_irf_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Flimera files", "*.h5 *.H5 *.hdf5 *.ds3d"), ("All files", "*.*")]
        )
        if not path:
            return
        self.irf_path = path
        self.irf_label.config(text=Path(path).name, foreground="#333")

    def _on_fit_method_changed(self):
        self._show_fit_method_settings()
        self._update_fix_tau_short_state()
        self._set_plugin_availability()
        current = self.plugins_by_id.get(self.plugin_id)
        if self.fit_method_var.get() in ("irf", "mean_irf") and current is not None and not current["irf_capable"] \
                and self.filepath is not None:
            # Fall back to any IRF-capable plugin rather than a specific one by name.
            fallback = next((p for p in self.plugins if p["irf_capable"]), None)
            if fallback is not None:
                self.plugin_var.set(fallback["id"])
                self._on_plugin_selected()

    def _build_model_settings(self, parent):
        f = ttk.LabelFrame(parent, text="Model")
        f.pack(fill="x", pady=(0, 8))

        ttk.Label(f, text="Model:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.model_var = tk.StringVar(value="Biexponential")
        self.model_combo = ttk.Combobox(f, textvariable=self.model_var, state="readonly", width=15,
                                        values=["Biexponential", "Mono-exponential"])
        self.model_combo.grid(row=0, column=1, padx=4)
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self._update_fix_tau_short_state())

        self.fix_tau_short_var = tk.BooleanVar(value=False)
        self.fix_tau_short_check = ttk.Checkbutton(f, text="Fix tau_short", variable=self.fix_tau_short_var,
                                                    command=self._update_fix_tau_short_state)
        self.fix_tau_short_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        ttk.Label(f, text="tau_short value (ns):").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        self.tau_short_value_var = tk.StringVar(value="")
        self.tau_short_value_entry = ttk.Entry(f, textvariable=self.tau_short_value_var, width=10)
        self.tau_short_value_entry.grid(row=2, column=1, padx=4)
        ttk.Label(f, text="(blank = auto-derive from the data)", foreground="#777",
                 font=("", 8)).grid(row=3, column=0, columnspan=2, sticky="w", padx=4)

        ttk.Separator(f, orient="horizontal").grid(row=4, column=0, columnspan=2, sticky="ew", pady=6)

        # -- lifetime-bounds preset --
        ttk.Label(f, text="Lifetime preset:").grid(row=5, column=0, sticky="w", padx=4, pady=3)
        self.preset_var = tk.StringVar(value=DEFAULT_PRESET_NAME)
        self.preset_combo = ttk.Combobox(f, textvariable=self.preset_var, state="readonly", width=15,
                                         values=list(LIFETIME_PRESETS.keys()))
        self.preset_combo.grid(row=5, column=1, padx=4)
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        # -- mono-exponential bounds --
        self.mono_bounds_frame = ttk.Frame(f)
        ttk.Label(self.mono_bounds_frame, text="tau bounds (ns):").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.tau_mono_min_var = tk.DoubleVar(value=TAU_MONO_BOUNDS[0])
        self.tau_mono_max_var = tk.DoubleVar(value=TAU_MONO_BOUNDS[1])
        ttk.Entry(self.mono_bounds_frame, textvariable=self.tau_mono_min_var, width=6).grid(row=0, column=1, padx=2)
        ttk.Label(self.mono_bounds_frame, text="-").grid(row=0, column=2)
        ttk.Entry(self.mono_bounds_frame, textvariable=self.tau_mono_max_var, width=6).grid(row=0, column=3, padx=2)

        # -- biexponential bounds --
        self.bi_bounds_frame = ttk.Frame(f)
        ttk.Label(self.bi_bounds_frame, text="tau_short bounds (ns):").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.tau_short_min_var = tk.DoubleVar(value=TAU_SHORT_BOUNDS[0])
        self.tau_short_max_var = tk.DoubleVar(value=TAU_SHORT_BOUNDS[1])
        ttk.Entry(self.bi_bounds_frame, textvariable=self.tau_short_min_var, width=6).grid(row=0, column=1, padx=2)
        ttk.Label(self.bi_bounds_frame, text="-").grid(row=0, column=2)
        ttk.Entry(self.bi_bounds_frame, textvariable=self.tau_short_max_var, width=6).grid(row=0, column=3, padx=2)
        ttk.Label(self.bi_bounds_frame, text="tau_long bounds (ns):").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.tau_long_min_var = tk.DoubleVar(value=TAU_LONG_BOUNDS[0])
        self.tau_long_max_var = tk.DoubleVar(value=TAU_LONG_BOUNDS[1])
        ttk.Entry(self.bi_bounds_frame, textvariable=self.tau_long_min_var, width=6).grid(row=1, column=1, padx=2)
        ttk.Label(self.bi_bounds_frame, text="-").grid(row=1, column=2)
        ttk.Entry(self.bi_bounds_frame, textvariable=self.tau_long_max_var, width=6).grid(row=1, column=3, padx=2)

        ttk.Label(f, text="Bounds fill in from the preset but stay editable -- "
                 "picking a preset doesn't lock the fields.",
                 foreground="#777", font=("", 8), wraplength=220).grid(
            row=7, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0))

        self.mean_note_label = ttk.Label(f,
            text="Not applicable -- mean arrival time has only one number "
                 "to compute, no model or bounds needed.",
            foreground="#777", font=("", 8), wraplength=220, justify="left")

        self._update_fix_tau_short_state()
        # Apply the default preset's values immediately so the fields start
        # in a matching state rather than the raw config.py defaults.
        self._on_preset_selected()

    def _on_preset_selected(self, event=None):
        """Fill the model + bounds fields from a named preset. The fields
        stay fully editable afterward -- selecting a preset is a one-time
        "load these values" action, not a locked mode. Selecting "Custom"
        leaves whatever values are currently in the fields untouched."""
        preset = LIFETIME_PRESETS.get(self.preset_var.get())
        if preset is None:
            return
        self.model_var.set("Mono-exponential" if preset["mono"] else "Biexponential")
        if preset["mono"]:
            lo, hi = preset["tau_mono"]
            self.tau_mono_min_var.set(lo); self.tau_mono_max_var.set(hi)
        else:
            lo, hi = preset["tau_short"]
            self.tau_short_min_var.set(lo); self.tau_short_max_var.set(hi)
            lo, hi = preset["tau_long"]
            self.tau_long_min_var.set(lo); self.tau_long_max_var.set(hi)
        self._update_fix_tau_short_state()

    def _update_fix_tau_short_state(self):
        """Enable/disable every Model-section widget based on both the
        selected model (mono vs bi -- disables Fix tau_short for mono, as
        before) AND the selected fit method: when fit_method is "mean",
        the whole Model section is irrelevant (mean arrival time has no
        model, no bounds, nothing to fix) and gets greyed out entirely,
        the same way individual widgets already grey out for mono."""
        mean_selected = getattr(self, "fit_method_var", None) is not None \
            and self.fit_method_var.get() in ("mean", "mean_irf")
        mono = self.model_var.get() == "Mono-exponential"

        self.model_combo.state(["disabled"] if mean_selected else ["!disabled"])
        self.preset_combo.state(["disabled"] if mean_selected else ["!disabled"])
        self.fix_tau_short_check.state(["disabled"] if (mean_selected or mono) else ["!disabled"])
        enable_value = (not mean_selected) and (not mono) and self.fix_tau_short_var.get()
        self.tau_short_value_entry.state(["!disabled"] if enable_value else ["disabled"])

        self.mono_bounds_frame.grid_forget()
        self.bi_bounds_frame.grid_forget()
        self.mean_note_label.grid_forget()
        if mean_selected:
            self.mean_note_label.grid(row=6, column=0, columnspan=2, sticky="w", padx=4)
        elif mono:
            self.mono_bounds_frame.grid(row=6, column=0, columnspan=2, sticky="w")
        else:
            self.bi_bounds_frame.grid(row=6, column=0, columnspan=2, sticky="w")

    def _get_lifetime_bounds(self):
        """Read+validate the bounds fields. Returns (tau_short_bounds,
        tau_long_bounds, tau_mono_bounds) as (min, max) tuples -- only the
        pair relevant to the current model actually affects the fit, but
        all three are always returned so the namespace can be built
        uniformly regardless of which model is selected."""
        pairs = dict(
            tau_short_bounds=(self.tau_short_min_var.get(), self.tau_short_max_var.get()),
            tau_long_bounds=(self.tau_long_min_var.get(), self.tau_long_max_var.get()),
            tau_mono_bounds=(self.tau_mono_min_var.get(), self.tau_mono_max_var.get()),
        )
        for name, (lo, hi) in pairs.items():
            if lo >= hi:
                raise ValueError(f"{name.replace('_', ' ')}: min ({lo}) must be less than max ({hi}).")
        return pairs

    # -- file loading -----------------------------------------------------------
    def open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Flimera files", "*.h5 *.H5 *.hdf5 *.ds3d"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            ftype = detect_file_type(path)
        except Exception as e:
            messagebox.showerror("Error reading file", str(e))
            return
        if ftype is None:
            messagebox.showerror("Unrecognized file",
                                 "This doesn't look like a Flimera raw photon-stream or "
                                 "histogram file (no matching dataset found).")
            return

        try:
            if ftype == "histogram":
                data = load_flim_data(path)
                preview_img = data["img"]
            else:
                data = load_raw_with_time(path)
                preview_img, _ = build_group_cube(data, slice(None))
        except Exception as e:
            messagebox.showerror("Error loading file", f"{e}\n\n{traceback.format_exc()}")
            return

        self.filepath = path
        self.file_type = ftype
        self.loaded_data = data
        name = Path(path).name
        kind = "raw photon-stream" if ftype == "raw" else "pre-computed histogram (exported/.ds3d)"
        self.file_label.config(text=f"{name}  [{kind}]")

        # A different file is loaded now -- close any open decay viewer
        # rather than leave it showing data from the previous file.
        if self.decay_viewer_window is not None and self.decay_viewer_window.winfo_exists():
            self.decay_viewer_window.destroy()
        self.decay_viewer_window = None
        self.decay_viewer_button.state(["!disabled"])

        self.overview_panel.show_photon_image(preview_img, "Photons per pixel")
        self.mask_panel.clear("Click Apply to preview the mask")
        self.last_preview_mask = None
        self.save_mask_button.state(["disabled"])
        self._set_plugin_availability()
        self.plugin_var.set("")
        self.plugin_id = None
        self.run_button.state(["disabled"])
        self.apply_button.state(["!disabled"])
        self._log_clear()
        self._log(f"Loaded {name} ({kind}).\n")

    def open_decay_viewer(self):
        """Open (or re-focus, if already open) the per-pixel decay curve
        inspector for the currently loaded file. Never re-reads the file --
        passes along the same loaded_data the main window already has."""
        if self.filepath is None or self.loaded_data is None:
            return
        if self.decay_viewer_window is not None and self.decay_viewer_window.winfo_exists():
            self.decay_viewer_window.lift()
            self.decay_viewer_window.focus_force()
            return
        self.decay_viewer_window = DecayViewer(
            self, self.loaded_data, self.file_type, Path(self.filepath).name)

    # -- plugin selection ---------------------------------------------------
    def _set_plugin_availability(self):
        irf_active = getattr(self, "fit_method_var", None) is not None \
            and self.fit_method_var.get() in ("irf", "mean_irf")
        for p in self.plugins:
            btn = self.plugin_buttons[p["id"]]
            ok = self.filepath is not None and (not p["requires_raw"] or self.file_type == "raw")
            if irf_active and not p["irf_capable"]:
                ok = False
            btn.state(["!disabled"] if ok else ["disabled"])

    def _on_plugin_selected(self):
        self.plugin_id = self.plugin_var.get()
        self._show_plugin_settings(self.plugin_id)
        self.run_button.state(["!disabled"] if self.filepath else ["disabled"])
        self.mask_panel.clear("Click Apply to preview the mask")
        self.last_preview_mask = None
        self.save_mask_button.state(["disabled"])

    # -- mask preview (Apply button) -------------------------------------------
    def _extra_settings(self):
        """Current values of every declared plugin EXTRA_ARGS field, keyed
        by field name -- the "extra" sub-dict every plugin's run()/
        preview_slice() receives, regardless of which plugin is selected."""
        return {name: var.get() for name, var in self.extra_vars.items()}

    def _on_spatial_mode_changed(self):
        if self.spatial_mode_var.get() == "moving_average":
            self.bin_size_label.config(text="Window (NxN, odd):")
            self.spatial_note_label.config(
                text="Full resolution kept; neighbouring pixels share photons (not independent). "
                     "Pixel-wise plugins only.")
        else:
            self.bin_size_label.config(text="Binning (NxN):")
            self.spatial_note_label.config(text="")

    def _apply_spatial_smoothing(self, img, cube):
        """The SAME spatial step the real run applies (block bin or moving
        average), so the mask preview's photon counts -- and therefore the
        threshold you pick from it -- match what actually gets analyzed.
        Returns (img, cube, unit)."""
        n = self.bin_var.get()
        if self.spatial_mode_var.get() == "moving_average":
            if n and n > 1:
                if n % 2 == 0:
                    raise ValueError(f"Moving-average window must be odd (got {n}).")
                img, cube = moving_average_cube(img, cube, n)
            return img, cube, "pixel"
        if n and n > 1:
            img, cube = bin_cube(img, cube, n)
            return img, cube, "super-pixel"
        return img, cube, "pixel"

    def _smoothing_desc(self):
        """Short human description of the current spatial smoothing, for
        titles/log lines (e.g. "bin=2", "moving average 5x5")."""
        n = self.bin_var.get()
        if self.spatial_mode_var.get() == "moving_average":
            return f"moving average {n}x{n}" if n > 1 else "no smoothing"
        return f"bin={n}"

    def _spatial_settings(self):
        """bin / moving_average / time_bin for the settings dict, validated.
        In moving-average mode `bin` is reported as 1 (block binning isn't
        applied), so anything reading settings['bin'] sees the truth."""
        n = self.bin_var.get()
        time_bin = self.time_bin_var.get()
        if time_bin < 1:
            raise ValueError("Time binning must be at least 1.")
        if self.spatial_mode_var.get() == "moving_average":
            if n > 1 and n % 2 == 0:
                raise ValueError(f"Moving-average window must be odd (got {n}) -- an even window "
                                 f"has no single center pixel.")
            if self.plugin_id and self.plugin_id.endswith("_wholeimage") and n > 1:
                raise ValueError("Moving average isn't available for whole-image plugins: they pool "
                                 "every masked pixel, and moving-averaged pixels share photons, so "
                                 "pooling would count each photon up to N\u00b2 times and overstate "
                                 "precision. Use block binning, or a pixel-wise plugin.")
            return dict(bin=1, moving_average=(n if n > 1 else None), time_bin=time_bin)
        return dict(bin=n, moving_average=None, time_bin=time_bin)

    def _first_slice_img_cube(self):
        """Return (img, cube, description) for whatever "first slice" the
        current plugin wants to preview the mask against. This is entirely
        the selected plugin's own call -- the GUI just hands it the loaded
        data and current settings via preview_slice() and shows whatever
        comes back. It never needs to know what kind of splitting (if any)
        a given plugin does."""
        plugin_id = self.plugin_id
        if plugin_id is None or plugin_id not in self.plugins_by_id:
            raise ValueError("Select a plugin first.")
        plugin = self.plugins_by_id[plugin_id]
        if plugin["requires_raw"] and self.file_type != "raw":
            raise ValueError("This plugin needs a raw photon-stream file.")
        settings = dict(extra=self._extra_settings())
        return plugin["module"].preview_slice(self.loaded_data, settings)

    def apply_mask_preview(self):
        if not self.filepath:
            return
        try:
            self.config(cursor="watch"); self.update_idletasks()
            img, cube, desc = self._first_slice_img_cube()
            mode = self.selection_mode_var.get()

            if mode == "roi":
                roi = self._get_roi_bounds()
                img, cube, roi = crop_to_roi(img, cube, roi)
                img, cube, unit = self._apply_spatial_smoothing(img, cube)
                mask = np.ones(img.shape, dtype=bool)
                removed = np.zeros(img.shape, dtype=bool)
                title = f"{desc}, ROI {roi}: all {img.size} {unit}s used (no threshold)"
                self.mask_panel.show_photon_image(img, title, mask=mask, removed=removed, unit=unit)
                self._log(f"Mask preview ({desc}): ROI {roi} -> {img.shape[1]}x{img.shape[0]} "
                          f"{unit}s, every one used, no threshold filtering.\n")
            elif mode == "external":
                if not self.external_mask_path:
                    raise ValueError("Selection mode is External mask but no mask file has been "
                                     "chosen (use 'Load mask file...' above).")
                img, cube, unit = self._apply_spatial_smoothing(img, cube)
                mask = load_mask(self.external_mask_path)
                if mask.shape != img.shape:
                    raise ValueError(f"External mask shape {mask.shape} does not match this "
                                     f"image's shape {img.shape} -- check that binning/ROI match "
                                     f"how the mask was saved.")
                removed = np.zeros(img.shape, dtype=bool)
                title = f"{desc}: {int(mask.sum())} {unit}s from external mask"
                self.mask_panel.show_photon_image(img, title, mask=mask, removed=removed, unit=unit)
                self._log(f"Mask preview ({desc}): loaded {int(mask.sum())} {unit}s from "
                          f"{Path(self.external_mask_path).name}.\n")
            else:
                img, cube, unit = self._apply_spatial_smoothing(img, cube)
                threshold = self.threshold_var.get()
                min_neighbors = self.min_neighbors_var.get()
                raw_mask = img > threshold
                mask = clean_mask(raw_mask, min_neighbors)
                removed = raw_mask & ~mask
                title = (f"{desc}: {int(mask.sum())} {unit}s kept "
                        f"(threshold={threshold}, {self._smoothing_desc()})")
                self.mask_panel.show_photon_image(img, title, mask=mask, removed=removed, unit=unit)
                self._log(f"Mask preview ({desc}): {int(raw_mask.sum())} {unit}s above threshold, "
                          f"{int(mask.sum())} kept after neighbour filter.\n")

            self.last_preview_mask = mask
            self.save_mask_button.state(["!disabled"])
        except Exception as e:
            messagebox.showerror("Couldn't build mask preview", f"{e}\n\n{traceback.format_exc()}")
        finally:
            self.config(cursor="")

    # -- run analysis ---------------------------------------------------------
    def _gather_settings(self):
        tau_short_value = None
        raw = self.tau_short_value_var.get().strip()
        if raw:
            try:
                tau_short_value = float(raw)
            except ValueError:
                raise ValueError(f"tau_short value '{raw}' isn't a number.")

        fit_method = self.fit_method_var.get()
        if fit_method in ("irf", "mean_irf") and not self.irf_path:
            raise ValueError("This fit method requires an IRF file but none has been chosen "
                             "(use 'Select IRF file...' in the Fit method settings).")

        selection_mode = self.selection_mode_var.get()
        if selection_mode == "roi":
            roi_bounds = self._get_roi_bounds()
            threshold = 0
            min_neighbors = 0
            external_mask_path = None
        elif selection_mode == "external":
            if not self.external_mask_path:
                raise ValueError("Selection mode is External mask but no mask file has been "
                                 "chosen (use 'Load mask file...' in the Masking settings).")
            roi_bounds = None
            threshold = None
            min_neighbors = None
            external_mask_path = self.external_mask_path
        else:
            roi_bounds = None
            threshold = self.threshold_var.get()
            min_neighbors = self.min_neighbors_var.get()
            external_mask_path = None

        extra = self._extra_settings()
        lifetime_bounds = self._get_lifetime_bounds()   # raises ValueError if a min >= max

        return dict(
            **self._spatial_settings(),
            threshold=threshold,
            mono=(self.model_var.get() == "Mono-exponential"),
            fix_tau_short=self.fix_tau_short_var.get(),
            tau_short_value=tau_short_value,
            min_neighbors=min_neighbors,
            extra=extra,
            fit_method=fit_method,
            tail_start_offset=self.tail_start_offset_var.get(),
            irf_path=self.irf_path if fit_method in ("irf", "mean_irf") else None,
            no_shift=self.no_shift_var.get() if fit_method in ("irf", "mean_irf") else False,
            irf_full=self.irf_full_var.get() if fit_method in ("irf", "mean_irf") else False,
            irf_window_end_ns=self.irf_window_end_ns_var.get(),
            pixel_offset_path=(self.pixel_offset_path
                               if self.pixel_offset_var.get() else None),
            selection_mode=selection_mode,
            roi_bounds=roi_bounds,
            external_mask_path=external_mask_path,
            lifetime_preset=self.preset_var.get(),
            save_quality_plots=self.save_quality_var.get(),
            save_overview=self.save_overview_var.get(),
            save_results=self.save_results_var.get(),
            save_irf_check=self.save_irf_check_var.get(),
            **lifetime_bounds,
        )

    def _make_namespace(self, settings, prefix):
        """Build the argparse.Namespace-like object every plugin's run(args)
        reads from. Common attributes are set unconditionally; the current
        plugin's EXTRA_ARGS are copied in by name so any plugin (present or
        future) gets exactly the extra attributes it declared needing.

        tau_short_bounds/tau_long_bounds/tau_mono_bounds are always set,
        regardless of which model is selected -- every plugin's fitting
        calls read whichever pair is actually relevant (mono vs bi) via
        getattr, and ignore the other."""
        ns = ArgsStub()
        ns.h5file = self.filepath
        ns.bin = settings["bin"]
        ns.moving_average = settings["moving_average"]
        ns.time_bin = settings["time_bin"]
        ns.threshold = settings["threshold"]
        ns.mono = settings["mono"]
        ns.fix_tau_short = settings["fix_tau_short"]
        ns.tau_short_value = settings["tau_short_value"]
        ns.min_neighbors = settings["min_neighbors"]
        ns.no_show = True
        ns.prefix = prefix
        ns.roi = settings["roi_bounds"]
        ns.roi_prompt = False
        ns.tail_start_offset = settings["tail_start_offset"]
        ns.fit_method = settings["fit_method"]
        ns.irf = settings["irf_path"]
        ns.no_shift = settings["no_shift"]
        ns.irf_full = settings["irf_full"]
        ns.irf_window_end_ns = settings["irf_window_end_ns"]
        ns.pixel_offset = settings["pixel_offset_path"]
        ns.tau_short_bounds = settings["tau_short_bounds"]
        ns.tau_long_bounds = settings["tau_long_bounds"]
        ns.tau_mono_bounds = settings["tau_mono_bounds"]
        ns.save_quality_plots = settings["save_quality_plots"]
        ns.save_overview = settings["save_overview"]
        ns.save_results = settings["save_results"]
        ns.save_irf_check = settings["save_irf_check"]
        ns.external_mask = settings["external_mask_path"]
        for name in self.plugins_by_id[self.plugin_id]["extra_args"]:
            setattr(ns, name, settings["extra"].get(name))
        return ns

    # Base word each plugin id contributes to the output folder name's
    # measurement-type postfix (see _output_folder_postfix). Empty string
    # means "no word" for that plugin family -- currently just "normal",
    # per the user's choice to keep single-acquisition folder names
    # uncluttered. Matched by stripping a trailing "_wholeimage" from the
    # plugin id first, so this dict only needs one entry per FAMILY, not
    # one per pixel-wise/whole-image pair; an id that isn't a recognized
    # family (a plugin added later) falls back to using its own id as the
    # word, so new plugins still get a sensible postfix with no code change
    # required here.
    MEASUREMENT_WORDS = {"normal": "", "normal_old": "", "periodic": "periodic", "timelapse": "timelapse"}

    def _output_folder_postfix(self, plugin_id, settings):
        """Build the "_timelapse_pixel_bin2_thr100_biexp_irf"-style postfix
        appended to the output folder name, in 4 groups, each independently
        omittable/adaptable but always in this fixed order:
          1. measurement type: family word (see MEASUREMENT_WORDS) plus
             "_pixel" for the pixel-wise variant of a family, nothing extra
             for the whole-image variant (e.g. timelapse pixel-wise ->
             "_timelapse_pixel", timelapse whole-image -> "_timelapse",
             normal pixel-wise -> "_pixel", normal whole-image -> "").
          2. binning: "_bin{N}"
          3. masking: "_thr{N}" for a threshold mask (the common case), or
             "_roi"/"_extmask" for the other two selection modes -- masking
             always leaves SOME marker, since which pixels got analyzed is
             as important to the filename as the fit settings are.
          4. fit method: "_mono"/"_biexp"/"_mean" (no IRF), or
             "_mono_irf"/"_biexp_irf"/"_mean_irf" (with IRF) -- mean/mean_irf
             have no mono/bi distinction, matching settings['mono'] having
             no effect for either (see _write_metadata's is_mean handling)."""
        is_whole = plugin_id.endswith("_wholeimage")
        family = plugin_id[: -len("_wholeimage")] if is_whole else plugin_id
        word = self.MEASUREMENT_WORDS.get(family, family)
        if is_whole:
            measurement = f"_{word}" if word else ""
        else:
            measurement = f"_{word}_pixel" if word else "_pixel"

        if settings.get("moving_average"):
            binning = f"_ma{settings['moving_average']}"
        else:
            binning = f"_bin{settings['bin']}"
        if settings.get("time_bin", 1) > 1:
            binning += f"_tb{settings['time_bin']}"

        mode = settings["selection_mode"]
        if mode == "threshold":
            masking = f"_thr{settings['threshold']}"
        elif mode == "roi":
            masking = "_roi"
        else:
            masking = "_extmask"

        fit_method = settings["fit_method"]
        if fit_method == "mean":
            fit = "_mean"
        elif fit_method == "mean_irf":
            fit = "_mean_irf"
        elif fit_method == "irf":
            fit = "_mono_irf" if settings["mono"] else "_biexp_irf"
        else:  # "tail"
            fit = "_mono" if settings["mono"] else "_biexp"

        return measurement + binning + masking + fit

    def run_analysis(self):
        if self.running:
            return
        if not self.filepath or not self.plugin_id:
            messagebox.showwarning("Nothing to run", "Open a file and select a plugin first.")
            return

        try:
            settings = self._gather_settings()
        except ValueError as e:
            messagebox.showerror("Invalid setting", str(e))
            return

        plugin = self.plugins_by_id[self.plugin_id]
        if settings["fit_method"] in ("irf", "mean_irf") and not plugin["irf_capable"]:
            messagebox.showerror("Not supported",
                                 "This fit method isn't available for this plugin.")
            return

        src = Path(self.filepath)
        out_folder = src.parent / (src.stem + self._output_folder_postfix(self.plugin_id, settings))
        try:
            out_folder.mkdir(exist_ok=True)
        except Exception as e:
            messagebox.showerror("Can't create output folder", str(e))
            return

        self._write_metadata(out_folder, plugin, settings)

        prefix = None
        ns = self._make_namespace(settings, prefix)
        self.cancel_event = threading.Event()
        ns.cancel_check = self.cancel_event.is_set

        self.running = True
        self.run_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.status_label.config(text=f"Running {plugin['name']} ...")
        self._log(f"\n=== Running: {plugin['name']} ===\n")
        self._log(f"Output folder: {out_folder}\n")

        t = threading.Thread(target=self._run_worker,
                             args=(plugin, ns, out_folder), daemon=True)
        t.start()
        self.after(100, self._poll_queue)

    def stop_analysis(self):
        """Signal the running worker thread to stop. Python threads can't
        be safely force-killed, so this is cooperative: it sets an Event
        that the plugin's per-pixel fitting loops (and, for the fast
        fit methods, the per-window/group loops) check periodically and
        raise AnalysisCancelled from -- see flim_core/fitting.py. The run
        stops at the next checkpoint, not instantly, but for any run worth
        stopping that's at most a moment away."""
        if not self.running or self.cancel_event is None:
            return
        self.cancel_event.set()
        self.stop_button.state(["disabled"])
        self.status_label.config(text="Stopping... (finishing current pixel/window)")
        self._log("\n--- Stop requested; finishing the current step, then halting. ---\n")

    def _run_worker(self, plugin, ns, out_folder):
        """Every plugin, whatever it is, gets run the exact same way: call
        its run(ns). No per-plugin branching -- that's the whole point of
        every plugin sharing one interface (see analyses/_registry.py)."""
        writer = QueueWriter(self.q)
        original_cwd = os.getcwd()
        try:
            os.chdir(out_folder)
            with redirect_stdout(writer):
                plugin["module"].run(ns)
            self.q.put(("done", str(out_folder)))
        except AnalysisCancelled as e:
            self.q.put(("cancelled", str(e)))
        except Exception:
            self.q.put(("error", traceback.format_exc()))
        finally:
            os.chdir(original_cwd)

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "done":
                    self._on_run_finished(status="done", out_folder=payload)
                elif kind == "cancelled":
                    self._log(f"\n=== Stopped: {payload} ===\n")
                    self._on_run_finished(status="cancelled", out_folder=None)
                elif kind == "error":
                    self._log(f"\n--- ERROR ---\n{payload}\n")
                    self._on_run_finished(status="error", out_folder=None)
        except queue.Empty:
            pass
        if self.running:
            self.after(100, self._poll_queue)

    def _on_run_finished(self, status, out_folder):
        self.running = False
        self.cancel_event = None
        self.run_button.state(["!disabled"])
        self.stop_button.state(["disabled"])
        if status == "done":
            self.status_label.config(text="Done.")
            self._log(f"\n=== Finished. Files saved to: {out_folder} ===\n")
            if messagebox.askyesno("Analysis complete",
                                   f"Saved to:\n{out_folder}\n\nOpen the folder now?"):
                self._open_folder(out_folder)
        elif status == "cancelled":
            self.status_label.config(text="Stopped by user.")
        else:
            self.status_label.config(text="Failed -- see log.")
            messagebox.showerror("Analysis failed", "See the log panel for details.")

    def _open_folder(self, path):
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", path])
            elif sys.platform.startswith("win"):
                subprocess.Popen(["explorer", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass

    def _write_metadata(self, out_folder, plugin, settings):
        """Only record settings that actually applied to this run. The
        plugin's own EXTRA_ARGS list drives which extra fields get logged --
        no per-plugin-id special casing."""
        meta_path = out_folder / "metadata.txt"
        lines = [
            "=" * 60,
            f"Run at:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Source file: {self.filepath}",
            f"File type:   {self.file_type}",
            f"Plugin:      {plugin['name']}",
            "Settings:",
            (f"  moving_average: {settings['moving_average']}" if settings.get("moving_average")
             else f"  bin: {settings['bin']}"),
            *([f"  time_bin: {settings['time_bin']}"] if settings.get("time_bin", 1) > 1 else []),
            f"  selection_mode: {settings['selection_mode']}",
        ]
        if settings["selection_mode"] == "roi":
            lines.append(f"  roi (x0,y0,x1,y1): {settings['roi_bounds']}")
        elif settings["selection_mode"] == "external":
            lines.append(f"  external_mask_file: {settings['external_mask_path']}")
        else:
            lines.append(f"  threshold: {settings['threshold']}")
            lines.append(f"  min_neighbors: {settings['min_neighbors']}")

        # fit_method=="mean" or "mean_irf" (mean arrival time, with or without
        # IRF correction) doesn't fit any lifetime model at all -- no mono/bi
        # choice, no lifetime_preset, no tau bounds, and fix_tau_short is
        # meaningless for either (normal.py itself prints "has no effect" and
        # ignores both when either fit method is selected). Keep the metadata
        # file honest about what actually ran instead of always recording the
        # biexponential settings regardless of fit method.
        is_mean = settings["fit_method"] in ("mean", "mean_irf")
        if not is_mean:
            lines.append(f"  model: {'mono' if settings['mono'] else 'bi'}")
            lines.append(f"  lifetime_preset: {settings['lifetime_preset']}")
            if settings["mono"]:
                lines.append(f"  tau_mono_bounds: {settings['tau_mono_bounds']}")
            else:
                lines.append(f"  tau_short_bounds: {settings['tau_short_bounds']}")
                lines.append(f"  tau_long_bounds: {settings['tau_long_bounds']}")
        lines.append(f"  fit_method: {settings['fit_method']}")
        if settings["fit_method"] in ("irf", "mean_irf"):
            lines.append(f"  irf_file: {settings['irf_path']}")
            lines.append(f"  no_shift: {settings['no_shift']}")
            lines.append(f"  irf_full: {settings['irf_full']}")
            if settings["irf_full"]:
                lines.append(f"  irf_window_end_ns: {settings['irf_window_end_ns']}")
        else:
            # Also applies to fit_method=="mean": mean_arrival_time still gets its
            # tail axis (s_idx, tn) from _tail_axis(..., offset=tail_start_offset)
            # in normal.py, so the offset genuinely affects its result too -- it's
            # not tail-fit-specific, and correctly belongs here for "mean" as well.
            lines.append(f"  tail_start_offset: {settings['tail_start_offset']}")

        for name in plugin["extra_args"]:
            lines.append(f"  {name}: {settings['extra'].get(name)}")

        lines.append(f"  save_overview: {settings['save_overview']}")
        lines.append(f"  save_results: {settings['save_results']}")
        lines.append(f"  save_quality_plots: {settings['save_quality_plots']}")
        lines.append(f"  save_irf_check: {settings['save_irf_check']}")

        if not is_mean and not settings["mono"]:
            lines.append(f"  fix_tau_short: {settings['fix_tau_short']}")
            if settings["fix_tau_short"]:
                value = settings["tau_short_value"]
                lines.append(f"  tau_short_value: {value if value is not None else 'auto-derived'}")

        lines.append("")
        with open(meta_path, "a") as f:
            f.write("\n".join(lines) + "\n")

    # -- log helpers ------------------------------------------------------------
    def _log(self, text):
        self.log_box.config(state="normal")
        self.log_box.insert("end", text)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _log_clear(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")
