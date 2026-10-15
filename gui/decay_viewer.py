"""
gui/decay_viewer.py -- per-pixel decay curve inspector.

Launched from the main window's "Decay viewer..." button (next to Open
File). Click any pixel/super-pixel in the intensity image to see its
fluorescence decay curve, optionally compare it across specific time
windows within the acquisition (raw photon-stream files only), and export
whatever's currently plotted to CSV.

Adapted from an original standalone viewer script: the pixel click /
super-pixel binning / time-window-comparison / CSV-export logic is carried
over essentially unchanged. What's different here is that this version is
a Toplevel embedded in the main app rather than its own Tk root, reuses the
main window's ALREADY-LOADED data (no separate file loader, no Open File
button of its own -- it already knows which file it's looking at), and its
own spatial binning is independent of whatever binning the main window's
Masking settings are currently set to (so you can inspect at a different
granularity than an actual analysis run would use).

The decay panel is now a DecayPlotPanel (a CurveViewerWidget subclass),
which provides the matplotlib figure, log/linear toggle, mouse-position
readout, and CSV export button.  DecayViewer only handles data wrangling,
pixel selection, and the image panel.
"""

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from analyses._common import build_group_cube
from gui.curve_viewer_widget import CurveViewerWidget


# ---------------------------------------------------------------------------
# Data helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _build_viewer_data(loaded_data, file_type):
    """Adapt the main window's already-loaded data -- a raw photon-stream
    dict from analyses._common.load_raw_with_time, or a histogram dict
    from flim_core.io.load_flim_data -- into the flat dict this viewer's
    plotting/pixel-decay helpers expect. Never reads the file from disk;
    everything here was already loaded when the main window opened it."""
    if file_type == "raw":
        d = loaded_data
        img, _ = build_group_cube(d, slice(None))
        return dict(img=img, x=d["x"], y=d["y"], micro=d["micro"], time_ms=d["time_ms"],
                    cube=None, tpb=d["tpb"], W=d["W"], H=d["H"], B0=d["B0"], B1=d["B1"], nb=d["nb"],
                    time_resolved=True)
    else:
        s = loaded_data
        return dict(img=s["img"], x=None, y=None, micro=None, time_ms=None,
                    cube=s["cube"], tpb=s["tpb"], W=s["W"], H=s["H"], B0=s["B0"], B1=s["B1"],
                    nb=s["cube"].shape[2], time_resolved=False)


def bin_image(img, bin_size):
    """Spatially bin an [H,W] image by summing bin_size x bin_size blocks.
    If H or W isn't evenly divisible by bin_size, the image is cropped
    (dropping the high-index edge rows/cols) to the nearest multiple
    first. Returns the binned image plus (Hc, Wc), the cropped shape it
    was computed from -- needed to map a block index back to the original
    pixel range it covers."""
    if bin_size <= 1:
        return img, img.shape
    H, W = img.shape
    Hc, Wc = (H // bin_size) * bin_size, (W // bin_size) * bin_size
    img_c = img[:Hc, :Wc]
    Hb, Wb = Hc // bin_size, Wc // bin_size
    binned = img_c.reshape(Hb, bin_size, Wb, bin_size).sum(axis=(1, 3))
    return binned, (Hc, Wc)


def whole_image_decay(s, t0_ms=None, t1_ms=None):
    """Decay histogram pooled over EVERY pixel in the sensor (not just one
    binned block) -- the whole-image equivalent of pixel_decay_block.
    Ignores self.bin_size entirely (binning only matters for picking a
    SPATIAL subset; summing everything makes that moot), so there's no
    bin_size argument here. Same two code paths as pixel_decay_block: raw
    photon-stream files filter the photon-level arrays directly (so time
    windows still work), exported histogram files just sum the whole cube
    (t0_ms/t1_ms must be None there, same restriction as per-pixel)."""
    if s["time_resolved"]:
        if t0_ms is not None and t1_ms is not None:
            mask = (s["time_ms"] >= t0_ms) & (s["time_ms"] < t1_ms)
            m = s["micro"][mask]
            n = int(mask.sum())
        else:
            m = s["micro"]
            n = int(len(m))
        hist = np.bincount(m - s["B0"], minlength=s["nb"])
    else:
        hist = s["cube"].astype(float).sum(axis=(0, 1))
        n = int(hist.sum())

    tn = np.arange(s["nb"]) * s["tpb"] * 1e9
    return tn, hist, n


def pixel_decay_block(s, xb, yb, bin_size, t0_ms=None, t1_ms=None):
    """Decay histogram for one BINNED super-pixel (xb, yb are block
    indices, each covering a bin_size x bin_size square of original
    pixels). Pools photons from every original pixel in that square --
    this is what makes a dead/no-signal camera pixel harmless: it just
    contributes 0 to the block instead of leaving a hole. Works for both
    raw photon-stream files (filters the photon-level arrays directly, so
    time windows still work) and exported histogram files (sums the
    pre-built cube over the block, where no per-photon timing exists so
    t0_ms/t1_ms must be None)."""
    x0, x1 = xb * bin_size, xb * bin_size + bin_size
    y0, y1 = yb * bin_size, yb * bin_size + bin_size

    if s["time_resolved"]:
        mask = (s["x"] >= x0) & (s["x"] < x1) & (s["y"] >= y0) & (s["y"] < y1)
        if t0_ms is not None and t1_ms is not None:
            mask &= (s["time_ms"] >= t0_ms) & (s["time_ms"] < t1_ms)
        m = s["micro"][mask]
        hist = np.bincount(m - s["B0"], minlength=s["nb"])
        n = int(mask.sum())
    else:
        hist = s["cube"][y0:y1, x0:x1, :].astype(float).sum(axis=(0, 1))
        n = int(hist.sum())

    tn = np.arange(s["nb"]) * s["tpb"] * 1e9
    return tn, hist, n


# ---------------------------------------------------------------------------
# Decay panel -- CurveViewerWidget subclass
# ---------------------------------------------------------------------------

class DecayPlotPanel(CurveViewerWidget):
    """
    FLIM-specific decay panel.

    The controller (DecayViewer) sets the public data attributes then calls
    refresh().  DecayPlotPanel never touches the file or the pixel selection
    logic -- that stays in DecayViewer._draw_decay().

    Attributes set by the controller before refresh()
    --------------------------------------------------
    t_ns : np.ndarray | None        nanosecond time axis
    curves : list[tuple]            [(label, counts_array), ...] in plot order
    windows_active : bool           True if time-window comparison is active
    """

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(
            parent,
            title="Select a pixel",
            x_label="Time",  y_label="Photon counts",
            x_unit="ns",     y_unit="",
            log_y=True,
            figsize=(6, 5),
        )
        self.t_ns: np.ndarray | None = None
        self.curves: list[tuple[str, np.ndarray]] = []

    def refresh(self) -> None:
        """Called by DecayViewer after updating t_ns / curves."""
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        self.clear()
        if self.t_ns is None or not self.curves:
            self.finalize(show_legend=False)
            return

        for label, hist in self.curves:
            self.add_curve(self.t_ns, hist, label=label)

        self.finalize()


# ---------------------------------------------------------------------------
# Main viewer window
# ---------------------------------------------------------------------------

class DecayViewer(tk.Toplevel):
    def __init__(self, master, loaded_data, file_type, filename):
        super().__init__(master)
        self.title(f"Decay viewer -- {filename}")
        self.geometry("1200x720")
        self.s = _build_viewer_data(loaded_data, file_type)
        self.selected_pixel = None    # (xb, yb) BLOCK indices of the currently selected super-pixel
        self.show_whole_image = False # True -> decay panel shows the whole-sensor pooled sum instead
        self.windows = []             # list of (t0_ms, t1_ms) comparison windows
        self.bin_size = 1             # current spatial binning factor (1 = no binning)
        self._binned_img = None       # img after binning, what's actually displayed
        self._Hb = self._Wb = None    # shape of the binned image (block-index bounds)
        self._build_ui(filename)
        self._recompute_binned_view()

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------

    def _build_ui(self, filename):
        # -- top bar: file info + binning controls --
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=8, pady=6)
        kind = "raw photon-stream" if self.s["time_resolved"] else "pre-computed histogram (no time windows)"
        ttk.Label(top, text=f"{filename}   [{kind}]", foreground="#333").pack(side="left")

        ttk.Label(top, text="   Bin (NxN):").pack(side="left")
        self.bin_var = tk.StringVar(value="1")
        ttk.Spinbox(top, from_=1, to=20, width=4, textvariable=self.bin_var).pack(side="left")
        ttk.Button(top, text="Apply binning", command=self.apply_binning).pack(side="left", padx=(4, 12))
        ttk.Button(top, text="Show whole-image sum", command=self.show_whole_image_sum).pack(
            side="left", padx=(0, 12))

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        # -- left: intensity image, clickable, with zoom/pan toolbar --
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Tip: use the zoom tool below to zoom in before "
                              "clicking -- pixels are small and easy to miss.",
                  wraplength=460, foreground="#555").pack(fill="x", padx=4, pady=(4, 0))
        self.fig_img = Figure(figsize=(5, 5))
        self.ax_img = self.fig_img.add_subplot(111)
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=left)
        self.canvas_img.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar_img = NavigationToolbar2Tk(self.canvas_img, left)
        self.toolbar_img.update()
        self.canvas_img.mpl_connect("button_press_event", self.on_pixel_click)
        self._img_im = None
        self._colorbar = None
        self._pixel_box = None

        # -- right: DecayPlotPanel + time-window controls + photon-count readout --
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        # Decay panel (CurveViewerWidget subclass) -- replaces the old bare
        # Figure + canvas + toolbar + Export button block.
        self._decay_panel = DecayPlotPanel(right)
        self._decay_panel.pack(fill="both", expand=True)

        # Time-window comparison controls
        ctrl = ttk.LabelFrame(right, text="Compare time windows (ms)")
        ctrl.pack(fill="x", padx=6, pady=6)
        ttk.Label(ctrl, text="Start:").grid(row=0, column=0, padx=2)
        self.t0_var = tk.StringVar(value="0")
        ttk.Entry(ctrl, textvariable=self.t0_var, width=8).grid(row=0, column=1)
        ttk.Label(ctrl, text="End:").grid(row=0, column=2, padx=2)
        self.t1_var = tk.StringVar(value="300")
        ttk.Entry(ctrl, textvariable=self.t1_var, width=8).grid(row=0, column=3)
        ttk.Button(ctrl, text="Add window", command=self.add_window).grid(row=0, column=4, padx=6)
        ttk.Button(ctrl, text="Clear windows", command=self.clear_windows).grid(row=0, column=5)
        if not self.s["time_resolved"]:
            ttk.Label(ctrl, text="(disabled -- this file has no per-photon timing)",
                     foreground="#a55", font=("", 8)).grid(row=2, column=0, columnspan=6, sticky="w", padx=2)

        self.window_list = tk.Listbox(ctrl, height=4)
        self.window_list.grid(row=1, column=0, columnspan=6, sticky="ew", pady=4, padx=2)

        # Photon-count readout for the selected pixel
        info = ttk.LabelFrame(right, text="Photon counts for selected pixel")
        info.pack(fill="x", padx=6, pady=(0, 6))
        self.info_text = tk.Text(info, height=6, wrap="word", state="disabled",
                                  background=self.cget("background"), relief="flat")
        self.info_text.pack(fill="x", padx=4, pady=4)
        self._update_info_text(None, None, [])

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------

    def apply_binning(self):
        """Re-bin the current view to a new NxN factor and redraw. Existing
        pixel selection and time windows are cleared since block indices
        mean something different at a new bin size."""
        try:
            n = int(self.bin_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Bin size must be a whole number.")
            return
        if n < 1:
            messagebox.showerror("Invalid input", "Bin size must be at least 1.")
            return
        self.bin_size = n
        self.selected_pixel = None
        self.show_whole_image = False
        self._recompute_binned_view()
        # Clear the decay panel
        self._decay_panel.t_ns = None
        self._decay_panel.curves = []
        self._decay_panel.refresh()
        self._update_info_text(None, None, [])

    def show_whole_image_sum(self):
        """Show the decay pooled over the ENTIRE sensor instead of one
        pixel/block -- independent of the current bin_size (summing
        everything makes spatial binning moot) and of any pixel selection,
        though the pixel selection itself is left alone so clicking back
        into pixel mode doesn't lose it."""
        self.show_whole_image = True
        self._update_pixel_highlight()   # clears the red selection box (nothing to highlight)
        self._draw_decay()

    def add_window(self):
        if not self.s["time_resolved"]:
            messagebox.showwarning(
                "Not available",
                "This file has no per-photon timing (it's a pre-summed export), "
                "so time-window comparison isn't possible -- only the raw "
                "photon-stream format preserves per-photon timing."
            )
            return
        try:
            t0 = float(self.t0_var.get())
            t1 = float(self.t1_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Start/End must be numbers (ms).")
            return
        if t1 <= t0:
            messagebox.showerror("Invalid range", "End must be greater than Start.")
            return

        self.windows.append((t0, t1))
        self.window_list.insert(tk.END, f"{t0:.0f}-{t1:.0f} ms")
        # Colour the listbox entry to match what the panel will draw.
        # _PALETTE is imported from curve_viewer_widget, index 0 = full range
        # so window colours start at index 1.
        from gui.curve_viewer_widget import _PALETTE
        colour = _PALETTE[(len(self.windows)) % len(_PALETTE)]
        self.window_list.itemconfig(tk.END, fg=colour)
        self._draw_decay()

    def clear_windows(self):
        self.windows = []
        self.window_list.delete(0, tk.END)
        self._draw_decay()

    # -----------------------------------------------------------------------
    # Decay drawing  (data wrangling only; rendering delegated to panel)
    # -----------------------------------------------------------------------

    def _draw_decay(self):
        """Compute the decay histogram(s) for the current selection and
        hand them to DecayPlotPanel for rendering."""
        if self.selected_pixel is None and not self.show_whole_image:
            self._decay_panel.t_ns = None
            self._decay_panel.curves = []
            self._decay_panel.set_title("Select a pixel")
            self._decay_panel.refresh()
            self._update_info_text(None, None, [])
            return

        info_lines = []
        curves = []

        if self.show_whole_image:
            decay_fn = lambda t0=None, t1=None: whole_image_decay(self.s, t0, t1)
            title = "Whole image"
            total_for_readout = float(self.s["img"].sum())
            readout_pixel = None
        else:
            xb, yb = self.selected_pixel
            b = self.bin_size
            decay_fn = lambda t0=None, t1=None: pixel_decay_block(self.s, xb, yb, b, t0, t1)
            title = f"Super-pixel block ({xb},{yb})" if b > 1 else f"Pixel ({xb},{yb})"
            total_for_readout = self._binned_img[yb, xb]
            readout_pixel = (xb, yb)

        tn, hist, n = decay_fn()
        curves.append((f"Full range (n={n})", hist))
        info_lines.append(("Full acquisition", n))

        if self.s["time_resolved"]:
            for (t0, t1) in self.windows:
                tn_w, hist_w, n_w = decay_fn(t0, t1)
                curves.append((f"{t0:.0f}-{t1:.0f} ms (n={n_w})", hist_w))
                info_lines.append((f"{t0:.0f}-{t1:.0f} ms window", n_w))
        elif self.windows:
            info_lines.append(("(time windows disabled for this file)", ""))

        self._decay_panel.t_ns = tn
        self._decay_panel.curves = curves
        self._decay_panel.set_title(f"{title} decay")
        self._decay_panel.refresh()

        self._update_info_text(readout_pixel, total_for_readout, info_lines)

    # -----------------------------------------------------------------------
    # Image panel (unchanged from original)
    # -----------------------------------------------------------------------

    def _recompute_binned_view(self):
        """Rebuild the binned intensity image for the current bin_size and
        redraw the image panel from scratch."""
        self._binned_img, (Hc, Wc) = bin_image(self.s["img"], self.bin_size)
        self._Hb, self._Wb = self._binned_img.shape
        if self.bin_size > 1:
            dropped_h, dropped_w = self.s["H"] - Hc, self.s["W"] - Wc
            note = f" (bin {self.bin_size}x{self.bin_size}"
            if dropped_h or dropped_w:
                note += f", cropped {dropped_h} row(s)/{dropped_w} col(s) at the edge to fit"
            note += ")"
        else:
            note = ""
        self._draw_image_base(note)

    def _draw_image_base(self, note=""):
        """Draw the (possibly binned) intensity image and its colorbar from
        scratch. Called on load and whenever binning changes -- NOT on
        every pixel click, since recreating the colorbar each time
        progressively shrinks the axes (colorbar() resizes its parent axes
        to make room for itself, and that resize is not undone by removing
        the old colorbar)."""
        self.fig_img.clf()
        self.ax_img = self.fig_img.add_subplot(111)
        self._pixel_box = None

        img = self._binned_img
        vmax = np.percentile(img, 99) if img.max() > 0 else 1
        self._img_im = self.ax_img.imshow(img, cmap="viridis", vmax=vmax,
                                           interpolation="nearest", aspect=2)
        self.ax_img.set_title(f"Click a super-pixel to see its decay{note}")
        self._colorbar = self.fig_img.colorbar(self._img_im, ax=self.ax_img, fraction=0.046, pad=0.04)
        label = "Photon counts (per super-pixel)" if self.bin_size > 1 else "Photon counts (per pixel)"
        self._colorbar.set_label(label)

        self.fig_img.tight_layout()
        self.canvas_img.draw()

    def _update_pixel_highlight(self):
        """Redraw only the red selection box, leaving the image/colorbar
        layout untouched. Called on every pixel click."""
        if self._pixel_box is not None:
            try:
                self._pixel_box.remove()
            except Exception:
                pass
            self._pixel_box = None
        if self.selected_pixel and not self.show_whole_image:
            xi, yi = self.selected_pixel
            self._pixel_box = Rectangle((xi - 0.5, yi - 0.5), 1, 1,
                                         fill=False, edgecolor="red", linewidth=1.5)
            self.ax_img.add_patch(self._pixel_box)
        self.canvas_img.draw_idle()

    def on_pixel_click(self, event):
        if event.inaxes != self.ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Ignore clicks that are actually zoom/pan drags from the toolbar --
        # otherwise every zoom-box drag would also (mis-)select a pixel.
        if self.toolbar_img.mode != "":
            return
        xb, yb = int(round(event.xdata)), int(round(event.ydata))
        if not (0 <= xb < self._Wb and 0 <= yb < self._Hb):
            return
        self.selected_pixel = (xb, yb)
        self.show_whole_image = False   # a pixel click always exits whole-image mode
        self._update_pixel_highlight()
        self._draw_decay()

    # -----------------------------------------------------------------------
    # Photon-count readout (unchanged from original)
    # -----------------------------------------------------------------------

    def _update_info_text(self, pixel, total_from_image, info_lines):
        """Refresh the photon-count readout panel below the decay plot."""
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", tk.END)
        if pixel is None and self.show_whole_image:
            lines = [f"Whole image (all {self.s['W']}x{self.s['H']} pixels)",
                     f"  Total photons (whole acquisition): {int(total_from_image)}"]
            for label, count in info_lines:
                lines.append(f"  {label}: {count} photons" if count != "" else f"  {label}")
            self.info_text.insert(tk.END, "\n".join(lines))
        elif pixel is None:
            self.info_text.insert(tk.END, "Click a pixel/super-pixel in the intensity image to see counts here, "
                                           "or use \"Show whole-image sum\" above.")
        else:
            xb, yb = pixel
            b = self.bin_size
            if b > 1:
                x0, x1 = xb * b, xb * b + b - 1
                y0, y1 = yb * b, yb * b + b - 1
                header = (f"Super-pixel block (x={xb}, y={yb}), bin {b}x{b}\n"
                          f"  covers original pixels x=[{x0}-{x1}], y=[{y0}-{y1}]")
            else:
                header = f"Pixel (x={xb}, y={yb})"
            lines = [header,
                     f"  Total photons in this block/pixel (whole acquisition): {int(total_from_image)}"]
            for label, count in info_lines:
                lines.append(f"  {label}: {count} photons" if count != "" else f"  {label}")
            self.info_text.insert(tk.END, "\n".join(lines))
        self.info_text.config(state="disabled")
