"""
gui/curve_viewer_widget.py -- reusable Tkinter + matplotlib curve panel.

Mirrors the CurveViewerWidget / AFMPlotWidget pattern from softmech, but
stays in Tkinter/matplotlib so it can be embedded anywhere in FlimArc
without a toolkit conflict.

Usage (subclass pattern):

    class DecayPlotPanel(CurveViewerWidget):
        def __init__(self, parent):
            super().__init__(parent,
                title="Pixel decay",
                x_label="Time", x_unit="ns",
                y_label="Counts", y_unit="",
                log_y=True)

        def _refresh_plot(self):
            self.clear()
            self.add_curve(t_ns, decay, label="Raw decay")
            self.add_curve(t_ns, fit,   label="Fit",  style="--")
            self.add_vline(tail_start_ns, label="Tail start")
            self.finalize()

Public API for subclasses
--------------------------
clear()                         -- wipe axes, reset colour cycle
add_curve(x, y, **kw)          -- plot one named curve; returns Line2D
add_vline(x, **kw)             -- vertical reference line
add_fill_between(x, lo, hi)    -- confidence / uncertainty band
finalize(show_legend=True)      -- update legend and flush to canvas
set_title(text)                 -- update the axes title in place

Everything else (log/linear toggle, cursor readout, matplotlib toolbar,
CSV export button) is handled here; subclasses do not need to touch it.
"""

from __future__ import annotations

import csv
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np


# Fixed colour sequence: blue (raw), orange (fit), green (IRF),
# then extras.  Matches the existing plt.rcParams cycle order closely
# so users see consistent colours across the app.
_PALETTE = [
    "#1f77b4",  # blue   -- raw / full-range
    "#ff7f0e",  # orange -- fit overlay
    "#2ca02c",  # green  -- IRF
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
]


class CurveViewerWidget(tk.Frame):
    """
    Generic single-axis curve panel embedded in a Tkinter parent widget.

    Parameters
    ----------
    parent : tk.Widget
    title : str        Axes title.
    x_label, y_label : str   Axis labels (units appended automatically).
    x_unit, y_unit : str     Unit strings, e.g. "ns", "counts". "" = suppress.
    log_y : bool       Initial y-axis scale (user can toggle at runtime).
    figsize : (float, float)  Matplotlib figure size in inches.
    """

    def __init__(
        self,
        parent: tk.Widget,
        *,
        title: str = "",
        x_label: str = "x",
        y_label: str = "y",
        x_unit: str = "",
        y_unit: str = "",
        log_y: bool = False,
        figsize: tuple[float, float] = (6, 5),
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._x_label = x_label
        self._y_label = y_label
        self._x_unit = x_unit
        self._y_unit = y_unit
        self._log_y = tk.BooleanVar(value=log_y)

        self._colour_idx: int = 0
        self._lines: list[Line2D] = []
        # Mirrors what's plotted, for CSV export: [(label, x_array, y_array)]
        self._export_data: list[tuple[str, np.ndarray, np.ndarray]] = []

        self._build_figure(figsize)
        self._build_controls()
        self._build_status_bar()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_figure(self, figsize: tuple[float, float]) -> None:
        self._fig = Figure(figsize=figsize, tight_layout=True)
        self._ax = self._fig.add_subplot(111)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        tb_frame = ttk.Frame(self)
        tb_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self._nav = NavigationToolbar2Tk(self._canvas, tb_frame)
        self._nav.update()

        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._apply_axes_style()

    def _build_controls(self) -> None:
        """Thin toolbar above the figure: log toggle + export button."""
        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X, before=self._canvas.get_tk_widget())

        ttk.Checkbutton(
            bar, text="Log Y", variable=self._log_y,
            command=self._on_log_toggle,
        ).pack(side=tk.LEFT, padx=(4, 8))

        ttk.Button(
            bar, text="Export CSV…", command=self._on_export_csv,
        ).pack(side=tk.LEFT, padx=4)

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(value="")
        ttk.Label(
            self, textvariable=self._status_var,
            anchor=tk.W, font=("", 8), foreground="#555",
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=4, before=self._canvas.get_tk_widget())

    # ------------------------------------------------------------------
    # Axes styling
    # ------------------------------------------------------------------

    def _apply_axes_style(self) -> None:
        ax = self._ax
        ax.set_title(self._title, fontsize=10, pad=4)

        x_lbl = f"{self._x_label} ({self._x_unit})" if self._x_unit else self._x_label
        y_lbl = f"{self._y_label} ({self._y_unit})" if self._y_unit else self._y_label
        ax.set_xlabel(x_lbl, fontsize=9)
        ax.set_ylabel(y_lbl, fontsize=9)

        ax.set_yscale("log" if self._log_y.get() else "linear")
        ax.tick_params(labelsize=8)
        ax.grid(True, which="both", alpha=0.25, linestyle="--")

    def set_title(self, text: str) -> None:
        """Update the axes title without a full redraw."""
        self._title = text
        self._ax.set_title(text, fontsize=10, pad=4)

    # ------------------------------------------------------------------
    # Curve-building API  (call from _refresh_plot in subclasses)
    # ------------------------------------------------------------------

    def add_curve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        label: str = "",
        colour: Optional[str] = None,
        width: float = 1.5,
        style: str = "-",
        alpha: float = 1.0,
        zorder: int = 2,
    ) -> Line2D:
        """
        Plot one named curve and return its Line2D.

        On log-y axes, zeros are replaced with NaN so the line has gaps
        rather than crashing or clamping to 1 (the old ``hist + 1`` trick
        is NOT needed when using this method).
        """
        if colour is None:
            colour = _PALETTE[self._colour_idx % len(_PALETTE)]
            self._colour_idx += 1

        y_plot = np.where(y > 0, y, np.nan) if self._log_y.get() else np.asarray(y, float)

        (line,) = self._ax.plot(
            x, y_plot,
            color=colour, linewidth=width, linestyle=style,
            alpha=alpha, label=label, zorder=zorder,
        )
        self._lines.append(line)
        self._export_data.append((label, np.asarray(x), np.asarray(y)))
        return line

    def add_vline(
        self,
        x: float,
        *,
        label: str = "",
        colour: str = "#888888",
        style: str = "--",
        alpha: float = 0.7,
    ) -> None:
        """Vertical reference line (tail-fit start, IRF peak, …)."""
        self._ax.axvline(
            x, color=colour, linestyle=style, alpha=alpha,
            label=label, zorder=1,
        )

    def add_fill_between(
        self,
        x: np.ndarray,
        y_lo: np.ndarray,
        y_hi: np.ndarray,
        *,
        colour: str = "#1f77b4",
        alpha: float = 0.15,
    ) -> None:
        """Shaded confidence band (e.g. ± 1-sigma fit uncertainty)."""
        self._ax.fill_between(x, y_lo, y_hi, color=colour, alpha=alpha, zorder=1)

    def finalize(self, *, show_legend: bool = True) -> None:
        """
        Call after all add_curve() calls to update the legend and redraw.
        Must be the last call in every _refresh_plot() implementation.
        """
        handles = [l for l in self._ax.lines if l.get_label()]
        if show_legend and handles:
            self._ax.legend(fontsize=8, framealpha=0.7)
        self._canvas.draw_idle()

    def clear(self) -> None:
        """Remove all curves, reset the colour cycle, reapply axis style."""
        self._ax.cla()
        self._lines.clear()
        self._export_data.clear()
        self._colour_idx = 0
        self._apply_axes_style()

    # ------------------------------------------------------------------
    # Override point
    # ------------------------------------------------------------------

    def _refresh_plot(self) -> None:
        """
        Subclasses MUST override this.  Always call clear() first and
        finalize() last.  Example::

            def _refresh_plot(self):
                self.clear()
                self.add_curve(t, decay, label="Raw")
                self.add_curve(t, fit,   label="Fit", style="--")
                self.add_vline(tail_start, label="Tail start")
                self.finalize()
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_log_toggle(self) -> None:
        self._ax.set_yscale("log" if self._log_y.get() else "linear")
        self._refresh_plot()

    def _on_mouse_move(self, event) -> None:
        if event.inaxes is self._ax and event.xdata is not None:
            xu = f" {self._x_unit}" if self._x_unit else ""
            yu = f" {self._y_unit}" if self._y_unit else ""
            self._status_var.set(
                f"  x = {event.xdata:.4g}{xu}    y = {event.ydata:.4g}{yu}"
            )
        else:
            self._status_var.set("")

    def _on_export_csv(self) -> None:
        """Export all currently plotted curves to one CSV file."""
        if not self._export_data:
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        # All curves share the same x axis (same TCSPC time bins), so one
        # x column + one y column per curve is the right shape.
        xu = f" ({self._x_unit})" if self._x_unit else ""
        yu = f" ({self._y_unit})" if self._y_unit else ""
        header = [f"time_ns{xu}"] + [f"{lbl}{yu}" for lbl, _, _ in self._export_data]

        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            x_ref = self._export_data[0][1]
            ys = [y for _, _, y in self._export_data]
            for i, xi in enumerate(x_ref):
                writer.writerow([f"{xi:.6f}"] + [f"{y[i]:.6g}" for y in ys])
