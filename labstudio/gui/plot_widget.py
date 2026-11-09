"""The "Plot" tab of LabStudio (module B).

Layout: left = dataset/trace manager (+ display units), centre = matplotlib
navigation toolbar + LabStudio tool toolbar + tool options + canvas + status line,
right = customisation panel, bottom = results table. Side panels sit in scroll
areas and can be hidden ("Hide side panels") so the tab fits a 1200×680 window.

Unit conventions
----------------
* Stored data are canonical (``labstudio.core.units.CANONICAL``).
* Each axis may show another *display* unit (``settings.x.unit`` …) — conversion
  happens only when drawing.
* :meth:`Trace.xy` returns **canonical, full-resolution** arrays (use them for analysis).
* The overlay API (``add_marker``, ``add_label``, ``add_vline``, ``add_hline``,
  ``add_line``, ``fill_between``) takes coordinates in **canonical units**;
  overlays are re-drawn automatically if the display unit changes.
  :meth:`current_span` also returns canonical units.
* Lines with many points are min/max-decimated for drawing only (view-aware).
* Waterfall offsets (``settings.offset_y`` …) are display-only: ``Trace.xy()`` /
  ``Trace.xy_display()`` and every tool use the original values.

Multi-panel figures (v1.1)
--------------------------
``set_layout(rows, cols)`` creates a grid of panels; each trace has a
``panel_index``. The single-axes API (``ax``, ``ax2``, ``add_marker`` …) targets
the *active* panel (``set_active_panel(i)``, default 0).
"""
from __future__ import annotations

import copy
import weakref
from dataclasses import dataclass, field
from typing import Any

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QSizePolicy, QSplitter, QVBoxLayout, QHBoxLayout, QWidget

from labstudio.core import units as U
from labstudio.core.dataset import Channel, Dataset, DatasetStore
from labstudio.core.provenance import provenance

from . import plot_math as PM
from . import plot_templates as TPL
from .plot_palettes import colormap_colors, palette_colors
from .plot_results import ResultsTable  # noqa: F401  (re-exported for backward compatibility)
from .plot_style import PlotSettings, cycle_style, preset_settings, settings_from_dict, style_axes

#: Above this many points a trace is decimated for display (analysis always uses all points).
DISPLAY_MAX_POINTS = 10000

AXIS_KEYS = ("x", "y", "y2")
SD_SUFFIXES = (" SD", "_SD", " sd", " std", " (SD)", " Std")
QWIDGETSIZE_MAX = 16777215


# =========================================================================== Trace


@dataclass(eq=False)
class Trace:
    """One plotted curve: column ``y`` versus column ``x`` of dataset ``ds``."""
    ds: Dataset
    x: str
    y: str
    label: str
    line: Line2D | None = None
    secondary_y: bool = False
    _panel: Any = field(default=None, repr=False)
    panel_index: int = 0                 # which subplot panel the trace is drawn in
    error_mode: str = "none"             # none | band | bars (needs a '<y> SD' column)
    error_every: int = 10                # error bar every N points

    # ---- data -------------------------------------------------------------
    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Canonical-unit, full-resolution (x, y) arrays — use these for analysis."""
        return self.ds.values(self.x), self.ds.values(self.y)

    def xy_display(self) -> tuple[np.ndarray, np.ndarray]:
        """Full-resolution (x, y) in the axes' current display units (no waterfall offset)."""
        p = self._panel() if self._panel is not None else None
        x, y = self.xy()
        if p is None:
            return x, y
        return (p._to_display(x, self.x_dim, self.x_unit, "x", self.panel_index),
                p._to_display(y, self.y_dim, self.y_unit, "y2" if self.secondary_y else "y", self.panel_index))

    @property
    def offset(self) -> tuple[float, float]:
        """Display-only waterfall offset (x, y) in display units."""
        return float(getattr(self, "_ox", 0.0)), float(getattr(self, "_oy", 0.0))

    @property
    def sd_column(self) -> str | None:
        """Name of a matching standard-deviation column ('<y> SD'), if the dataset has one."""
        cols = self.ds.data.columns
        for suf in SD_SUFFIXES:
            if self.y + suf in cols:
                return self.y + suf
        return None

    # ---- metadata ---------------------------------------------------------
    @property
    def x_dim(self) -> str:
        return self.ds.channel(self.x).dimension

    @property
    def y_dim(self) -> str:
        return self.ds.channel(self.y).dimension

    @property
    def x_unit(self) -> str:
        return self.ds.channel(self.x).unit

    @property
    def y_unit(self) -> str:
        return self.ds.channel(self.y).unit

    @property
    def visible(self) -> bool:
        return bool(self.line is not None and self.line.get_visible())

    @property
    def style(self) -> dict:
        """Current style as add_trace keywords."""
        ln = self.line
        if ln is None:
            return {}
        return dict(color=matplotlib.colors.to_hex(ln.get_color()), linestyle=ln.get_linestyle(),
                    linewidth=ln.get_linewidth(), marker=ln.get_marker(), markersize=ln.get_markersize(),
                    alpha=1.0 if ln.get_alpha() is None else ln.get_alpha())

    def __len__(self) -> int:
        return len(self.ds.data)


@dataclass(eq=False)
class _Overlay:
    kind: str
    data: dict
    kw: dict
    secondary: bool
    units: dict                  # axis key -> (dimension, unit) the coords were given in
    annotation: bool = False
    artists: list = field(default_factory=list)
    panel: int = 0
    trace: Any = None            # weakref to a Trace whose waterfall offset the overlay follows
    label_art: Any = None        # repelled label annotation (marker text / add_label)


@dataclass
class _PanelAxes:
    ax: Any
    ax2: Any = None


# =========================================================================== navigation toolbar


class _NavToolbar(NavigationToolbar2QT):
    """matplotlib toolbar without the subplot/figure-options dialogs (the
    customisation panel replaces them); Save opens LabStudio's export dialog."""

    toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] not in ("Subplots", "Customize")]

    def __init__(self, canvas, panel: "PlotPanel", parent=None):
        self._panel_ref = weakref.ref(panel)
        super().__init__(canvas, parent, coordinates=False)
        self.setIconSize(self.iconSize() * 0.85)

    def set_message(self, s):
        p = self._panel_ref()
        if p is not None and s and getattr(p, "tools", None) is not None and p.tools.mode in ("navigate", "span"):
            p.statusMessage.emit(s.replace("\n", "   "))

    def save_figure(self, *args):
        p = self._panel_ref()
        if p is not None:
            p.show_export_dialog()


def _scroll(w: QWidget, max_width: int | None = None) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidget(w)
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setMinimumWidth(180)
    if max_width:
        sa.setMaximumWidth(max_width)
    return sa


# =========================================================================== PlotPanel


class PlotPanel(QWidget):
    """Whole Plot tab. See module docstring for unit conventions."""

    resultsAdded = Signal(list)           # list[dict] rows (normalised keys)
    tracesChanged = Signal()
    settingsChanged = Signal()
    statusMessage = Signal(str)
    layoutChanged = Signal()

    def __init__(self, store: DatasetStore, parent=None):
        super().__init__(parent)
        if not matplotlib.rcParams.get("agg.path.chunksize"):
            matplotlib.rcParams["agg.path.chunksize"] = 20000
        self.store = store
        self.settings: PlotSettings = PlotSettings()
        self.traces: list[Trace] = []
        self._selected: Trace | None = None
        self._overlays: list[_Overlay] = []
        self._axes: list[_PanelAxes] = []
        self._active = 0
        self._span: tuple[float, float] | None = None       # display units of the active panel
        self._span_selector: SpanSelector | None = None
        self._full_res = False
        self._exporting = False
        self._label_redraws = 0
        self._colorbars: list[dict] = []    # {"cb", "cmap", "vmin", "vmax", "label", "ticks", "panel", "tag"}
        self._auto_cbars: list = []
        self._deco: list = []               # panel labels / offset labels (rebuilt on every apply)
        self.display_max_points = DISPLAY_MAX_POINTS

        self.figure = Figure(figsize=(8, 5.5), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(320, 220)
        self.figure.get_layout_engine().set(w_pad=0.08, h_pad=0.08)
        self._decim_timer = QTimer(self, singleShot=True, interval=40)
        self._decim_timer.timeout.connect(self._redecimate)
        self._build_axes()

        self._build_ui()
        self.canvas.mpl_connect("draw_event", self._on_draw_labels)
        self._make_span_selector()
        self._apply_default_template()
        self._apply_all(autoscale=True)
        self._store_cb = self._on_store_changed
        store.subscribe(self._store_cb)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        from .plot_customize import CustomizePanel
        from .plot_tools import ToolController
        from .plot_traces import TraceManager

        self.nav_toolbar = _NavToolbar(self.canvas, self, self)
        self.tools = ToolController(self)
        self.trace_manager = TraceManager(self)
        self.customize_panel = CustomizePanel(self)
        self.results = ResultsTable()
        self.status_label = QLabel("Ready")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_label.setWordWrap(False)
        self.status_label.setMinimumWidth(50)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.statusMessage.connect(self._set_status)

        self.canvas_scroll = QScrollArea()
        self.canvas_scroll.setWidget(self.canvas)
        self.canvas_scroll.setWidgetResizable(True)
        self.canvas_scroll.setAlignment(Qt.AlignCenter)
        self.canvas_scroll.setFrameShape(QFrame.NoFrame)
        self.canvas_scroll.setMinimumSize(200, 150)

        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(2, 2, 2, 2)
        cl.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(4)
        top.addWidget(self.nav_toolbar)
        top.addStretch(1)
        top.addWidget(self.tools.actions_bar)
        cl.addLayout(top)
        cl.addWidget(self.tools.toolbar)
        cl.addWidget(self.tools.options_scroll)
        cl.addWidget(self.canvas_scroll, 1)
        cl.addWidget(self.status_label)
        self.nav_toolbar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.nav_toolbar.setMinimumWidth(120)
        self.tools.actions_bar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.tools.actions_bar.setMinimumWidth(120)
        self.tools.toolbar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.tools.toolbar.setMinimumWidth(150)
        centre.setMinimumWidth(300)

        self.left_scroll = _scroll(self.trace_manager, 420)
        self.right_scroll = _scroll(self.customize_panel, 420)
        self.h_split = QSplitter(Qt.Horizontal)
        self.h_split.addWidget(self.left_scroll)
        self.h_split.addWidget(centre)
        self.h_split.addWidget(self.right_scroll)
        self.h_split.setStretchFactor(0, 0)
        self.h_split.setStretchFactor(1, 1)
        self.h_split.setStretchFactor(2, 0)
        lw = min(420, max(270, self.trace_manager.minimumSizeHint().width() + 18))
        self.h_split.setSizes([lw, 900, 300])
        self.h_split.setChildrenCollapsible(False)

        self.v_split = QSplitter(Qt.Vertical)
        self.v_split.addWidget(self.h_split)
        self.v_split.addWidget(self.results)
        self.v_split.setStretchFactor(0, 1)
        self.v_split.setSizes([700, 160])
        self.v_split.setChildrenCollapsible(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.v_split)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setToolTip(text)
        if text.startswith("⚠"):
            self.status_label.setStyleSheet("color:#5A3B00; background:#FFE3A3; padding:2px 4px;")
        else:
            self.status_label.setStyleSheet("color:#444; padding:2px 4px;")

    def warn(self, text: str) -> None:
        """Show an amber warning in the status line."""
        self.statusMessage.emit(text if text.startswith("⚠") else "⚠ " + text)

    def side_panels_visible(self) -> bool:
        return not self.left_scroll.isHidden()

    def set_side_panels_visible(self, visible: bool) -> None:
        """Show/hide the trace manager and customisation panel (more room for the canvas)."""
        self.left_scroll.setVisible(bool(visible))
        self.right_scroll.setVisible(bool(visible))
        act = getattr(self.tools, "act_side", None)
        if act is not None and act.isChecked() == bool(visible):
            act.blockSignals(True)
            act.setChecked(not visible)
            act.blockSignals(False)

    # ------------------------------------------------------------------ axes / panels
    def _build_axes(self) -> None:
        s = self.settings
        rows, cols = max(1, int(s.layout_rows)), max(1, int(s.layout_cols))
        gs = self.figure.add_gridspec(rows, cols)
        self._axes = []
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            kw = {}
            if i > 0 and s.share_x:
                kw["sharex"] = self._axes[0].ax
            if i > 0 and s.share_y:
                kw["sharey"] = self._axes[0].ax
            a = self.figure.add_subplot(gs[r, c], **kw)
            self._connect_lim_callbacks(a)
            self._axes.append(_PanelAxes(a))
            s.panel(i)
        self._active = min(self._active, len(self._axes) - 1)
        self._built_sig = self._layout_signature()

    @property
    def n_panels(self) -> int:
        return len(self._axes)

    @property
    def active_panel(self) -> int:
        return self._active

    def panel_axes(self, i: int | None = None):
        """Primary Axes of panel ``i`` (default: active panel)."""
        return self._axes[self._pi(i)].ax

    def _pi(self, panel: int | None) -> int:
        if panel is None:
            return self._active
        return max(0, min(int(panel), len(self._axes) - 1))

    @property
    def ax(self):
        """Primary Axes of the active panel."""
        return self._axes[self._active].ax

    @property
    def _ax2(self):
        """Secondary axes of the active panel or None (read-only)."""
        return self._axes[self._active].ax2

    @property
    def ax2(self):
        """Secondary (twin) y-axes of the active panel, created on first access."""
        return self._get_ax2(self._active)

    def _get_ax2(self, i: int):
        P = self._axes[i]
        if P.ax2 is None:
            P.ax2 = P.ax.twinx()
            P.ax2.set_zorder(P.ax.get_zorder() + 1)
            P.ax2.patch.set_visible(False)
            self._connect_lim_callbacks(P.ax2)
            if i == self._active:
                self._make_span_selector()
        return P.ax2

    def _top_axes(self):
        P = self._axes[self._active]
        return P.ax2 if P.ax2 is not None else P.ax

    def _axes_for(self, secondary: bool, panel: int | None = None):
        i = self._pi(panel)
        return self._get_ax2(i) if secondary else self._axes[i].ax

    def _drop_ax2_if_unused(self) -> None:
        for i, P in enumerate(self._axes):
            if P.ax2 is None or any(t.secondary_y and t.panel_index == i for t in self.traces):
                continue
            for ov in [o for o in self._overlays if o.secondary and o.panel == i]:
                self._remove_overlay_artists(ov)
                self._overlays.remove(ov)
            if i == self._active:
                self._release_span_selector()
            P.ax2.remove()
            P.ax2 = None
            if i == self._active:
                self._make_span_selector()

    def _connect_lim_callbacks(self, a) -> None:
        a.callbacks.connect("xlim_changed", lambda *_: self._decim_timer.start())
        a.callbacks.connect("ylim_changed", lambda *_: self._decim_timer.start())

    def set_active_panel(self, i: int) -> None:
        """Make panel ``i`` the target of ``ax``/``ax2``/overlays/span/customisation."""
        i = self._pi(i)
        if i == self._active:
            return
        self._active = i
        self._span = None
        self._make_span_selector()
        self.trace_manager.refresh_units()
        self.customize_panel.sync_from_settings()
        self.redraw()

    def set_layout(self, rows: int, cols: int, *, share_x: bool | None = None, share_y: bool | None = None) -> None:
        """Arrange the figure as ``rows × cols`` panels (traces in removed panels move to panel 0)."""
        s = self.settings
        s.layout_rows, s.layout_cols = max(1, min(6, int(rows))), max(1, min(6, int(cols)))
        if share_x is not None:
            s.share_x = bool(share_x)
        if share_y is not None:
            s.share_y = bool(share_y)
        self._rebuild_axes()

    def _rebuild_axes(self) -> None:
        saved = [(t, t.style, t.visible) for t in self.traces]
        for t in self.traces:
            self._remove_error_artists(t)
            if t.line is not None:
                t.line.remove()
                t.line = None
        for ov in self._overlays:
            self._remove_overlay_artists(ov)
        self._remove_deco()
        self._remove_auto_colorbars()
        for spec in self._colorbars:
            self._remove_cb(spec.get("cb"))
            spec["cb"] = None
        self._release_span_selector()
        for P in self._axes:
            if P.ax2 is not None:
                P.ax2.remove()
            P.ax.remove()
        self._axes = []
        self._span = None
        self._build_axes()
        n = len(self._axes)
        for t, st, vis in saved:
            if t.panel_index >= n:
                t.panel_index = 0
            a = self._axes_for(t.secondary_y, t.panel_index)
            (t.line,) = a.plot([], [], label=t.label, zorder=2, **st)
            t.line.set_gid("trace")
            t.line.set_visible(vis)
        for ov in self._overlays:
            if ov.panel >= n:
                ov.panel = 0
            self._render_overlay(ov)
        for spec in self._colorbars:
            spec["panel"] = min(spec["panel"], n - 1)
            spec["cb"] = self._make_cb(spec)
        self._make_span_selector()
        self._validate_units()
        self._refresh_all_data()
        self._apply_all(autoscale=True)
        self.trace_manager.refresh_traces()
        self.customize_panel.sync_from_settings()
        self.layoutChanged.emit()
        self.tracesChanged.emit()

    def set_trace_panel(self, trace: Trace, panel: int) -> None:
        """Move ``trace`` to subplot panel ``panel``."""
        panel = self._pi(panel)
        if trace.panel_index == panel:
            return
        st, vis = trace.style, trace.visible
        self._remove_error_artists(trace)
        trace.line.remove()
        trace.panel_index = panel
        a = self._axes_for(trace.secondary_y, panel)
        (trace.line,) = a.plot([], [], label=trace.label, zorder=2, **st)
        trace.line.set_gid("trace")
        trace.line.set_visible(vis)
        self._drop_ax2_if_unused()
        self._validate_units()
        self._refresh_all_data()
        self._after_traces_changed(autoscale=True)

    # ------------------------------------------------------------------ units
    def axis_dimension(self, key: str, panel: int | None = None) -> str | None:
        """Dimension of the first trace plotted on axis ``key`` ('x', 'y', 'y2') of a panel."""
        i = self._pi(panel)
        shared = (key == "x" and self.settings.share_x) or (key == "y" and self.settings.share_y)
        for t in self.traces:
            if t.panel_index != i and not shared:
                continue
            if key == "x":
                return t.x_dim
            if (key == "y2") == t.secondary_y:
                return t.y_dim
        return None

    def _axis_settings(self, key: str, panel: int | None = None):
        i = self._pi(panel)
        if (key == "x" and self.settings.share_x) or (key == "y" and self.settings.share_y):
            i = 0
        ps = self.settings.panel(i)
        return {"x": ps.x, "y": ps.y, "y2": ps.y2}[key]

    def display_unit(self, key: str, panel: int | None = None) -> str | None:
        """Current display unit of axis ``key`` (canonical unless overridden)."""
        dim = self.axis_dimension(key, panel)
        if dim is None:
            return None
        u = self._axis_settings(key, panel).unit
        if u and dim in U.dimensions() and u in U.units_for(dim):
            return u
        return U.CANONICAL.get(dim)

    def set_display_unit(self, key: str, unit: str | None, panel: int | None = None) -> None:
        """Show axis ``key`` in ``unit`` (None → canonical). Data are not modified."""
        dim = self.axis_dimension(key, panel)
        if unit and dim and unit not in U.units_for(dim):
            raise ValueError(f"'{unit}' is not a unit of {dim}")
        axs = self._axis_settings(key, panel)
        if axs.unit == unit:
            return
        span = self.current_span() if key == "x" and self._pi(panel) == self._active else None
        axs.unit = unit
        axs.auto_limits = True
        if span is not None:
            self.set_span(*span)
        self._refresh_all_data()
        self._rerender_overlays()
        self._apply_all(autoscale=True)
        self.trace_manager.refresh_units()
        self.customize_panel.sync_from_settings()

    def _to_display(self, values, dim: str, unit: str, key: str, panel: int | None = None):
        """Convert ``values`` (in ``unit`` of ``dim``) into axis ``key``'s display unit."""
        target = self.display_unit(key, panel)
        if not target or target == unit or dim != self.axis_dimension(key, panel):
            return values
        try:
            return U.convert(values, unit, target, dim)
        except ValueError:
            return values

    def _from_display(self, values, key: str, panel: int | None = None):
        dim = self.axis_dimension(key, panel)
        du = self.display_unit(key, panel)
        if not dim or not du or du == U.CANONICAL.get(dim):
            return values
        return U.convert(values, du, U.CANONICAL[dim], dim)

    # ------------------------------------------------------------------ traces
    def add_trace(self, ds: Dataset, x: str, y: str, label: str | None = None, *, secondary_y: bool = False,
                  panel: int | None = None, **style) -> Trace:
        """Plot column ``y`` vs ``x`` of ``ds`` in ``panel`` (default: active panel).
        ``style``: color, linestyle, linewidth, marker, markersize, alpha (any other
        Line2D keyword is passed through)."""
        for c in (x, y):
            if c not in ds.data.columns:
                raise KeyError(f"Column '{c}' not in dataset '{ds.name}'")
        if label is None:
            same = [t for t in self.traces if t.ds is ds]
            label = ds.name if not same else f"{ds.name} – {y}"
        idx = self._next_style_index()
        color, ls = cycle_style(idx, self.settings.color_cycle)
        st = dict(color=color, linestyle=ls, linewidth=self.settings.linewidth, marker="None",
                  markersize=self.settings.markersize, alpha=1.0)
        st.update({k: v for k, v in style.items() if v is not None})
        t = Trace(ds, x, y, label, None, bool(secondary_y), weakref.ref(self), self._pi(panel))
        a = self._axes_for(t.secondary_y, t.panel_index)
        (t.line,) = a.plot([], [], label=label, zorder=2, **st)
        t.line.set_gid("trace")
        self.traces.append(t)
        self._validate_units()
        if self._offsets_active():
            self._refresh_all_data()
        else:
            self._update_trace_data(t)
        self._selected = t
        self._after_traces_changed(autoscale=True)
        return t

    def _next_style_index(self) -> int:
        used = {matplotlib.colors.to_hex(t.line.get_color()).lower() + str(t.line.get_linestyle())
                for t in self.traces}
        for i in range(200):
            c, ls = cycle_style(i, self.settings.color_cycle)
            if matplotlib.colors.to_hex(c).lower() + ls not in used:
                return i
        return len(self.traces)

    def remove_trace(self, trace: Trace) -> None:
        """Remove ``trace`` from the plot (its dataset stays in the store)."""
        if trace not in self.traces:
            return
        self._remove_error_artists(trace)
        if trace.line is not None:
            trace.line.remove()
        self.traces.remove(trace)
        if self._selected is trace:
            self._selected = self.traces[-1] if self.traces else None
        self._drop_ax2_if_unused()
        self._validate_units()
        if self._offsets_active():
            self._refresh_all_data()
        self._after_traces_changed(autoscale=True)

    def clear(self) -> None:
        """Remove all traces, overlays, annotations, colorbars and the span."""
        for t in self.traces:
            self._remove_error_artists(t)
            if t.line is not None:
                t.line.remove()
        self.traces.clear()
        self._selected = None
        self.remove_colorbars()
        self.clear_overlays(include_annotations=True)
        self.clear_span()
        self._drop_ax2_if_unused()
        self._after_traces_changed(autoscale=True)

    def selected_trace(self) -> Trace | None:
        """Trace selected in the trace list (defaults to the most recently added)."""
        if self._selected in self.traces:
            return self._selected
        return self.traces[-1] if self.traces else None

    def selected_traces(self) -> list[Trace]:
        """All traces selected in the trace list (multi-selection), else [selected_trace()]."""
        rows = sorted({i.row() for i in self.trace_manager.list.selectedIndexes()})
        out = [self.traces[r] for r in rows if 0 <= r < len(self.traces)]
        if not out and self.selected_trace() is not None:
            out = [self.selected_trace()]
        return out

    def select_trace(self, trace: Trace | None) -> None:
        self._selected = trace
        self.trace_manager.refresh_traces()
        self.tools.on_selection_changed()

    def set_trace_style(self, trace: Trace, **style) -> None:
        """Update style (color, linestyle, linewidth, marker, markersize, alpha, label)."""
        if "label" in style:
            trace.label = str(style.pop("label"))
            trace.line.set_label(trace.label)
        if style:
            trace.line.set(**style)
            if "color" in style:
                self._update_error_artists(trace)
        self._after_traces_changed(autoscale=False)

    def set_trace_visible(self, trace: Trace, visible: bool) -> None:
        trace.line.set_visible(bool(visible))
        for a in getattr(trace, "_err", []):
            a.set_visible(bool(visible))
        self.tools.invalidate_cache()
        if self._offsets_active():
            self._refresh_all_data()
        self._after_traces_changed(autoscale=self._any_auto())

    def set_trace_secondary(self, trace: Trace, secondary: bool) -> None:
        """Move ``trace`` to the secondary (right) or primary y-axis of its panel."""
        if trace.secondary_y == bool(secondary):
            return
        st, vis = trace.style, trace.visible
        self._remove_error_artists(trace)
        trace.line.remove()
        trace.secondary_y = bool(secondary)
        a = self._axes_for(trace.secondary_y, trace.panel_index)
        (trace.line,) = a.plot([], [], label=trace.label, zorder=2, **st)
        trace.line.set_gid("trace")
        trace.line.set_visible(vis)
        self._drop_ax2_if_unused()
        self._validate_units()
        self._refresh_all_data()
        self._after_traces_changed(autoscale=True)

    def set_trace_error(self, trace: Trace, mode: str = "band", every: int | None = None) -> None:
        """Error display from the dataset's '<y> SD' column: 'none', 'band' (shaded ±SD) or 'bars'."""
        if mode not in ("none", "band", "bars"):
            raise ValueError("mode must be 'none', 'band' or 'bars'")
        trace.error_mode = mode
        if every:
            trace.error_every = max(1, int(every))
        self._update_error_artists(trace)
        self._after_traces_changed(autoscale=False)

    def move_trace(self, trace: Trace, delta: int) -> None:
        i = self.traces.index(trace)
        j = max(0, min(len(self.traces) - 1, i + delta))
        if i == j:
            return
        self.traces.insert(j, self.traces.pop(i))
        for k, t in enumerate(self.traces):
            t.line.set_zorder(2 + k * 1e-3)
        if self._offsets_active():
            self._refresh_all_data()
        self._after_traces_changed(autoscale=False)

    def apply_style_preset(self, name: str) -> None:
        """Switch to a preset from ``plot_style.PRESET_NAMES`` (also re-widths existing traces)."""
        self.settings = preset_settings(name, self.settings)
        for t in self.traces:
            t.line.set_linewidth(self.settings.linewidth)
        self._after_traces_changed(autoscale=False)
        self.customize_panel.sync_from_settings()

    def _validate_units(self) -> None:
        for i in range(len(self._axes)):
            for key in AXIS_KEYS:
                axs = self._axis_settings(key, i)
                dim = self.axis_dimension(key, i)
                if axs.unit and (not dim or axs.unit not in U.units_for(dim)):
                    axs.unit = None

    def _after_traces_changed(self, autoscale: bool) -> None:
        self.tools.invalidate_cache()
        self._apply_all(autoscale=autoscale)
        self.trace_manager.refresh_traces()
        self.tracesChanged.emit()

    def _any_auto(self) -> bool:
        return any(self._axis_settings(k, i).auto_limits for i in range(len(self._axes)) for k in AXIS_KEYS)

    # ------------------------------------------------------------------ dataset management
    def rename_dataset(self, ds: Dataset, name: str) -> None:
        """Rename ``ds`` in the store (trace labels derived from the old name follow)."""
        name = str(name).strip()
        if not name or name == ds.name:
            return
        old = ds.name
        for t in self.traces:
            if t.ds is ds and (t.label == old or t.label.startswith(old + " – ")):
                t.label = name + t.label[len(old):]
                t.line.set_label(t.label)
        self.store.rename(ds, name)

    def remove_dataset(self, ds: Dataset) -> None:
        """Remove ``ds`` from the store (its traces disappear from the plot)."""
        self.store.remove(ds)

    def move_dataset(self, ds: Dataset, index: int) -> None:
        """Move ``ds`` to position ``index`` in the store."""
        self.store.move(ds, index)

    def duplicate_dataset(self, ds: Dataset) -> Dataset:
        """Add an independent copy of ``ds`` (data, channels, metadata) to the store."""
        new = Dataset(name=f"{ds.name} (copy)", data=ds.data.copy(),
                      channels={k: Channel(**vars(v)) for k, v in ds.channels.items()}, kind=ds.kind,
                      source_path=ds.source_path, metadata=copy.deepcopy(ds.metadata), default_x=ds.default_x,
                      default_y=ds.default_y)
        self.store.add(new)
        return new

    # ------------------------------------------------------------------ waterfall offsets
    def _offsets_active(self) -> bool:
        s = self.settings
        return bool(s.offset_y) or bool(s.offset_x)

    def set_offsets(self, y: float = 0.0, *, mode: str = "absolute", x: float = 0.0,
                    labels: bool | None = None) -> None:
        """Waterfall: shift the k-th visible trace of each axis by k·(x, y) for display only.
        ``mode='percent'`` gives ``y`` as % of the y range of the panel."""
        if mode not in ("absolute", "percent"):
            raise ValueError("mode must be 'absolute' or 'percent'")
        s = self.settings
        s.offset_y, s.offset_mode, s.offset_x = float(y), mode, float(x)
        if labels is not None:
            s.offset_labels = bool(labels)
        self._refresh_all_data()
        self._rerender_overlays()
        self._apply_all(autoscale=True)
        self.customize_panel.sync_from_settings()

    def _compute_offsets(self) -> None:
        s = self.settings
        for t in self.traces:
            t._ox = t._oy = 0.0
        if not self._offsets_active():
            return
        groups: dict[tuple[int, bool], list[Trace]] = {}
        for t in self.traces:
            if t.visible:
                groups.setdefault((t.panel_index, t.secondary_y), []).append(t)
        for grp in groups.values():
            dy = s.offset_y
            if s.offset_mode == "percent":
                lo, hi = np.inf, -np.inf
                for t in grp:
                    a = getattr(t, "_dy", None)
                    if a is None:
                        continue
                    a = a[np.isfinite(a)]
                    if a.size:
                        lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
                dy = s.offset_y / 100.0 * (hi - lo) if hi > lo else 0.0
            for k, t in enumerate(grp):
                t._ox, t._oy = k * s.offset_x, k * dy

    # ------------------------------------------------------------------ data → lines
    def _update_trace_data(self, t: Trace, use_view: bool = False, recompute: bool = True) -> None:
        if recompute or getattr(t, "_dx", None) is None:
            dx, dy = t.xy_display()
            t._dx, t._dy = dx, dy  # cached full-resolution display arrays (no offset)
        dx, dy = t._dx, t._dy
        ox, oy = t.offset
        if ox or oy:
            dx, dy = dx + ox, dy + oy
        n = dx.size
        if self._full_res or n <= self.display_max_points:
            t.line.set_data(dx, dy)
        else:
            a = t.line.axes
            lims = (a.get_xlim(), a.get_ylim()) if use_view else (None, None)
            idx = PM.display_indices(dx, dy, lims[0], lims[1], self.display_max_points)
            t.line.set_data(PM.take_with_breaks(dx, idx), PM.take_with_breaks(dy, idx))
        if t.error_mode != "none" and not use_view:
            self._update_error_artists(t)

    def _refresh_all_data(self) -> None:
        for t in self.traces:
            t._dx, t._dy = t.xy_display()
        self._compute_offsets()
        for t in self.traces:
            self._update_trace_data(t, recompute=False)
        self.tools.invalidate_cache()

    def _redecimate(self) -> None:
        big = [t for t in self.traces if len(t) > self.display_max_points]
        if not big or self._full_res:
            return
        for t in big:
            self._update_trace_data(t, use_view=True, recompute=False)
        self.canvas.draw_idle()

    def _remove_error_artists(self, t: Trace) -> None:
        for a in getattr(t, "_err", []):
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        t._err = []

    def _update_error_artists(self, t: Trace) -> None:
        self._remove_error_artists(t)
        col = t.sd_column
        if t.error_mode == "none" or col is None or t.line is None:
            return
        x, y = t.xy()
        sd = t.ds.values(col)
        key = "y2" if t.secondary_y else "y"
        X = t._dx if getattr(t, "_dx", None) is not None else t.xy_display()[0]
        lo = self._to_display(y - sd, t.y_dim, t.y_unit, key, t.panel_index)
        hi = self._to_display(y + sd, t.y_dim, t.y_unit, key, t.panel_index)
        ox, oy = t.offset
        X, lo, hi = X + ox, lo + oy, hi + oy
        a = t.line.axes
        color = t.line.get_color()
        if t.error_mode == "band":
            step = max(1, X.size // 5000)
            art = a.fill_between(X[::step], lo[::step], hi[::step], color=color, alpha=0.22, linewidth=0,
                                 zorder=1.5, label="_err")
        else:
            idx = np.arange(0, X.size, max(1, int(t.error_every)))
            art = a.vlines(X[idx], lo[idx], hi[idx], colors=color, linewidth=0.9, zorder=1.9, label="_err")
        art.set_visible(t.visible)
        t._err = [art]

    def _data_limits(self, panel: int | None = None) -> dict:
        i = self._pi(panel)
        lims: dict[str, tuple[float, float] | None] = {}
        for key in AXIS_KEYS:
            axs = self._axis_settings(key, i)
            vals = []
            for t in self.traces:
                if not t.visible or t.panel_index != i:
                    continue
                ox, oy = t.offset
                if key == "x":
                    a, off = getattr(t, "_dx", None), ox
                elif (key == "y2") == t.secondary_y:
                    a, off = getattr(t, "_dy", None), oy
                else:
                    continue
                if a is None or a.size == 0:
                    continue
                a = a[np.isfinite(a)] + off
                if axs.scale == "log":
                    a = a[a > 0]
                if a.size:
                    vals.append((float(a.min()), float(a.max())))
                if key != "x" and t.error_mode != "none" and t.sd_column:
                    for art in getattr(t, "_err", []):
                        try:
                            bb = art.get_datalim(art.axes.transData)
                            if np.all(np.isfinite([bb.y0, bb.y1])):
                                vals.append((float(bb.y0), float(bb.y1)))
                        except (AttributeError, ValueError):
                            pass
            if vals:
                lims[key] = (min(v[0] for v in vals), max(v[1] for v in vals))
        return lims

    def auto_label(self, key: str, panel: int | None = None) -> str:
        """Automatic axis label from the channels plotted on axis ``key`` of a panel."""
        i = self._pi(panel)
        unit = self.display_unit(key, i)
        names, dims = [], []
        for t in self.traces:
            if t.panel_index != i:
                continue
            if key == "x":
                dim, name = t.x_dim, t.ds.channel(t.x).name
            elif (key == "y2") == t.secondary_y:
                dim, name = t.y_dim, t.ds.channel(t.y).name
            else:
                continue
            if name not in names:
                names.append(name)
            if dim not in dims:
                dims.append(dim)
        if not names:
            return ""
        if len(names) == 1:
            return U.axis_label(names[0], unit)
        if len(dims) == 1:
            return U.axis_label(U.DIMENSION_LABELS.get(dims[0], names[0]), unit)
        return ", ".join(names)

    def _view_settings(self, i: int) -> PlotSettings:
        s = copy.copy(self.settings)
        s.title = self.settings.panel(i).title
        s.x, s.y, s.y2 = (self._axis_settings(k, i) for k in AXIS_KEYS)
        return s

    def _apply_all(self, autoscale: bool = False) -> None:
        """Apply settings to the figure (labels, fonts, limits, legend…) and redraw."""
        big = [t for t in self.traces if len(t) > self.display_max_points]
        if autoscale and big:
            for t in big:
                self._update_trace_data(t, use_view=False, recompute=False)
        self._remove_deco()
        self._remove_auto_colorbars()
        self._restyle_colorbars()
        s = self.settings
        n = len(self._axes)
        all_lims = [self._data_limits(i) if any(t.panel_index == i for t in self.traces) else {}
                    for i in range(n)]
        for key, shared in (("x", s.share_x), ("y", s.share_y)):
            if shared and n > 1:
                vals = [lm[key] for lm in all_lims if lm.get(key)]
                if vals:
                    u = (min(v[0] for v in vals), max(v[1] for v in vals))
                    for lm in all_lims:
                        lm[key] = u
        rows, cols = max(1, s.layout_rows), max(1, s.layout_cols)
        for i, P in enumerate(self._axes):
            labels = {k: self.auto_label(k, i) for k in AXIS_KEYS}
            handles = [t.line for t in self.traces if t.panel_index == i] + [
                a for ov in self._overlays if ov.kind == "line" and ov.panel == i for a in ov.artists
                if isinstance(a, Line2D) and not str(a.get_label()).startswith("_")]
            handles = self._legend_handles(handles, i)
            vs = self._view_settings(i)
            style_axes(self.figure, P.ax, P.ax2, vs, auto_labels=labels, data_limits=all_lims[i],
                       legend_handles=handles, autoscale=autoscale)
            r, c = divmod(i, cols)
            if n > 1 and s.share_x and r < rows - 1 and i + cols < n:
                P.ax.tick_params(axis="x", which="both", labelbottom=False)
                P.ax.set_xlabel("")
            elif n > 1:
                P.ax.tick_params(axis="x", which="both", labelbottom=True)
            if n > 1 and s.share_y and c > 0:
                P.ax.tick_params(axis="y", which="both", labelleft=False)
                P.ax.set_ylabel("")
            elif n > 1:
                P.ax.tick_params(axis="y", which="both", labelleft=True)
        self._update_auto_colorbars()
        self._draw_panel_labels()
        self._draw_offset_labels()
        if self._span_selector is not None and self._span is not None:
            self._span_selector.extents = self._span
        self.redraw()
        if big:
            self._decim_timer.start()

    def apply_settings(self) -> None:
        """Re-apply ``self.settings`` (call after changing it programmatically)."""
        if self._layout_signature() != getattr(self, "_built_sig", None):
            self._rebuild_axes()
        self._validate_units()
        self._refresh_all_data()
        self._rerender_overlays()
        self._apply_all(autoscale=True)
        self._sync_preview()
        self.customize_panel.sync_from_settings()
        self.trace_manager.refresh_units()
        self.settingsChanged.emit()

    def _layout_signature(self) -> tuple:
        s = self.settings
        return (s.layout_rows, s.layout_cols, s.share_x, s.share_y)

    def redraw(self) -> None:
        """Schedule a canvas redraw."""
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ legend modes / colorbars
    def _legend_handles(self, handles: list, panel: int) -> list:
        mode = self.settings.legend_mode
        if mode in ("full", "off") or len(handles) < 13:
            return handles
        labels = [str(h.get_label()) for h in handles]
        groups = PM.legend_groups(labels, 13)
        if not groups:
            return handles
        has_cb = any(sp["panel"] == panel for sp in self._colorbars)
        drop, ell = set(), {}
        for g in groups:
            if mode == "colorbar" or has_cb:
                drop.update(g)
            else:
                drop.update(g[1:-1])
                ell[g[0]] = Line2D([], [], linestyle="None", marker="None", label=f"⋮  ({len(g) - 2} more)")
        out = []
        for k, h in enumerate(handles):
            if k not in drop:
                out.append(h)
            if k in ell:
                out.append(ell[k])
        return out

    def _make_cb(self, spec: dict):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        norm = spec.get("norm") or Normalize(spec["vmin"], spec["vmax"])
        sm = ScalarMappable(norm=norm, cmap=spec["cmap"])
        P = self._axes[min(spec["panel"], len(self._axes) - 1)]
        axs = [P.ax] + ([P.ax2] if P.ax2 is not None else [])
        cb = self.figure.colorbar(sm, ax=axs, label=spec.get("label") or "", ticks=spec.get("ticks"),
                                  fraction=0.05, pad=0.02)
        from .plot_style import resolve_family
        s = self.settings
        fam = resolve_family(s.font_family)
        cb.ax.tick_params(labelsize=s.tick_size, labelfontfamily=fam, direction=s.tick_direction,
                          length=s.tick_major_len, width=s.tick_major_width)
        cb.set_label(spec.get("label") or "", fontsize=s.label_size, fontfamily=fam)
        cb.outline.set_linewidth(s.axes_linewidth)
        return cb

    def _restyle_colorbars(self) -> None:
        from .plot_style import resolve_family
        s = self.settings
        fam = resolve_family(s.font_family)
        for sp in self._colorbars:
            cb = sp.get("cb")
            if cb is None:
                continue
            cb.ax.tick_params(labelsize=s.tick_size, labelfontfamily=fam, direction=s.tick_direction,
                              length=s.tick_major_len, width=s.tick_major_width)
            cb.set_label(sp.get("label") or "", fontsize=s.label_size, fontfamily=fam)
        for ov in self._overlays:
            if ov.label_art is not None:
                ov.label_art.set_fontfamily(fam)
                ov.label_art.set_fontsize(max(7.0, s.tick_size - 1))

    @staticmethod
    def _remove_cb(cb) -> None:
        if cb is None:
            return
        try:
            cb.remove()
        except (ValueError, AttributeError, KeyError):
            pass

    def add_colorbar(self, cmap, vmin: float, vmax: float, label: str = "", *, ticks=None,
                     panel: int | None = None, tag: str = ""):
        """Colorbar for many series coloured from ``cmap`` (name or Colormap) over [vmin, vmax].

        While a colorbar exists, legend entries of a numbered series (e.g. 'Cycle 1…50')
        are dropped in 'auto' legend mode. Returns the matplotlib Colorbar.
        """
        if isinstance(cmap, str):
            cmap = matplotlib.colormaps[cmap]
        spec = dict(cmap=cmap, vmin=float(vmin), vmax=float(vmax) if vmax != vmin else float(vmin) + 1.0,
                    label=label, ticks=None if ticks is None else list(ticks), panel=self._pi(panel), tag=tag)
        spec["cb"] = self._make_cb(spec)
        self._colorbars.append(spec)
        self._apply_all(autoscale=False)
        return spec["cb"]

    @property
    def colorbars(self) -> list:
        """Colorbars added with :meth:`add_colorbar` (and the automatic legend colorbars)."""
        return [sp["cb"] for sp in self._colorbars if sp.get("cb") is not None] + list(self._auto_cbars)

    def remove_colorbars(self, tag: str | None = None) -> None:
        """Remove colorbars added with :meth:`add_colorbar` (only those with ``tag`` if given)."""
        keep = []
        for sp in self._colorbars:
            if tag is None or sp.get("tag") == tag:
                self._remove_cb(sp.get("cb"))
            else:
                keep.append(sp)
        self._colorbars = keep
        self.redraw()

    def _remove_auto_colorbars(self) -> None:
        for cb in self._auto_cbars:
            self._remove_cb(cb)
        self._auto_cbars = []

    def _update_auto_colorbars(self) -> None:
        """Legend mode 'colorbar': numbered series (> 12 entries) become a colorbar."""
        if self.settings.legend_mode != "colorbar" or not self.settings.legend_on:
            return
        from matplotlib.colors import BoundaryNorm, ListedColormap
        for i in range(len(self._axes)):
            if any(sp["panel"] == i for sp in self._colorbars):
                continue
            tr = [t for t in self.traces if t.panel_index == i and t.visible]
            groups = PM.legend_groups([t.label for t in tr], 13)
            for g in groups[:1]:
                keyed = sorted(((PM.label_key(tr[k].label), tr[k]) for k in g), key=lambda kv: kv[0][2])
                nums = np.array([kv[0][2] for kv in keyed])
                cols = [kv[1].line.get_color() for kv in keyed]
                if nums.size > 1:
                    mids = (nums[1:] + nums[:-1]) / 2
                    b = np.r_[nums[0] - (mids[0] - nums[0]), mids, nums[-1] + (nums[-1] - mids[-1])]
                else:
                    b = np.array([nums[0] - 0.5, nums[0] + 0.5])
                if not np.all(np.diff(b) > 0):
                    continue
                cmap = ListedColormap(cols)
                ticks = nums if nums.size <= 8 else nums[np.unique(np.linspace(0, nums.size - 1, 6).round()
                                                                   .astype(int))]
                prefix, unit = keyed[0][0][0], keyed[0][0][1]
                spec = dict(cmap=cmap, vmin=b[0], vmax=b[-1], norm=BoundaryNorm(b, cmap.N), ticks=list(ticks),
                            label=prefix + (f" ({unit})" if unit else ""), panel=i)
                self._auto_cbars.append(self._make_cb(spec))

    # ------------------------------------------------------------------ decorations
    def _remove_deco(self) -> None:
        for a in self._deco:
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        self._deco = []

    def _draw_panel_labels(self) -> None:
        s = self.settings
        if not s.panel_labels or len(self._axes) < 2:
            return
        from .plot_style import resolve_family
        fam = resolve_family(s.font_family)
        pos = {"top left": ((0, 1), (4, -4), "left", "top"), "top left outside": ((0, 1), (-2, 4), "right", "bottom"),
               "top right": ((1, 1), (-4, -4), "right", "top"), "bottom left": ((0, 0), (4, 4), "left", "bottom")}
        xy, off, ha, va = pos.get(s.panel_label_pos, pos["top left"])
        for i, P in enumerate(self._axes):
            t = P.ax.annotate(f"({chr(97 + i)})", xy, xycoords="axes fraction", xytext=off,
                              textcoords="offset points", ha=ha, va=va, fontsize=s.label_size + 1,
                              fontweight="bold" if s.panel_label_bold else "normal", fontfamily=fam, zorder=20,
                              annotation_clip=False)
            t.set_gid("panel_label")
            t.set_in_layout(s.panel_label_pos == "top left outside")
            self._deco.append(t)

    def _draw_offset_labels(self) -> None:
        s = self.settings
        if not s.offset_labels:
            return
        from .plot_style import resolve_family
        fam = resolve_family(s.font_family)
        for t in self.traces:
            if not t.visible or getattr(t, "_dx", None) is None or t._dx.size == 0:
                continue
            ok = np.isfinite(t._dx) & np.isfinite(t._dy)
            if not ok.any():
                continue
            k = int(np.flatnonzero(ok)[np.argmax(t._dx[ok])])
            ox, oy = t.offset
            a = t.line.axes
            txt = a.annotate(t.label, (t._dx[k] + ox, t._dy[k] + oy), xytext=(4, 0), textcoords="offset points",
                             ha="left", va="center", fontsize=s.legend_size, color=t.line.get_color(),
                             fontfamily=fam, zorder=8, annotation_clip=False)
            txt.set_gid("offset_label")
            self._deco.append(txt)

    # ------------------------------------------------------------------ overlays
    def _ov_units(self, secondary: bool, coords: str, panel: int) -> dict:
        ykey = "y2" if secondary else "y"
        out = {}
        for key in ("x", ykey):
            dim = self.axis_dimension(key, panel)
            if dim:
                unit = self.display_unit(key, panel) if coords == "display" else U.CANONICAL.get(dim)
                out["x" if key == "x" else "y"] = (dim, unit)
        return out

    def _ov_conv(self, ov: _Overlay, which: str, values):
        if values is None:
            return None
        src = ov.units.get(which)
        out = values
        if src:
            key = "x" if which == "x" else ("y2" if ov.secondary else "y")
            dim, unit = src
            target = self.display_unit(key, ov.panel)
            if dim == self.axis_dimension(key, ov.panel) and target and target != unit:
                try:
                    out = U.convert(np.asarray(values, float) if not np.isscalar(values) else values, unit,
                                    target, dim)
                except ValueError:
                    out = values
        t = ov.trace() if ov.trace is not None else None
        if t is not None and t in self.traces:
            off = t.offset[0 if which == "x" else 1]
            if off:
                out = (np.asarray(out, float) if not np.isscalar(out) else out) + off
        return out

    def _add_overlay(self, kind: str, data: dict, kw: dict, secondary: bool = False,
                     coords: str = "canonical", annotation: bool = False, panel: int | None = None,
                     trace: Trace | None = None):
        pi = self._pi(panel if trace is None else trace.panel_index)
        ov = _Overlay(kind, data, dict(kw), bool(secondary), self._ov_units(bool(secondary), coords, pi),
                      annotation, panel=pi, trace=weakref.ref(trace) if trace is not None else None)
        self._overlays.append(ov)
        self._render_overlay(ov)
        if kind == "line" and data.get("label"):
            self._apply_all(autoscale=False)
        else:
            self.redraw()
        if kind == "label":
            return ov.label_art
        return ov.artists[0] if ov.artists else None

    def _make_label(self, a, ov: _Overlay, x: float, y: float, text: str, color) -> Any:
        from .plot_style import resolve_family
        fs = max(7.0, self.settings.tick_size - 1)
        ann = a.annotate(text, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=fs,
                         fontfamily=resolve_family(self.settings.font_family),
                         color=color or "#222222", zorder=7, ha="left", va="bottom",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                         arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6, shrinkA=0, shrinkB=2))
        ann.set_in_layout(False)
        ann.arrow_patch.set_visible(False)
        ann.set_gid("label")
        ann._ls_priority = float(ov.data.get("priority", 0))
        ov.label_art = ann
        return ann

    def _render_overlay(self, ov: _Overlay) -> None:
        a = self._axes_for(ov.secondary, ov.panel)
        d, kw = ov.data, dict(ov.kw)
        fs = max(7.0, self.settings.tick_size - 1)
        arts = []
        ov.label_art = None
        if ov.kind in ("marker", "label"):
            x, y = self._ov_conv(ov, "x", d["x"]), self._ov_conv(ov, "y", d["y"])
            color = kw.pop("color", None)
            mk = d.get("marker", "o" if ov.kind == "marker" else None)
            if mk and mk != "None":
                (m,) = a.plot([x], [y], linestyle="None", marker=mk, color=color or "#222222",
                              markersize=kw.pop("markersize", 7), markeredgecolor="white", markeredgewidth=0.8,
                              zorder=6, label="_overlay", **kw)
                arts.append(m)
            if d.get("text"):
                arts.append(self._make_label(a, ov, x, y, d["text"], color if ov.kind == "label" else None))
        elif ov.kind in ("vline", "hline"):
            kw.setdefault("color", "#555555")
            kw.setdefault("linestyle", "--")
            kw.setdefault("linewidth", 1.0)
            kw.setdefault("zorder", 4)
            if ov.kind == "vline":
                v = self._ov_conv(ov, "x", d["v"])
                arts.append(a.axvline(v, label="_overlay", **kw))
                if d.get("text"):
                    arts.append(a.text(v, 0.98, " " + d["text"], transform=a.get_xaxis_transform(),
                                       rotation=90, va="top", ha="right", fontsize=fs, color="#222222",
                                       zorder=7))
            else:
                v = self._ov_conv(ov, "y", d["v"])
                arts.append(a.axhline(v, label="_overlay", **kw))
                if d.get("text"):
                    arts.append(a.text(0.01, v, d["text"], transform=a.get_yaxis_transform(), va="bottom",
                                       ha="left", fontsize=fs, color="#222222", zorder=7))
        elif ov.kind == "line":
            xs = self._ov_conv(ov, "x", np.asarray(d["xs"], float))
            ys = self._ov_conv(ov, "y", np.asarray(d["ys"], float))
            kw.setdefault("color", "#333333")
            kw.setdefault("linestyle", "--")
            kw.setdefault("linewidth", 1.2)
            kw.setdefault("zorder", 4)
            (ln,) = a.plot(xs, ys, label=d.get("label") or "_overlay", **kw)
            arts.append(ln)
        elif ov.kind == "fill":
            x = self._ov_conv(ov, "x", np.asarray(d["x"], float))
            y1 = self._ov_conv(ov, "y", np.asarray(d["y1"], float) if not np.isscalar(d["y1"]) else d["y1"])
            y2 = self._ov_conv(ov, "y", np.asarray(d["y2"], float) if not np.isscalar(d["y2"]) else d["y2"])
            kw.setdefault("alpha", 0.25)
            kw.setdefault("linewidth", 0)
            kw.setdefault("zorder", 1)
            kw.setdefault("color", "#56B4E9")
            arts.append(a.fill_between(x, y1, y2, label="_overlay", **kw))
        elif ov.kind == "polygon":
            xs = self._ov_conv(ov, "x", np.asarray(d["xs"], float))
            ys = self._ov_conv(ov, "y", np.asarray(d["ys"], float))
            kw.setdefault("alpha", 0.25)
            kw.setdefault("linewidth", 0)
            kw.setdefault("zorder", 1)
            kw.setdefault("color", "#56B4E9")
            arts.extend(a.fill(xs, ys, label="_overlay", **kw))
        elif ov.kind == "text":
            x, y = self._ov_conv(ov, "x", d["x"]), self._ov_conv(ov, "y", d["y"])
            kw.setdefault("fontsize", self.settings.label_size)
            kw.setdefault("color", "#222222")
            t = a.annotate(d["text"], (x, y), zorder=8, **kw)
            t.draggable(True)
            arts.append(t)
        ov.artists = arts

    def _remove_overlay_artists(self, ov: _Overlay) -> None:
        for art in ov.artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        ov.artists = []
        ov.label_art = None

    def _rerender_overlays(self) -> None:
        for ov in self._overlays:
            self._remove_overlay_artists(ov)
            self._render_overlay(ov)

    def add_marker(self, x: float, y: float, text: str = "", *, color=None, marker: str = "o",
                   secondary_y: bool = False, priority: float = 0, **kw):
        """Point marker (canonical units) with optional label (placed without collisions,
        see :meth:`add_label`). Returns the marker Line2D."""
        return self._add_overlay("marker", dict(x=float(x), y=float(y), text=text, marker=marker,
                                                priority=priority), dict(color=color, **kw), secondary_y)

    def add_label(self, x: float, y: float, text: str, *, color=None, marker: str | None = None,
                  secondary_y: bool = False, priority: float = 0):
        """Text label for the point (x, y) (canonical units) on the active panel.

        Labels are placed automatically so they never overlap each other or other
        labels' points (re-run on every redraw/resize); a thin leader line joins a
        label to its point when it had to move away. Higher ``priority`` labels stay
        closest to their points. ``marker`` also draws a point marker. Returns the
        text annotation artist."""
        return self._add_overlay("label", dict(x=float(x), y=float(y), text=str(text), marker=marker,
                                               priority=priority), dict(color=color), secondary_y)

    def add_vline(self, x: float, text="", **kw):
        """Vertical line at ``x`` (canonical units) with optional rotated label."""
        sec = kw.pop("secondary_y", False)
        return self._add_overlay("vline", dict(v=float(x), text=text), kw, sec)

    def add_hline(self, y: float, text="", **kw):
        """Horizontal line at ``y`` (canonical units; ``secondary_y=True`` for the right axis)."""
        sec = kw.pop("secondary_y", False)
        return self._add_overlay("hline", dict(v=float(y), text=text), kw, sec)

    def add_line(self, xs, ys, label: str | None = None, **kw):
        """Overlay polyline (e.g. tangent, modulus fit) in canonical units. A ``label``
        puts it in the legend. Overlays never affect automatic axis limits."""
        sec = kw.pop("secondary_y", False)
        return self._add_overlay("line", dict(xs=np.asarray(xs, float), ys=np.asarray(ys, float), label=label),
                                 kw, sec)

    def fill_between(self, x, y1, y2=0, **kw):
        """Shade between y1(x) and y2(x) (canonical units), e.g. integration areas."""
        sec = kw.pop("secondary_y", False)
        y1 = y1 if np.isscalar(y1) else np.asarray(y1, float)
        y2 = y2 if np.isscalar(y2) else np.asarray(y2, float)
        return self._add_overlay("fill", dict(x=np.asarray(x, float), y1=y1, y2=y2), kw, sec)

    def add_polygon(self, xs, ys, **kw):
        """Shade a closed polygon (canonical units), e.g. a hysteresis loop area."""
        sec = kw.pop("secondary_y", False)
        return self._add_overlay("polygon", dict(xs=np.asarray(xs, float), ys=np.asarray(ys, float)), kw, sec)

    def add_text(self, x: float, y: float, text: str, *, secondary_y: bool = False, coords: str = "canonical",
                 **kw):
        """Draggable text annotation (kept by :meth:`clear_overlays`; removed by :meth:`clear`)."""
        return self._add_overlay("text", dict(x=float(x), y=float(y), text=text), kw, secondary_y, coords,
                                 annotation=True)

    def clear_overlays(self, include_annotations: bool = False) -> None:
        """Remove markers/labels/lines/fills added through the API and the tools."""
        keep = []
        for ov in self._overlays:
            if ov.annotation and not include_annotations:
                keep.append(ov)
            else:
                self._remove_overlay_artists(ov)
        self._overlays = keep
        self.tools.clear_cursor()
        self._apply_all(autoscale=False)

    @property
    def overlays(self) -> list:
        """Artists of all current overlays (read-only view)."""
        return [a for ov in self._overlays for a in ov.artists]

    @property
    def labels(self) -> list:
        """Repelled label annotations (from add_label and add_marker(text=…))."""
        return [ov.label_art for ov in self._overlays if ov.label_art is not None]

    # ------------------------------------------------------------------ label placement
    def _on_draw_labels(self, evt) -> None:
        if getattr(self, "_in_relayout", False):
            return
        changed = self.relayout_labels(evt.renderer)
        if self._exporting:
            return
        if changed and self._label_redraws < 2:
            self._label_redraws += 1
            self.canvas.draw_idle()
        else:
            self._label_redraws = 0

    def relayout_labels(self, renderer=None) -> bool:
        """Place all repelled labels without overlaps (display coordinates). Returns
        True if any label moved. Runs automatically after every draw."""
        from matplotlib.text import Text
        anns = [a for a in self.labels if a.axes is not None and a.get_visible()]
        if not anns:
            return False
        self._in_relayout = True
        try:
            if renderer is None:
                renderer = self.figure.canvas.get_renderer()
            dpi = self.figure.dpi
            anchors, sizes, bounds, prio, used = [], [], [], [], []
            for a in anns:
                p = a.axes.transData.transform(a.xy)
                bb = a.axes.bbox
                if not np.all(np.isfinite(p)) or not (bb.x0 - 1 <= p[0] <= bb.x1 + 1 and bb.y0 - 1 <= p[1] <= bb.y1 + 1):
                    continue
                ext = Text.get_window_extent(a, renderer)
                anchors.append(p)
                sizes.append((ext.width, ext.height))
                bounds.append((bb.x0, bb.y0, bb.x1, bb.y1))
                prio.append(getattr(a, "_ls_priority", 0.0))
                used.append(a)
            if not used:
                return False
            pad = 0.15 * max(a.get_fontsize() for a in used) * dpi / 72.0 + 1.0
            offs = PM.place_labels(anchors, sizes, bounds, prio, pad=pad)
            changed = False
            for a, (dx, dy), (w, h) in zip(used, offs, sizes):
                pts = (dx * 72.0 / dpi, dy * 72.0 / dpi)
                old = a.xyann
                if abs(old[0] - pts[0]) > 0.5 or abs(old[1] - pts[1]) > 0.5:
                    changed = True
                a.xyann = pts
                # distance from the anchor to the label box -> leader line if far
                gx = 0.0 if dx <= 0 <= dx + w else min(abs(dx), abs(dx + w))
                gy = 0.0 if dy <= 0 <= dy + h else min(abs(dy), abs(dy + h))
                a.arrow_patch.set_visible(bool(np.hypot(gx, gy) > 9.0))
            return changed
        finally:
            self._in_relayout = False

    # ------------------------------------------------------------------ span
    def _release_span_selector(self) -> bool:
        """Disconnect the current SpanSelector; returns whether it was active."""
        sel, self._span_selector = self._span_selector, None
        if sel is None:
            return getattr(self, "_span_was_active", False)
        active = sel.active
        try:
            sel.set_visible(False)
            sel.disconnect_events()
        except (AttributeError, ValueError):  # its axes were already removed
            pass
        self._span_was_active = active
        return active

    def _make_span_selector(self) -> None:
        active = self._release_span_selector()
        self._span_selector = SpanSelector(
            self._top_axes(), self._on_span, "horizontal", useblit=True, interactive=True,
            drag_from_anywhere=True, props=dict(facecolor="#0072B2", alpha=0.12),
            handle_props=dict(color="#0072B2", linewidth=1.0), minspan=0, ignore_event_outside=False)
        self._span_selector.set_active(active)
        if self._span is not None:
            self._span_selector.extents = self._span
            self._span_selector.set_visible(True)

    def _on_span(self, xmin: float, xmax: float) -> None:
        if xmin == xmax:
            return
        self._span = (float(min(xmin, xmax)), float(max(xmin, xmax)))
        du = self.display_unit("x") or ""
        self.statusMessage.emit(f"Span: {PM.fmt(self._span[0])} – {PM.fmt(self._span[1])} {du}")
        self.tools.on_span_selected(self._span)

    def set_span(self, xmin: float, xmax: float, coords: str = "canonical") -> None:
        """Programmatically select an x-span (canonical units by default) on the active panel."""
        lo, hi = float(xmin), float(xmax)
        dim = self.axis_dimension("x")
        if coords == "canonical" and dim:
            lo, hi = (float(self._to_display(v, dim, U.CANONICAL[dim], "x")) for v in (lo, hi))
        lo, hi = sorted((lo, hi))
        self._span = (lo, hi)
        if self._span_selector is not None:
            self._span_selector.extents = (lo, hi)
            self._span_selector.set_visible(True)
        self.redraw()

    def clear_span(self) -> None:
        self._span = None
        if self._span_selector is not None:
            self._span_selector.clear()
            self._span_selector.set_visible(False)
        self.redraw()

    def current_span(self) -> tuple[float, float] | None:
        """x-range selected with the span tool, in **canonical** x units (None if none)."""
        if self._span is None:
            return None
        lo, hi = (float(self._from_display(v, "x")) for v in self._span)
        return (min(lo, hi), max(lo, hi))

    def display_span(self) -> tuple[float, float] | None:
        """Selected span in the x-axis display unit."""
        return self._span

    # ------------------------------------------------------------------ results
    def add_results(self, rows: list[dict], provenance_rec: dict | None = None) -> None:
        """Append rows (keys Tool/Trace/Quantity/Value/Unit, case-insensitive; optional
        ``warning`` / ``provenance``) and emit resultsAdded."""
        norm = self.results.add_rows(list(rows), provenance_rec)
        if norm:
            self.resultsAdded.emit(norm)

    # ------------------------------------------------------------------ palettes / colour-by
    def apply_palette(self, palette, n: int | None = None, traces: list[Trace] | None = None) -> list[str]:
        """Colour traces (default: all, in order) from a palette name, colormap name or a
        list of colours; the palette also becomes the colour cycle for new traces."""
        traces = list(traces) if traces is not None else list(self.traces)
        if isinstance(palette, str):
            colors = palette_colors(palette, n or (max(len(traces), 1) if not _is_categorical(palette) else None))
        else:
            colors = [matplotlib.colors.to_hex(c) for c in palette]
        if not colors:
            raise ValueError("Empty palette.")
        self.settings.color_cycle = list(colors)
        for k, t in enumerate(traces):
            t.line.set_color(colors[k % len(colors)])
            self._update_error_artists(t)
        self._after_traces_changed(autoscale=False)
        return colors

    def metadata_keys(self) -> list[str]:
        """Numeric metadata keys found in the plotted datasets (for 'colour by')."""
        keys: list[str] = []
        for t in self.traces:
            for k, v in (t.ds.metadata or {}).items():
                if k not in keys and _as_float(v) is not None:
                    keys.append(k)
        return keys

    def color_traces_by(self, by: str = "order", cmap: str = "viridis", *, colorbar: bool = True,
                        traces: list[Trace] | None = None, label: str | None = None) -> list[str]:
        """Colour traces from ``cmap`` by their order (``by='order'``) or by a numeric
        dataset metadata value (``by=<metadata key>``, e.g. 'anneal_T_C'), with a colorbar."""
        from matplotlib.colors import Normalize
        traces = list(traces) if traces is not None else list(self.traces)
        if not traces:
            return []
        if by == "order":
            vals = [float(k + 1) for k in range(len(traces))]
            lab = label or "Trace number"
        else:
            vals = [_as_float((t.ds.metadata or {}).get(by)) for t in traces]
            lab = label or by
        ok = [v for v in vals if v is not None]
        if not ok:
            raise ValueError(f"No numeric metadata '{by}' in the plotted datasets.")
        vmin, vmax = min(ok), max(ok)
        norm = Normalize(vmin, vmax if vmax > vmin else vmin + 1)
        cm = matplotlib.colormaps[cmap]
        lo, hi = 0.0, 1.0
        from .plot_palettes import COLORMAPS
        lo, hi = COLORMAPS.get(cmap, (0.0, 1.0))
        out, missing = [], []
        for t, v in zip(traces, vals):
            c = "#999999" if v is None else matplotlib.colors.to_hex(cm(lo + (hi - lo) * float(norm(v))))
            if v is None:
                missing.append(t.label)
            t.line.set_color(c)
            self._update_error_artists(t)
            out.append(c)
        self.remove_colorbars(tag="colour-by")
        if colorbar:
            from matplotlib.colors import LinearSegmentedColormap
            sub = LinearSegmentedColormap.from_list(f"{cmap}_sub", colormap_colors(cmap, 256, lo, hi))
            ticks = None
            if by == "order" and len(traces) <= 10:
                ticks = list(range(1, len(traces) + 1))
            self.add_colorbar(sub, vmin, vmax if vmax > vmin else vmin + 1, lab, ticks=ticks,
                              panel=traces[0].panel_index, tag="colour-by")
        self._after_traces_changed(autoscale=False)
        if missing:
            self.warn(f"No '{by}' value for: {', '.join(missing)} (shown grey).")
        return out

    # ------------------------------------------------------------------ averaging
    def average_traces(self, traces: list[Trace] | None = None, *, n_points: int | None = None,
                       error: str = "band", every: int = 10, name: str | None = None) -> Trace:
        """Mean ± SD of the selected traces on a common x grid (overlap range).

        Creates a derived dataset (x, mean, '<y> SD', n) in the store and plots the
        mean with a shaded ±SD band (``error='band'``) or error bars every ``every``
        points (``error='bars'``). Interpolation uses canonical, full-resolution data.
        """
        traces = list(traces) if traces is not None else self.selected_traces()
        if len(traces) < 2:
            raise ValueError("Select at least two traces to average (Ctrl/Shift-click in the trace list).")
        t0 = traces[0]
        if any(t.x_dim != t0.x_dim or t.y_dim != t0.y_dim for t in traces):
            raise ValueError("The selected traces have different x or y quantities.")
        res = PM.average_curves([t.xy() for t in traces], n_points)
        xc, yc = t0.ds.channel(t0.x), t0.ds.channel(t0.y)
        sd_name = f"{t0.y} SD"
        md: dict = {"averaged_from": [t.ds.name for t in traces], "n_traces": len(traces)}
        for k in ("heating_rate_K_per_min", "mass_mg", "area_mm2", "gauge_length_mm"):
            v = {(t.ds.metadata or {}).get(k) for t in traces}
            if len(v) == 1 and None not in v:
                md[k] = v.pop()
        ds = Dataset.from_arrays(name or f"Mean of {len(traces)}: {t0.label}",
                                 {t0.x: (res.x, xc.dimension, xc.unit), t0.y: (res.mean, yc.dimension, yc.unit),
                                  sd_name: (res.sd, yc.dimension if yc.dimension != "temperature" else
                                            "temperature_difference", yc.unit if yc.dimension != "temperature"
                                            else "K"),
                                  "n": (res.n, "count", "#")}, kind=t0.ds.kind, metadata=md)
        prov = provenance("Plot tools: Average traces",
                          {"traces": [t.label for t in traces], "datasets": [t.ds.name for t in traces],
                           "x": t0.x, "y": t0.y, "grid_points": int(res.x.size), "x_from": res.xmin,
                           "x_to": res.xmax, "sd": "sample (ddof=1)", "interpolation": "linear"},
                          extra={"warnings": list(res.warnings)})
        ds.metadata["provenance"] = prov
        self.store.add(ds)
        tr = self.add_trace(ds, t0.x, t0.y, label=f"Mean ± SD (n={len(traces)})", panel=t0.panel_index,
                            secondary_y=t0.secondary_y)
        self.set_trace_error(tr, error if error in ("band", "bars", "none") else "band", every)
        xu = U.CANONICAL.get(t0.x_dim, "")
        rows = [dict(tool="Average", trace=tr.label, quantity="Traces averaged", value=len(traces), unit=""),
                dict(tool="Average", trace=tr.label, quantity="From x", value=res.xmin, unit=xu),
                dict(tool="Average", trace=tr.label, quantity="To x", value=res.xmax, unit=xu),
                dict(tool="Average", trace=tr.label, quantity="Grid points", value=int(res.x.size), unit="")]
        rows += [dict(tool="Average", trace=tr.label, quantity="⚠ Warning", value=w, unit="", warning=True)
                 for w in res.warnings]
        self.add_results(rows, prov)
        if res.warnings:
            self.warn(" ".join(res.warnings))
        else:
            self.statusMessage.emit(f"Averaged {len(traces)} traces → dataset '{ds.name}'.")
        return tr

    # ------------------------------------------------------------------ templates
    def apply_template(self, template, *, recolor: bool = True) -> None:
        """Apply a style template (name or dict) to the current figure: look, colour
        cycle (existing traces re-coloured in order), line widths and export size."""
        t = TPL.get_template(template) if isinstance(template, str) else TPL.validate_template(template)
        self.settings = TPL.apply_to_settings(t, self.settings)
        if recolor:
            for k, tr in enumerate(self.traces):
                c, _ls = cycle_style(k, self.settings.color_cycle)
                tr.line.set_color(c)
                tr.line.set_linewidth(self.settings.linewidth)
                if tr.line.get_marker() not in ("None", None, ""):
                    tr.line.set_markersize(self.settings.markersize)
        self.apply_settings()
        self.statusMessage.emit(f"Style template '{t['name']}' applied.")

    def save_template(self, name: str) -> str:
        """Save the current look as user template ``name``; returns the file path."""
        path = TPL.save_template(TPL.template_from_settings(self.settings, name))
        self.settings.template = name
        return path

    @staticmethod
    def set_default_template(name: str | None) -> None:
        """Template applied to every new figure ('' / None = none)."""
        from .app_settings import settings as app_settings
        app_settings().set("default_style_template", name or "")

    @staticmethod
    def default_template() -> str:
        from .app_settings import settings as app_settings
        return str(app_settings().get("default_style_template") or "")

    def _apply_default_template(self) -> None:
        try:
            name = self.default_template()
            if name:
                t = TPL.get_template(name)
                self.settings = TPL.apply_to_settings(t, self.settings)
                self.customize_panel.sync_from_settings()
        except (KeyError, ValueError, OSError):
            pass

    # ------------------------------------------------------------------ preview at export size
    def set_preview_export_size(self, on: bool, size_cm: tuple[float, float] | None = None) -> None:
        """Fix the canvas to the export width × height (cm) so the on-screen layout equals
        the exported layout (scroll if it does not fit); ``False`` restores free resizing."""
        s = self.settings
        if size_cm is not None:
            s.export_w_cm, s.export_h_cm = float(size_cm[0]), float(size_cm[1])
        s.preview_export = bool(on)
        self._sync_preview()
        self.customize_panel.sync_from_settings()

    def preview_size_px(self) -> tuple[int, int]:
        """Logical canvas size (px) used by preview mode for the export size."""
        dpi0 = getattr(self.figure, "_original_dpi", 100.0)
        s = self.settings
        return int(round(s.export_w_cm / 2.54 * dpi0)), int(round(s.export_h_cm / 2.54 * dpi0))

    def _sync_preview(self) -> None:
        if self.settings.preview_export:
            w, h = self.preview_size_px()
            self.canvas_scroll.setWidgetResizable(False)
            self.canvas.setMinimumSize(1, 1)
            self.canvas.setFixedSize(w, h)
        else:
            self.canvas.setMinimumSize(320, 220)
            self.canvas.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
            self.canvas_scroll.setWidgetResizable(True)
        self.redraw()

    # ------------------------------------------------------------------ export
    def export_figure(self, path: str, fmt: str | None = None, dpi: int = 300, transparent: bool = False,
                      size_in: tuple[float, float] | None = None) -> None:
        """Save the figure (PNG/SVG/PDF from ``fmt`` or the extension) at full data resolution."""
        self.tools.hide_transient()
        old = self.figure.get_size_inches().copy()
        old_dpi = self.figure.dpi
        self._full_res = True
        self._exporting = True
        try:
            for t in self.traces:
                if len(t) > self.display_max_points:
                    self._update_trace_data(t, recompute=False)
            if size_in is not None:
                w, h = float(size_in[0]), float(size_in[1])
                ext = (fmt or str(path).rsplit(".", 1)[-1]).lower() if isinstance(path, str) or fmt else "png"
                if ext in ("png", "jpg", "jpeg", "tif", "tiff"):
                    # matplotlib truncates; snap so that pixels = round(inches × dpi) exactly
                    w = (round(w * dpi) + 1e-3) / dpi
                    h = (round(h * dpi) + 1e-3) / dpi
                self.figure.set_size_inches(w, h, forward=False)
            if self.labels:
                for _ in range(2):  # settle constrained layout, then place labels for this size
                    self.figure.draw_without_rendering()
            kw = {}
            if transparent:
                kw.update(transparent=True)
            else:
                kw.update(facecolor=self.figure.get_facecolor())
            self.figure.savefig(path, format=fmt, dpi=dpi, **kw)
        finally:
            self._full_res = False
            self._exporting = False
            self.figure.set_size_inches(old, forward=False)
            self.figure.set_dpi(old_dpi)
            for t in self.traces:
                if len(t) > self.display_max_points:
                    self._update_trace_data(t, use_view=True, recompute=False)
            self.redraw()

    def export_png(self, path: str, dpi: int = 300, transparent: bool = False, size_in=None) -> None:
        """Save a PNG. ``size_in`` = (width, height) in inches; default = current figure size."""
        self.export_figure(path, "png", dpi, transparent, size_in)

    def figure_to_png_bytes(self, dpi: int = 150) -> bytes:
        import io
        buf = io.BytesIO()
        self.export_figure(buf, "png", dpi)
        return buf.getvalue()

    def copy_figure_to_clipboard(self, dpi: int = 150) -> None:
        img = QImage.fromData(self.figure_to_png_bytes(dpi), "PNG")
        QGuiApplication.clipboard().setImage(img)
        self.statusMessage.emit("Figure copied to clipboard.")

    def show_export_dialog(self) -> None:
        """Open the export dialog (PNG/SVG/PDF, DPI, size in cm, transparency)."""
        from .plot_customize import ExportDialog
        dlg = ExportDialog(self, self)
        dlg.exec()

    open_export_dialog = show_export_dialog

    # ------------------------------------------------------------------ session state
    def get_state(self) -> dict:
        """JSON-serialisable state (datasets referenced by ``ds.id``); overlays are not kept."""
        sel = self.selected_trace()
        return {
            "version": 2,
            "traces": [{"ds": t.ds.id, "x": t.x, "y": t.y, "label": t.label, "style": t.style,
                        "secondary": t.secondary_y, "panel": t.panel_index, "visible": t.visible,
                        "error": t.error_mode, "error_every": t.error_every} for t in self.traces],
            "selected": self.traces.index(sel) if sel is not None else None,
            "settings": self.settings.to_dict(),
            "active_panel": self._active,
            "span": list(self.current_span()) if self.current_span() is not None else None,
            "template": self.settings.template,
            "ui": {"h_split": list(self.h_split.sizes()), "v_split": list(self.v_split.sizes()),
                   "side_panels": self.side_panels_visible()},
        }

    def set_state(self, state: dict, id_map: dict) -> None:
        """Restore :meth:`get_state` output; ``id_map`` maps saved dataset ids to Datasets."""
        state = state or {}
        self.clear()
        sd = state.get("settings") or {}
        self.settings = settings_from_dict(sd)
        self._active = 0
        self._rebuild_axes()
        missing = []
        for td in state.get("traces", []):
            ds = id_map.get(td.get("ds"))
            if ds is None:
                ds = id_map.get(str(td.get("ds"))) or (id_map.get(int(td["ds"])) if str(td.get("ds", "")).isdigit()
                                                       else None)
            if ds is None or td.get("x") not in ds.data.columns or td.get("y") not in ds.data.columns:
                missing.append(str(td.get("label")))
                continue
            st = {k: v for k, v in (td.get("style") or {}).items()
                  if k in ("color", "linestyle", "linewidth", "marker", "markersize", "alpha")}
            t = self.add_trace(ds, td["x"], td["y"], td.get("label"), secondary_y=bool(td.get("secondary")),
                               panel=int(td.get("panel", 0)), **st)
            t.error_mode = td.get("error", "none") or "none"
            t.error_every = int(td.get("error_every", 10) or 10)
            if not td.get("visible", True):
                t.line.set_visible(False)
        # adding traces may have dropped unit overrides (axis had no dimension yet) → restore
        self.settings = settings_from_dict(sd)
        self._active = self._pi(state.get("active_panel", 0))
        self._make_span_selector()
        sel = state.get("selected")
        if isinstance(sel, int) and 0 <= sel < len(self.traces):
            self._selected = self.traces[sel]
        self.apply_settings()
        if state.get("span"):
            self.set_span(*state["span"])
        ui = state.get("ui") or {}
        if ui.get("h_split"):
            self.h_split.setSizes([int(v) for v in ui["h_split"]])
        if ui.get("v_split"):
            self.v_split.setSizes([int(v) for v in ui["v_split"]])
        if "side_panels" in ui:
            self.set_side_panels_visible(bool(ui["side_panels"]))
        self.trace_manager.refresh_traces()
        self.tools.on_selection_changed()
        if missing:
            self.warn(f"Session: {len(missing)} trace(s) could not be restored (dataset/column missing): "
                      + ", ".join(missing))

    # ------------------------------------------------------------------ store
    def _on_store_changed(self) -> None:
        try:
            ids = {d.id for d in self.store}
            gone = [t for t in self.traces if t.ds.id not in ids or t.x not in t.ds.data.columns
                    or t.y not in t.ds.data.columns]
            for t in gone:
                self._remove_error_artists(t)
                t.line.remove()
                self.traces.remove(t)
            if gone:
                self._drop_ax2_if_unused()
            self.trace_manager.refresh_datasets()
            self._refresh_all_data()
            self._after_traces_changed(autoscale=self._any_auto())
        except RuntimeError:  # widget already deleted (C++ side)
            pass

    def closeEvent(self, ev):  # noqa: N802
        try:
            self.store._listeners.remove(self._store_cb)
        except (ValueError, AttributeError):
            pass
        super().closeEvent(ev)


def _is_categorical(name: str) -> bool:
    from .plot_palettes import PALETTES
    return name in PALETTES


def _as_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# backward-compatible re-export
from .plot_traces import TraceManager  # noqa: E402,F401
