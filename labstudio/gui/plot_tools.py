"""Interactive plot tools for the Plot tab (module B).

``ToolController`` owns the tool toolbar (mutually exclusive modes), the
per-tool option strip and the canvas event handling. Each tool's computation is
also callable programmatically (``find_peaks``, ``integrate``, ``fit``,
``stats``, ``pin_nearest``) and works in the axes' *display* units on
full-resolution data (display decimation and waterfall offsets never affect results).

v1.1: the Integrate tool is segment-aware (curves that double back in x, e.g. DSC
heating+cooling or load/unload loops): ``segment='auto'`` integrates only the first
monotonic segment in the span for heat-flow traces (with a warning row when the span
crosses segments) and the enclosed loop area for other curves (labelled as such);
``'whole'`` = whole path, ``n`` = segment n. Every tool run records its parameters
in a provenance dict kept with its result rows; warnings become amber rows.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform
from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QInputDialog, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QStackedWidget, QToolBar, QWidget,
)

from labstudio.core.provenance import provenance

from . import plot_math as PM

_TEMPS = ("temperature", "temperature_difference")
_HEAT = ("heat_flow", "specific_heat_flow")

if TYPE_CHECKING:  # pragma: no cover
    from .plot_widget import PlotPanel, Trace

MODES: list[tuple[str, str, str]] = [
    ("navigate", "Navigate", "Use the matplotlib toolbar (pan / zoom); no LabStudio tool active."),
    ("cursor", "Data cursor", "Hover to read the nearest data point (snaps across all visible traces); "
                              "click to pin it and add it to the results."),
    ("peaks", "Peaks", "Find peaks or valleys on the selected trace (inside the span if one is selected)."),
    ("span", "Span", "Drag horizontally to select an x-range used by the other tools."),
    ("integrate", "Integrate", "Integrate the selected trace over the whole trace or the span, with optional "
                               "linear baseline; drag a span to integrate it directly."),
    ("fit", "Linear fit", "Least-squares line over the span: slope (with units; modulus for stress–strain), "
                          "intercept and R²."),
    ("stats", "Min/Max/Mean", "Statistics of the selected trace inside the span."),
    ("text", "Text", "Click on the plot to place a text annotation (drag it afterwards to move it)."),
]
SPAN_MODES = {"span", "integrate", "fit", "stats", "peaks"}


def _hint(text: str) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(6, 0, 6, 0)
    lab = QLabel(text)
    lab.setStyleSheet("color:#555;")
    lay.addWidget(lab)
    lay.addStretch(1)
    return w


class ToolController(QObject):
    """Tool toolbar + option strip + canvas interaction for a :class:`PlotPanel`."""

    CURSOR_SNAP_PX = 40.0

    def __init__(self, panel: "PlotPanel"):
        super().__init__(panel)
        self.panel = panel
        self.mode = "navigate"
        self._bg = None
        self._cache: list[tuple["Trace", np.ndarray, np.ndarray]] | None = None
        self._hover: tuple["Trace", int] | None = None
        self._build_toolbar()
        self._build_options()
        fig = panel.figure
        self._hl = Line2D([], [], marker="o", markersize=10, markerfacecolor="none", markeredgecolor="#222222",
                          markeredgewidth=1.6, linestyle="None", transform=IdentityTransform(), animated=True)
        fig.add_artist(self._hl)
        self._tip = fig.text(0, 0, "", transform=IdentityTransform(), animated=True, fontsize=9, color="#222222",
                             va="bottom", ha="left", zorder=100,
                             bbox=dict(boxstyle="round,pad=0.35", fc="#ffffffee", ec="#888888", lw=0.6))
        self._hl.set_visible(False)
        self._tip.set_visible(False)
        c = panel.canvas
        c.mpl_connect("draw_event", self._on_draw)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("key_press_event", self._on_key)
        c.mpl_connect("figure_leave_event", lambda e: self._hide_hover(blit=True))
        panel.tracesChanged.connect(self._refresh_target_combo)

    # ------------------------------------------------------------------ UI
    def _build_toolbar(self) -> None:
        tb = QToolBar("Tools")
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        tb.setMovable(False)
        tb.setStyleSheet("QToolButton{padding:3px 7px;} QToolButton:checked{background:#d6e6f5;"
                         "border:1px solid #7aa7d1; border-radius:3px;}")
        self.toolbar = tb
        self.group = QActionGroup(tb)
        self.group.setExclusive(True)
        self.actions: dict[str, QAction] = {}
        for key, text, tip in MODES:
            a = QAction(text, tb, checkable=True)
            a.setToolTip(tip)
            a.setStatusTip(tip)
            a.triggered.connect(lambda _=False, k=key: self.set_mode(k))
            self.group.addAction(a)
            tb.addAction(a)
            self.actions[key] = a
        self.actions["navigate"].setChecked(True)
        # second bar (placed beside the matplotlib toolbar): one-shot actions + tool target
        ab = QToolBar("Plot actions")
        ab.setToolButtonStyle(Qt.ToolButtonTextOnly)
        ab.setMovable(False)
        ab.setStyleSheet("QToolButton{padding:3px 6px;}")
        self.actions_bar = ab
        for text, tip, fn in (
                ("Clear overlays", "Remove markers, lines and shaded areas added by tools/analysis.",
                 lambda: self.panel.clear_overlays()),
                ("Clear span", "Deselect the x-span.", lambda: self.panel.clear_span()),
                ("Export…", "Export the figure as PNG / SVG / PDF (DPI, size in cm, transparency).",
                 lambda: self.panel.show_export_dialog()),
                ("Copy figure", "Copy the figure to the clipboard as an image.",
                 lambda: self.panel.copy_figure_to_clipboard())):
            a = QAction(text, ab)
            a.setToolTip(tip)
            a.triggered.connect(fn)
            ab.addAction(a)
        self.act_side = QAction("Hide side panels", ab, checkable=True)
        self.act_side.setToolTip("Hide the trace manager and the customisation panel to give the plot the whole "
                                 "width (useful on small laptop screens).")
        self.act_side.toggled.connect(lambda on: self.panel.set_side_panels_visible(not on))
        ab.addAction(self.act_side)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.addWidget(QLabel("Target: "))
        self.cb_target = QComboBox()
        self.cb_target.setToolTip("Trace used by Peaks / Integrate / Linear fit / Min-Max-Mean "
                                  "(same as the selection in the trace list).")
        self.cb_target.setMinimumContentsLength(14)
        self.cb_target.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_target.currentIndexChanged.connect(self._target_changed)
        tb.addWidget(self.cb_target)
        nav = self.panel.nav_toolbar
        for name in ("pan", "zoom"):
            act = getattr(nav, "_actions", {}).get(name)
            if act is not None:
                act.triggered.connect(self._nav_used)

    def _build_options(self) -> None:
        st = QStackedWidget()
        st.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.options = st
        self.pages: dict[str, QWidget] = {}
        self.pages["navigate"] = _hint("Pan/zoom with the toolbar above · choose a tool for read-outs, peaks, "
                                       "integration and fits.")
        self.pages["cursor"] = _hint("Hover near a curve to read x, y · click to pin the point to the results · "
                                     "Esc hides the read-out.")
        # peaks
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(6, 0, 6, 0)
        self.sp_prom = QDoubleSpinBox(minimum=0.0, maximum=1e12, decimals=4, singleStep=0.1)
        self.sp_prom.setSpecialValueText("auto")
        self.sp_prom.setToolTip("Minimum peak prominence in y display units (how far a peak stands out from "
                                "its surroundings). 'auto' = 5 % of the y range.")
        self.sp_dist = QDoubleSpinBox(minimum=0.0, maximum=1e12, decimals=4, singleStep=0.5)
        self.sp_dist.setSpecialValueText("none")
        self.sp_dist.setToolTip("Minimum separation between peaks in x display units.")
        self.cb_sign = QComboBox()
        self.cb_sign.addItems(["peaks", "valleys"])
        self.cb_sign.setToolTip("Look for maxima (peaks) or minima (valleys) — e.g. exothermic vs endothermic "
                                "DSC events depending on the sign convention.")
        self.chk_peak_span = QCheckBox("Within span")
        self.chk_peak_span.setChecked(True)
        self.chk_peak_span.setToolTip("Only search inside the selected x-span (if any).")
        b = QPushButton("Find peaks")
        b.clicked.connect(self._peaks_clicked)
        for lab, wd in (("Prominence", self.sp_prom), ("Min distance", self.sp_dist), ("Type", self.cb_sign)):
            h.addWidget(QLabel(lab))
            h.addWidget(wd)
        h.addWidget(self.chk_peak_span)
        h.addWidget(b)
        h.addStretch(1)
        self.pages["peaks"] = w
        self.pages["span"] = _hint("Drag horizontally on the plot to select an x-range (drag the edges to "
                                   "adjust). Used by Integrate, Linear fit, Min/Max/Mean and Peaks.")
        # integrate
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(6, 0, 6, 0)
        self.cb_range = QComboBox()
        self.cb_range.addItems(["Selected span", "Whole trace"])
        self.cb_range.setToolTip("Integrate over the selected span or the whole trace.")
        self.cb_seg = QComboBox()
        self.cb_seg.setToolTip("Which part of a curve that doubles back in x is integrated:\n"
                               "• auto (warn): heat flow → only the first monotonic segment in the span (e.g. "
                               "the heating run), with a warning if the span crosses heating and cooling; other "
                               "curves (stress–strain) → enclosed loop area, labelled as such.\n"
                               "• whole path (loop area): integrate along the whole recorded path.\n"
                               "• segment n: one monotonic segment only (heating 1, cooling 1, …).")
        self.cb_seg.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._fill_segment_combo([])
        self.cb_base = QComboBox()
        self.cb_base.addItems(["none", "linear"])
        self.cb_base.setToolTip("Baseline subtracted before integrating: none, or a straight line joining the "
                                "curve at the two span ends (typical for DSC peaks).")
        self.sp_rate = QDoubleSpinBox(minimum=0.0, maximum=1000.0, decimals=2, singleStep=1.0)
        self.sp_rate.setSpecialValueText("—")
        self.sp_rate.setSuffix(" K/min")
        self.sp_rate.setToolTip("Heating/cooling rate β. For heat flow vs temperature, ∫hf dT / β gives the "
                                "energy (J, or J/g for W/g). Pre-filled from the dataset metadata.")
        self.sp_mass = QDoubleSpinBox(minimum=0.0, maximum=1e6, decimals=3, singleStep=1.0)
        self.sp_mass.setSpecialValueText("—")
        self.sp_mass.setSuffix(" mg")
        self.sp_mass.setToolTip("Sample mass, used to convert J to J/g for heat flow in mW.")
        b = QPushButton("Integrate")
        b.clicked.connect(self._integrate_clicked)
        for lab, wd in (("Range", self.cb_range), ("Segment", self.cb_seg), ("Baseline", self.cb_base),
                        ("Heating rate", self.sp_rate), ("Mass", self.sp_mass)):
            h.addWidget(QLabel(lab))
            h.addWidget(wd)
        h.addWidget(b)
        h.addStretch(1)
        self.pages["integrate"] = w
        # fit / stats
        for key, text, fn in (("fit", "Fit span", self._fit_clicked), ("stats", "Compute", self._stats_clicked)):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(6, 0, 6, 0)
            h.addWidget(QLabel("Drag a span on the plot (runs automatically), or"))
            b = QPushButton(text)
            b.clicked.connect(fn)
            h.addWidget(b)
            h.addStretch(1)
            self.pages[key] = w
        self.pages["text"] = _hint("Click on the plot to add a text label; labels can be dragged. "
                                   "Removed with ‘Clear all’ in the trace list.")
        for key, _t, _tip in MODES:
            st.addWidget(self.pages[key])
        st.setFixedHeight(32)
        sc = QScrollArea()
        sc.setWidget(st)
        sc.setWidgetResizable(True)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setFixedHeight(46)
        sc.setMinimumWidth(120)
        self.options_scroll = sc

    def _refresh_target_combo(self) -> None:
        self.cb_target.blockSignals(True)
        self.cb_target.clear()
        for t in self.panel.traces:
            self.cb_target.addItem(t.label)
        sel = self.panel.selected_trace()
        if sel is not None:
            self.cb_target.setCurrentIndex(self.panel.traces.index(sel))
        self.cb_target.blockSignals(False)
        self.on_selection_changed()

    def _target_changed(self, i: int) -> None:
        if 0 <= i < len(self.panel.traces):
            self.panel._selected = self.panel.traces[i]
            self.panel.trace_manager.refresh_traces()
            self.on_selection_changed()

    def _fill_segment_combo(self, names: list[str]) -> None:
        cur = self.cb_seg.currentData()
        self.cb_seg.blockSignals(True)
        self.cb_seg.clear()
        self.cb_seg.addItem("auto (warn)", "auto")
        self.cb_seg.addItem("whole path (loop area)", "whole")
        for k, n in enumerate(names, 1):
            self.cb_seg.addItem(f"segment {k} ({n})", k)
        i = self.cb_seg.findData(cur)
        self.cb_seg.setCurrentIndex(i if i >= 0 else 0)
        self.cb_seg.blockSignals(False)

    def on_selection_changed(self) -> None:
        t = self.panel.selected_trace()
        if t is None:
            self._fill_segment_combo([])
            return
        try:
            segs, names = self.segments(t)
        except (ValueError, KeyError):
            names = []
        self._fill_segment_combo(names if len(names) > 1 else [])
        md = t.ds.metadata or {}
        self.sp_rate.setValue(float(md.get("heating_rate_K_per_min") or 0.0))
        self.sp_mass.setValue(float(md.get("mass_mg") or 0.0))
        if self.cb_target.currentIndex() != self.panel.traces.index(t):
            self.cb_target.blockSignals(True)
            self.cb_target.setCurrentIndex(self.panel.traces.index(t))
            self.cb_target.blockSignals(False)

    # ------------------------------------------------------------------ modes
    def set_mode(self, mode: str) -> None:
        """Activate a tool: navigate, cursor, peaks, span, integrate, fit, stats, text."""
        if mode not in self.actions:
            raise ValueError(f"Unknown tool '{mode}'")
        self.mode = mode
        self.actions[mode].setChecked(True)
        nav = self.panel.nav_toolbar
        if mode != "navigate":
            m = getattr(getattr(nav, "mode", None), "name", "NONE")
            if m == "PAN":
                nav.pan()
            elif m == "ZOOM":
                nav.zoom()
        sel = self.panel._span_selector
        if sel is not None:
            sel.set_active(mode in SPAN_MODES)
        self.options.setCurrentWidget(self.pages[mode])
        if mode != "cursor":
            self._hide_hover(blit=True)
        self.panel.statusMessage.emit(dict((k, tip) for k, _t, tip in MODES)[mode])

    def _nav_used(self, *_):
        if self.mode != "navigate":
            self.mode = "navigate"
            self.actions["navigate"].setChecked(True)
            self.options.setCurrentWidget(self.pages["navigate"])
            if self.panel._span_selector is not None:
                self.panel._span_selector.set_active(False)
            self._hide_hover(blit=True)

    # ------------------------------------------------------------------ helpers
    def _target(self, trace=None) -> "Trace | None":
        t = trace or self.panel.selected_trace()
        if t is None:
            self.panel.statusMessage.emit("No trace selected — add a trace first.")
        return t

    def _units(self, t: "Trace") -> tuple[str, str]:
        """Units of ``t.xy_display()`` (axis display unit, or the trace's own unit if its
        dimension differs from the axis's primary dimension)."""
        p = self.panel
        ykey = "y2" if t.secondary_y else "y"
        i = t.panel_index
        xu = p.display_unit("x", i) if t.x_dim == p.axis_dimension("x", i) else t.x_unit
        yu = p.display_unit(ykey, i) if t.y_dim == p.axis_dimension(ykey, i) else t.y_unit
        return xu or "", yu or ""

    def _span_mask(self, x: np.ndarray, use_span: bool) -> tuple[np.ndarray, tuple[float, float] | None]:
        sp = self.panel.display_span() if use_span else None
        if sp is None:
            return np.ones(x.size, bool), None
        return (x >= sp[0]) & (x <= sp[1]), sp

    def _ov(self, t: "Trace", kind: str, data: dict, kw: dict):
        return self.panel._add_overlay(kind, data, kw, t.secondary_y, coords="display", trace=t)

    def _prov(self, tool: str, t: "Trace", params: dict, warnings=()) -> dict:
        xu, yu = self._units(t)
        p = {"trace": t.label, "x": t.x, "y": t.y, "display_units": {"x": xu, "y": yu}}
        p.update(params)
        return provenance(f"Plot tools: {tool}", p, ds=t.ds, extra={"warnings": list(warnings)})

    def _rows(self, tool: str, t: "Trace", items, prov: dict | None = None, warnings=()) -> list[dict]:
        rows = [dict(tool=tool, trace=t.label, quantity=q, value=v, unit=u) for q, v, u in items]
        rows += [dict(tool=tool, trace=t.label, quantity="⚠ Warning", value=w, unit="", warning=True)
                 for w in warnings]
        if prov is None:
            prov = self._prov(tool, t, {}, warnings)
        self.panel.add_results(rows, prov)
        return rows

    def _span_param(self, lo, hi) -> list | None:
        return None if lo is None and hi is None else [lo, hi]

    def hide_transient(self) -> None:
        self._hide_hover(blit=False)

    def clear_cursor(self) -> None:
        self._hide_hover(blit=False)

    def invalidate_cache(self) -> None:
        self._cache = None

    # ------------------------------------------------------------------ cursor
    def _on_draw(self, _evt) -> None:
        self._cache = None
        self._bg = self.panel.canvas.copy_from_bbox(self.panel.figure.bbox)
        if self._hl.get_visible():
            # keep the read-out across full redraws (re-positioned for the new view)
            if self._hover is None or self._hover[0] not in self.panel.traces or not self._hover[0].visible:
                self._hl.set_visible(False)
                self._tip.set_visible(False)
                return
            self._place_hover(*self._hover)
            fig = self.panel.figure
            fig.draw_artist(self._hl)
            fig.draw_artist(self._tip)

    def _build_cache(self) -> list:
        cache = []
        for t in self.panel.traces:
            if not t.visible or getattr(t, "_dx", None) is None:
                continue
            a = t.line.axes
            ox, oy = t.offset
            dx, dy = t._dx + ox, t._dy + oy
            with np.errstate(all="ignore"):
                pts = a.transData.transform(np.column_stack([dx, dy]))
            bb = a.bbox
            ok = (np.isfinite(pts).all(axis=1) & (pts[:, 0] >= bb.x0 - 2) & (pts[:, 0] <= bb.x1 + 2)
                  & (pts[:, 1] >= bb.y0 - 2) & (pts[:, 1] <= bb.y1 + 2))
            idx = np.flatnonzero(ok)
            cache.append((t, pts[idx], idx))
        self._cache = cache
        return cache

    def nearest(self, px: float, py: float, max_dist: float | None = None):
        """Nearest visible data point to pixel (px, py): ``(trace, index, dist_px)`` or None."""
        cache = self._cache if self._cache is not None else self._build_cache()
        hit = PM.nearest_point([c[1] for c in cache], (px, py),
                               self.CURSOR_SNAP_PX if max_dist is None else max_dist)
        if hit is None:
            return None
        k, i, d = hit
        t, _pts, idx = cache[k]
        return t, int(idx[i]), d

    def _readout(self, t: "Trace", i: int) -> tuple[float, float, str]:
        xu, yu = self._units(t)
        x, y = float(t._dx[i]), float(t._dy[i])
        return x, y, f"{t.label}\nx = {PM.fmt(x)} {xu}\ny = {PM.fmt(y)} {yu}"

    def _on_motion(self, evt) -> None:
        if self.mode != "cursor" or evt.x is None:
            return
        hit = self.nearest(evt.x, evt.y)
        if hit is None:
            self._hover = None
            self._hide_hover(blit=True)
            return
        t, i, _d = hit
        self._hover = (t, i)
        txt = self._place_hover(t, i)
        self._blit()
        self.panel.statusMessage.emit(txt.replace("\n", "   "))

    def _place_hover(self, t: "Trace", i: int) -> str:
        x, y, txt = self._readout(t, i)
        ox, oy = t.offset
        px, py = t.line.axes.transData.transform((x + ox, y + oy))
        self._hl.set_data([px], [py])
        self._hl.set_markeredgecolor(t.line.get_color())
        self._tip.set_text(txt)
        W, H = self.panel.figure.bbox.width, self.panel.figure.bbox.height
        right = px > W * 0.7
        top = py > H * 0.75
        self._tip.set_position((px - 14 if right else px + 14, py - 14 if top else py + 14))
        self._tip.set_ha("right" if right else "left")
        self._tip.set_va("top" if top else "bottom")
        self._hl.set_visible(True)
        self._tip.set_visible(True)
        return txt

    def _blit(self) -> None:
        c = self.panel.canvas
        if self._bg is None:
            return
        c.restore_region(self._bg)
        fig = self.panel.figure
        if self._hl.get_visible():
            fig.draw_artist(self._hl)
            fig.draw_artist(self._tip)
        c.blit(fig.bbox)

    def _hide_hover(self, blit: bool = False) -> None:
        if not hasattr(self, "_hl"):
            return
        was = self._hl.get_visible()
        self._hover = None
        self._hl.set_visible(False)
        self._tip.set_visible(False)
        if blit and was:
            self._blit()

    def pin_nearest(self, px: float, py: float) -> list[dict]:
        """Pin the data point nearest to pixel (px, py): marker + results rows."""
        hit = self.nearest(px, py)
        if hit is None:
            return []
        t, i, _ = hit
        x, y, _txt = self._readout(t, i)
        xu, yu = self._units(t)
        self._ov(t, "marker", dict(x=x, y=y, text=f"({PM.fmt(x, 4)}, {PM.fmt(y, 4)})", marker="o"),
                 dict(color=t.line.get_color()))
        xl = t.ds.channel(t.x).name
        yl = t.ds.channel(t.y).name
        return self._rows("Cursor", t, [(xl, x, xu), (yl, y, yu)],
                          self._prov("Cursor", t, {"point_index": int(i), "x": x, "y": y}))

    def _on_press(self, evt) -> None:
        if evt.button != 1 or evt.x is None:
            return
        if self.mode == "cursor":
            self.pin_nearest(evt.x, evt.y)
        elif self.mode == "text" and evt.inaxes is not None:
            for i, P in enumerate(self.panel._axes):
                if evt.inaxes in (P.ax, P.ax2):
                    self.panel.set_active_panel(i)
            text, ok = QInputDialog.getText(self.panel, "Add text", "Annotation text:")
            if ok and text:
                x, y = self.panel.ax.transData.inverted().transform((evt.x, evt.y))
                self.panel.add_text(x, y, text, coords="display")

    def _on_key(self, evt) -> None:
        if evt.key == "escape":
            self._hide_hover(blit=True)
            if self.mode in SPAN_MODES:
                self.panel.clear_span()

    # ------------------------------------------------------------------ span hook
    def on_span_selected(self, span: tuple[float, float]) -> None:
        if self.mode == "integrate":
            self.cb_range.setCurrentText("Selected span")
            self._integrate_clicked()
        elif self.mode == "fit":
            self._fit_clicked()
        elif self.mode == "stats":
            self._stats_clicked()

    # ------------------------------------------------------------------ peaks
    def _peaks_clicked(self) -> None:
        self.find_peaks(prominence=self.sp_prom.value() or None, min_distance=self.sp_dist.value() or None,
                        sign=self.cb_sign.currentText(), use_span=self.chk_peak_span.isChecked())

    def find_peaks(self, trace=None, *, prominence: float | None = None, min_distance: float | None = None,
                   sign: str = "peaks", use_span: bool = True, mark: bool = True) -> list[PM.Peak]:
        """Peaks (or valleys) of the target trace in display units; marks them and lists
        x, y, prominence and FWHM in the results."""
        t = self._target(trace)
        if t is None:
            return []
        x, y = t.xy_display()
        m, _sp = self._span_mask(x, use_span)
        peaks = PM.find_peaks_xy(x[m], y[m], prominence=prominence, min_distance=min_distance, sign=sign)
        xu, yu = self._units(t)
        items = []
        word = "Valley" if sign.startswith("valley") else "Peak"
        for k, p in enumerate(peaks, 1):
            items += [(f"{word} {k} x", p.x, xu), (f"{word} {k} y", p.y, yu),
                      (f"{word} {k} prominence", p.prominence, yu), (f"{word} {k} FWHM", p.fwhm, xu)]
            if mark:
                self._ov(t, "marker", dict(x=p.x, y=p.y, text=f"{PM.fmt(p.x, 4)}", marker="v" if word == "Peak"
                                           else "^"), dict(color=t.line.get_color()))
                if np.isfinite(p.fwhm) and p.fwhm > 0:
                    self._ov(t, "line", dict(xs=[p.left_x, p.right_x], ys=[p.half_y, p.half_y], label=None),
                             dict(color=t.line.get_color(), linestyle=":", linewidth=1.2))
        if items:
            self._rows("Peaks", t, items, self._prov("Peaks", t, {
                "prominence": prominence if prominence else "auto (5 % of y range)",
                "min_distance": min_distance, "sign": sign, "span": list(_sp) if _sp else None}))
        self.panel.statusMessage.emit(f"{len(peaks)} {word.lower()}(s) found on '{t.label}'.")
        return peaks

    # ------------------------------------------------------------------ integrate
    def _integrate_clicked(self) -> None:
        whole = self.cb_range.currentText() == "Whole trace"
        try:
            self.integrate(baseline=self.cb_base.currentText(), use_span=not whole,
                           heating_rate_K_per_min=self.sp_rate.value() or None,
                           mass_mg=self.sp_mass.value() or None, segment=self.cb_seg.currentData() or "auto")
        except ValueError as e:
            self.panel.warn(f"Integrate: {e}")

    def segments(self, trace=None) -> tuple[list[tuple[int, int, int]], list[str]]:
        """Monotonic segments of the target trace's x (cached) and their names
        ('heating 1', 'cooling 1', 'loading 1' …)."""
        t = self._target(trace)
        if t is None:
            return [], []
        x, _y = t.xy_display()
        key = (x.size, self._units(t)[0], float(x[0]) if x.size else 0.0, float(x[-1]) if x.size else 0.0,
               float(np.nansum(x[:: max(1, x.size // 64)])) if x.size else 0.0)
        cache = getattr(t, "_segcache", None)
        if cache is None or cache[0] != key:
            segs = PM.monotonic_segments(x)
            cache = (key, segs, PM.segment_names(segs, t.x_dim))
            t._segcache = cache
        return cache[1], cache[2]

    def integrate(self, trace=None, *, xmin: float | None = None, xmax: float | None = None,
                  baseline: str = "none", use_span: bool = True, heating_rate_K_per_min: float | None = None,
                  mass_mg: float | None = None, shade: bool = True, segment="auto"):
        """Integrate the target trace (display units). Range: ``xmin/xmax`` if given, else the
        span (if ``use_span`` and one is selected), else the whole trace.

        ``segment``: ``'auto'`` (default), ``'whole'`` (whole path / loop area) or a 1-based
        segment number (see :meth:`segments`). For heat flow, 'auto' integrates only the
        first monotonic segment that lies in the span and adds a warning row if the span
        crosses several (heating + cooling); for other curves 'auto' gives the enclosed
        loop area along the whole path, labelled "(enclosed loop)".

        Returns ``(IntegrationResult, list[Quantity])``; rows (+ warning rows and a
        provenance record) are added to the results. Heating rate / mass default to
        the dataset metadata.
        """
        t = self._target(trace)
        if t is None:
            return None, []
        x, y = t.xy_display()
        range_kind = "explicit" if (xmin is not None or xmax is not None) else "whole trace"
        if xmin is None and xmax is None and use_span and self.panel.display_span() is not None:
            xmin, xmax = self.panel.display_span()
            range_kind = "span"
        fin = np.isfinite(x)
        if not fin.any():
            raise ValueError("The trace has no finite x values.")
        lo = float(np.min(x[fin])) if xmin is None else float(xmin)
        hi = float(np.max(x[fin])) if xmax is None else float(xmax)
        lo, hi = min(lo, hi), max(lo, hi)
        segs, names = self.segments(t)
        in_span = PM.segments_in_span(x, segs, lo, hi)
        dsc_like = t.y_dim in _HEAT and t.x_dim in _TEMPS + ("time",)
        where = "span" if range_kind == "span" else ("range" if range_kind == "explicit" else "trace")
        crossed = ", ".join(names[k] for k in in_span)
        warnings: list[str] = []
        seg_idx: int | None = None
        mode = segment if segment in ("auto", "whole") else int(segment)
        if mode == "auto":
            if len(in_span) > 1 and dsc_like:
                seg_idx = in_span[0]
                warnings.append(f"The {where} covers {len(in_span)} monotonic segments ({crossed}) — integrated "
                                f"'{names[seg_idx]}' only. Pick 'whole path' or another segment in the "
                                f"Integrate options if that is not what you want.")
            elif len(in_span) == 1 and len(segs) > 1:
                seg_idx = in_span[0]
            elif len(in_span) > 1:
                warnings.append(f"The curve doubles back in x within the {where} ({crossed}): the result is "
                                f"the enclosed loop area along the whole path. Pick 'segment n' to integrate one "
                                f"branch.")
        elif mode == "whole":
            if len(in_span) > 1 and dsc_like:
                warnings.append(f"Whole path across {crossed}: heating and cooling contributions are combined "
                                f"(net loop area) — this is not a transformation enthalpy.")
        else:
            k = mode - 1
            if not 0 <= k < len(segs):
                raise ValueError(f"Segment {mode} does not exist (the trace has {len(segs)}).")
            if k not in in_span:
                raise ValueError(f"Segment {mode} ({names[k]}) has no data in the {where}.")
            seg_idx = k
        if seg_idx is None:
            res = PM.integrate_span(x, y, lo, hi, baseline)
            res.loop = len(in_span) > 1
            res.segment = "whole path (loop area)" if res.loop else "whole trace"
        else:
            a, b, _d = segs[seg_idx]
            res = PM.integrate_span(x[a:b], y[a:b], lo, hi, baseline)
            res.segment = names[seg_idx]
        res.segment_index = seg_idx
        res.warnings = warnings
        md = t.ds.metadata or {}
        rate = heating_rate_K_per_min if heating_rate_K_per_min else md.get("heating_rate_K_per_min")
        rate_src = "user" if heating_rate_K_per_min else ("dataset metadata" if rate else None)
        mass = mass_mg if mass_mg else md.get("mass_mg")
        xu, yu = self._units(t)
        area = res.area
        if seg_idx is not None and dsc_like and t.x_dim in _TEMPS and segs[seg_idx][2] == -1:
            area = -res.area  # cooling: ∫hf dt = ∫hf dT / (dT/dt) with dT/dt = −β
        qs, notes = PM.area_conversions(area, t.y_dim, yu, t.x_dim, xu, heating_rate_K_per_min=rate,
                                        mass_mg=mass)
        if res.loop:
            for q in qs:
                q.quantity = f"{q.quantity} (enclosed loop)"
        if shade:
            color = t.line.get_color()
            if res.monotonic:
                self._ov(t, "fill", dict(x=res.xs, y1=res.ys, y2=res.base), dict(color=color, alpha=0.22))
            else:
                self._ov(t, "polygon", dict(xs=res.xs, ys=res.ys), dict(color=color, alpha=0.22))
            if baseline == "linear":
                self._ov(t, "line", dict(xs=[res.xs[0], res.xs[-1]], ys=[res.base[0], res.base[-1]], label=None),
                         dict(color="#444444", linestyle="--", linewidth=1.0))
        tag = " (linear baseline)" if baseline == "linear" else ""
        items = [(q.quantity + tag, q.value, q.unit) for q in qs]
        if len(segs) > 1:
            items.append(("Segment", res.segment, ""))
        items += [("From x", res.xmin, xu), ("To x", res.xmax, xu)]
        if rate and t.x_dim in _TEMPS and t.y_dim in _HEAT:
            items.append(("Heating rate used", float(rate), "K/min"))
        prov = self._prov("Integrate", t, {
            "range": range_kind, "span": [lo, hi], "baseline": baseline, "segment_mode": str(segment),
            "segment_used": res.segment, "segments_detected": names, "heating_rate_K_per_min": rate,
            "heating_rate_source": rate_src, "mass_mg": mass, "area_raw": res.area,
            "cooling_sign_corrected": bool(area != res.area)}, warnings)
        self.last_provenance = prov
        self._rows("Integrate", t, items, prov, warnings)
        main = qs[-1]
        msg = f"∫ = {PM.fmt(qs[0].value)} {qs[0].unit}"
        if len(qs) > 1:
            msg += f"  =  {PM.fmt(main.value)} {main.unit}"
        if len(segs) > 1:
            msg += f"  [{res.segment}]"
        if warnings:
            self.panel.warn("  ".join([msg] + warnings))
        else:
            self.panel.statusMessage.emit("  ".join([msg] + notes))
        return res, qs

    # ------------------------------------------------------------------ fit / stats
    def _fit_clicked(self) -> None:
        try:
            self.fit()
        except ValueError as e:
            self.panel.statusMessage.emit(f"Linear fit: {e}")

    def fit(self, trace=None, *, xmin: float | None = None, xmax: float | None = None, draw: bool = True):
        """Least-squares line over the span (display units). Returns ``LinearFit``."""
        t = self._target(trace)
        if t is None:
            return None
        x, y = t.xy_display()
        if xmin is None and xmax is None and self.panel.display_span() is not None:
            xmin, xmax = self.panel.display_span()
        f = PM.linear_fit(x, y, xmin, xmax)
        xu, yu = self._units(t)
        items = [(q.quantity, q.value, q.unit) for q in PM.slope_conversions(f.slope, t.y_dim, yu, t.x_dim, xu)]
        items += [("Intercept", f.intercept, yu), ("R²", f.r2, ""), ("Points", f.n, ""),
                  ("From x", f.xmin, xu), ("To x", f.xmax, xu)]
        if draw:
            xs = np.array([f.xmin, f.xmax])
            self._ov(t, "line", dict(xs=xs, ys=f.slope * xs + f.intercept, label=None),
                     dict(color="#222222", linestyle="--", linewidth=1.3))
        warns = []
        if len(self.segments(t)[0]) > 1:
            warns.append("The trace doubles back in x: points of every segment inside the span were fitted together.")
        self._rows("Linear fit", t, items, self._prov("Linear fit", t, {"span": self._span_param(xmin, xmax),
                                                                       "points": f.n}, warns), warns)
        self.panel.statusMessage.emit(f"slope = {PM.fmt(f.slope)} {PM.quotient_unit(yu, xu)}, R² = {f.r2:.5f}")
        return f

    def _stats_clicked(self) -> None:
        try:
            self.stats()
        except ValueError as e:
            self.panel.statusMessage.emit(f"Statistics: {e}")

    def stats(self, trace=None, *, xmin: float | None = None, xmax: float | None = None) -> dict:
        """min/max/mean/std of y within the span (display units)."""
        t = self._target(trace)
        if t is None:
            return {}
        x, y = t.xy_display()
        if xmin is None and xmax is None and self.panel.display_span() is not None:
            xmin, xmax = self.panel.display_span()
        s = PM.span_stats(x, y, xmin, xmax)
        xu, yu = self._units(t)
        self._rows("Statistics", t, [("Min", s["min"], yu), ("x at min", s["x_at_min"], xu),
                                     ("Max", s["max"], yu), ("x at max", s["x_at_max"], xu),
                                     ("Mean", s["mean"], yu), ("Std dev", s["std"], yu),
                                     ("Range", s["range"], yu), ("Points", s["n"], "")],
                   self._prov("Statistics", t, {"span": self._span_param(xmin, xmax)}))
        self.panel.statusMessage.emit(f"min {PM.fmt(s['min'])}, max {PM.fmt(s['max'])}, mean {PM.fmt(s['mean'])} {yu}")
        return s
