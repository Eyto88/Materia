"""Shared widgets/helpers of the analysis dock (tables with provenance, pickers, tab base class)."""
from __future__ import annotations

import csv
import logging
import math
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from ..core.dataset import Dataset, DatasetStore
from ..core.provenance import provenance, provenance_lines
from .app_settings import format_value, settings

log = logging.getLogger("labstudio.analysis")

NONE = "(none)"
AMBER = "#fff3cd"
AMBER_TEXT = "#8a5a00"
ERROR_TEXT = "#b00020"
OK_TEXT = "#2e7d32"


def fmt(v: Any) -> str:
    """Table/clipboard text of a value: significant-figure formatting (Preferences ▸ sig figs)."""
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "–"
    return format_value(f)


def csv_value(v: Any) -> Any:
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return ""
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return int(v)
    if settings().get("csv_full_precision"):
        return repr(f)
    return format_value(f)


def dspin(value: float, lo: float, hi: float, decimals: int = 3, step: float = 0.1, suffix: str = "",
          tip: str = "", special: str | None = None) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    if suffix:
        w.setSuffix(f" {suffix}")
    if special:
        w.setSpecialValueText(special)
    w.setToolTip(tip)
    w.setKeyboardTracking(False)
    w.setMinimumWidth(60)
    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return w


def combo(items: list[str], tip: str = "", current: int = 0) -> QComboBox:
    c = QComboBox()
    c.addItems(items)
    c.setCurrentIndex(current)
    c.setToolTip(tip)
    c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(8)
    return c


def button(text: str, tip: str = "", slot=None) -> QPushButton:
    b = QPushButton(text)
    b.setToolTip(tip or text)
    b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    b.setMinimumWidth(40)
    if slot is not None:
        b.clicked.connect(slot)
    return b


def hrow(*widgets, stretch_last: bool = False) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        lay.addWidget(x, 1)
    return w


def scroll(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QScrollArea.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    sa.setWidget(inner)
    sa.setMinimumWidth(0)
    return sa


class SummaryTable(QWidget):
    """Read-only table with Copy / Export CSV; warning rows in amber; CSV starts with provenance lines."""

    def __init__(self, headers: list[str] | None = None, parent=None, min_height: int = 120):
        super().__init__(parent)
        self.headers = headers or ["Quantity", "Value", "Unit"]
        self._rows: list[list[Any]] = []
        self._warn_rows: set[int] = set()
        self.provenance: dict | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(min_height)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setWordWrap(False)
        self.table.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setStretchLastSection(True)
        hh.setMinimumSectionSize(30)
        lay.addWidget(self.table)
        self.btn_copy = button("Copy", "Copy the table (tab separated, significant figures from Preferences)",
                               self.copy)
        self.btn_csv = button("Export CSV…", "Save this table as CSV (provenance comment lines first)",
                              self._export_dialog)
        lay.addWidget(hrow(self.btn_copy, self.btn_csv))

    def set_provenance(self, rec: dict | None) -> None:
        self.provenance = rec

    def set_rows(self, rows: list[list[Any]], headers: list[str] | None = None, *,
                 warn_rows: set[int] | None = None, tooltips: dict[int, str] | None = None) -> None:
        if headers is not None:
            self.headers = list(headers)
            self.table.setColumnCount(len(self.headers))
            self.table.setHorizontalHeaderLabels(self.headers)
        self._rows = [list(r) for r in rows]
        self._warn_rows = set(warn_rows or ())
        tooltips = tooltips or {}
        self.table.setRowCount(len(rows))
        amber = QBrush(QColor(AMBER))
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                it = QTableWidgetItem(fmt(v))
                if not isinstance(v, str):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if isinstance(v, (float, np.floating)):
                        it.setToolTip(repr(float(v)))
                if i in self._warn_rows:
                    it.setBackground(amber)
                    it.setToolTip(" ".join(str(x) for x in r if isinstance(x, str)))
                if i in tooltips:
                    it.setToolTip(tooltips[i])
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()
        hh = self.table.horizontalHeader()
        for j in range(self.table.columnCount()):
            if hh.sectionSize(j) > 220:
                hh.resizeSection(j, 220)

    def rows(self) -> list[list[Any]]:
        return [list(r) for r in self._rows]

    def warn_rows(self) -> set[int]:
        return set(self._warn_rows)

    def export_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            if self.provenance:
                for ln in provenance_lines(self.provenance):
                    fh.write(ln + "\n")
            w = csv.writer(fh)
            w.writerow(self.headers)
            for r in self._rows:
                w.writerow([csv_value(v) for v in r])

    def text(self) -> str:
        return "\t".join(self.headers) + "\n" + "\n".join("\t".join(fmt(v) for v in r) for r in self._rows)

    def copy(self) -> None:
        QApplication.clipboard().setText(self.text())

    def _export_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export table", "analysis.csv", "CSV (*.csv)")
        if path:
            self.export_csv(path)


class DatasetCombo(QComboBox):
    """Dataset picker: datasets of the preferred kinds first (others still selectable)."""

    def __init__(self, store: DatasetStore, kinds: tuple[str, ...], parent=None, optional: bool = False,
                 accept=None):
        super().__init__(parent)
        self.store = store
        self.kinds = kinds
        self.optional = optional
        self.accept = accept
        self.setToolTip("Dataset to analyse (datasets of the matching kind are listed first)")
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(12)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def refresh(self) -> None:
        cur = self.currentData()
        self.blockSignals(True)
        self.clear()
        if self.optional:
            self.addItem(NONE, None)
        items = [d for d in self.store if self.accept is None or self.accept(d)]
        pref = [d for d in items if d.kind in self.kinds] + [d for d in items if d.kind not in self.kinds]
        for d in pref:
            self.addItem(f"{d.name}  [{d.kind}]", d.id)
            self.setItemData(self.count() - 1, f"{d.name} [{d.kind}]", Qt.ToolTipRole)
        idx = self.findData(cur) if cur is not None else -1
        self.setCurrentIndex(idx if idx >= 0 else (0 if self.count() else -1))
        self.blockSignals(False)
        if self.currentData() != cur:
            self.currentIndexChanged.emit(self.currentIndex())

    def dataset(self) -> Dataset | None:
        i = self.currentData()
        return self.store.get(i) if i is not None else None

    def select(self, ds: Dataset | None) -> None:
        if ds is None:
            return
        i = self.findData(ds.id)
        if i >= 0:
            self.setCurrentIndex(i)


class ColumnCombo(QComboBox):
    """Column picker showing ``name (unit)``; pre-selects the first column of preferred dimensions.

    ``strict``: if no column of the preferred dimensions exists, select '(none)' instead of an
    arbitrary column (e.g. never default the strain picker to Time)."""

    def __init__(self, dims: tuple[str, ...], optional: bool = False, tip: str = "", parent=None,
                 strict: bool = False):
        super().__init__(parent)
        self.dims = dims
        self.optional = optional or strict
        self.strict = strict
        self.setToolTip(tip)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(10)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def populate(self, ds: Dataset | None, prefer_name: str | None = None) -> None:
        self.blockSignals(True)
        self.clear()
        if self.optional:
            self.addItem(NONE, None)
        if ds is not None:
            for c in ds.columns():
                self.addItem(ds.label(c), c)
            pick = None
            if prefer_name and prefer_name in ds.channels:
                pick = prefer_name
            else:
                for d in self.dims:
                    cols = ds.columns_of(d)
                    if cols:
                        pick = cols[0]
                        break
            if pick is not None:
                self.setCurrentIndex(self.findData(pick))
            elif not self.optional and self.count():
                self.setCurrentIndex(0)
            else:
                self.setCurrentIndex(0)
        self.blockSignals(False)

    def column(self) -> str | None:
        return self.currentData()

    def set_column(self, col: str | None) -> None:
        i = self.findData(col)
        if i >= 0:
            self.setCurrentIndex(i)


class TabBase(QWidget):
    """Common behaviour: status line (ok / amber warnings / error), result rows, overlay labels,
    provenance and widget-state (de)serialisation."""
    KINDS: tuple[str, ...] = ()
    TOOL = ""
    KEY = ""
    STATE_WIDGETS: tuple[str, ...] = ()     # attribute names saved by get_state / defaults
    DEFAULT_WIDGETS: tuple[str, ...] = ()   # subset used for analysis defaults (presets)

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.plot = panel.plot_panel
        self.store = panel.store
        self.last_result = None
        self.last_provenance: dict | None = None
        self.last_warnings: list[str] = []
        self._ran = False
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status.setStyleSheet("color: #555;")
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    # ---- status
    def _msg(self, text: str, error: bool = False, warnings: list[str] | None = None) -> None:
        warnings = list(warnings or [])
        if error:
            self.status.setText(text)
            self.status.setStyleSheet(f"color: {ERROR_TEXT};")
        elif warnings:
            self.status.setText(text + "\n⚠ " + "\n⚠ ".join(warnings))
            self.status.setStyleSheet(f"color: {AMBER_TEXT}; background: {AMBER}; padding: 2px;")
        else:
            self.status.setText(text)
            self.status.setStyleSheet(f"color: {OK_TEXT};")

    def _fail(self, what: str, exc: BaseException) -> None:
        log.exception("%s failed", what, exc_info=exc)
        self._msg(f"{what} failed: {exc.__class__.__name__}: {exc}", True)

    def safe(self, what: str, fn, *a, **k):
        """Call ``fn``; every exception is logged and shown in the status line (never silent)."""
        try:
            return fn(*a, **k)
        except Exception as exc:  # noqa: BLE001
            self._fail(what, exc)
            return None

    # ---- results / provenance
    def _record(self, analysis: str, params: dict, ds: Dataset | None, warnings: list[str],
                tables: list[SummaryTable] = ()) -> dict:
        rec = provenance(analysis, params, ds, extra={"warnings": list(warnings)})
        self.last_provenance = rec
        self.last_warnings = list(warnings)
        for t in tables:
            t.set_provenance(rec)
        return rec

    def _results(self, ds: Dataset, rows: list[tuple[str, float, str]], warnings: list[str] = ()) -> None:
        out = [{"Tool": self.TOOL, "Trace": ds.name, "Quantity": q, "Value": v, "Unit": u} for q, v, u in rows]
        out += [{"Tool": self.TOOL, "Trace": ds.name, "Quantity": "Warning", "Value": w, "Unit": ""}
                for w in warnings]
        self.plot.add_results(out)

    @staticmethod
    def table_rows(rows, warnings: list[str]) -> tuple[list[list], set[int]]:
        """Warning rows (amber) first, then the result rows."""
        out = [["⚠ Warning", w, ""] for w in warnings] + [list(r) for r in rows]
        return out, set(range(len(warnings)))

    # ---- overlays
    def label(self, x: float, y: float, text: str, color=None, marker: str | None = "o") -> None:
        """Marker + collision-free label (``plot_panel.add_label`` when available)."""
        if not (np.isfinite(x) and np.isfinite(y)):
            return
        if hasattr(self.plot, "add_label"):
            if marker:
                self.plot.add_marker(x, y, "", color=color, marker=marker)
            try:
                self.plot.add_label(x, y, text, color=color)
            except TypeError:
                self.plot.add_label(x, y, text)
        else:
            self.plot.add_marker(x, y, text, color=color, marker=marker or "o")

    def _ensure_trace(self, ds: Dataset, x: str, y: str, clear: bool, **style):
        if clear:
            self.plot.clear()
        else:
            self.plot.clear_overlays()
            for tr in getattr(self.plot, "traces", []):
                if getattr(tr, "ds", None) is ds and tr.x == x and tr.y == y:
                    return tr
        return self.plot.add_trace(ds, x, y, label=ds.name, **style)

    # ---- state
    def _widget_value(self, w):
        if isinstance(w, (QDoubleSpinBox, QSpinBox)):
            return w.value()
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentText()
        if isinstance(w, QLineEdit):
            return w.text()
        return None

    def _set_widget_value(self, w, v) -> None:
        if v is None:
            return
        if isinstance(w, QDoubleSpinBox):
            w.setValue(float(v))
        elif isinstance(w, QSpinBox):
            w.setValue(int(v))
        elif isinstance(w, QCheckBox):
            w.setChecked(bool(v))
        elif isinstance(w, QComboBox):
            if isinstance(v, int) and not isinstance(v, bool):
                w.setCurrentIndex(v)
            else:
                i = w.findText(str(v))
                if i >= 0:
                    w.setCurrentIndex(i)
        elif isinstance(w, QLineEdit):
            w.setText(str(v))

    def widget_values(self, names) -> dict:
        return {n: self._widget_value(getattr(self, n)) for n in names if hasattr(self, n)}

    def apply_values(self, values: dict) -> None:
        for n, v in (values or {}).items():
            w = getattr(self, n, None)
            if w is not None:
                try:
                    self._set_widget_value(w, v)
                except (TypeError, ValueError):
                    log.warning("could not restore %s=%r", n, v)

    def defaults(self) -> dict:
        return self.widget_values(self.DEFAULT_WIDGETS or self.STATE_WIDGETS)

    def get_state(self) -> dict:
        st = {"params": self.widget_values(self.STATE_WIDGETS), "ran": bool(self._ran)}
        dc = getattr(self, "ds_combo", None)
        if dc is not None:
            st["dataset"] = dc.currentData()
        st["columns"] = {n: getattr(self, n).column() for n in dir(self)
                         if n.startswith("c_") and isinstance(getattr(self, n, None), ColumnCombo)}
        return st

    def set_state(self, state: dict, id_map: dict) -> None:
        state = state or {}
        dc = getattr(self, "ds_combo", None)
        old = state.get("dataset")
        if dc is not None and old is not None and int(old) in id_map:
            dc.refresh()
            dc.select(id_map[int(old)])
            if hasattr(self, "_on_ds"):
                self._on_ds()
        self.apply_values(state.get("params", {}))
        for n, col in (state.get("columns") or {}).items():
            c = getattr(self, n, None)
            if isinstance(c, ColumnCombo) and col is not None:
                c.set_column(col)
