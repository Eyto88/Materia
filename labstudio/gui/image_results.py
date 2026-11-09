"""Results table and small plot widget used by the Micrographs workspace."""
from __future__ import annotations

import csv
import math

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QPushButton,
                               QTableView, QVBoxLayout, QWidget, QHeaderView, QCheckBox)


WARNING_KEY = "Warning"


def format_value(v) -> str:
    """Table / clipboard text of a value: numbers via ``app_settings.format_value`` (significant
    figures from Preferences), booleans as yes/no, ±∞ as symbols."""
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return "NaN"
        if math.isinf(f):
            return "∞" if f > 0 else "-∞"
        from labstudio.gui.app_settings import format_value as _fv

        return _fv(f)
    return str(v)


def csv_value(v):
    """CSV cell: formatted with significant figures unless Preferences ▸ csv_full_precision."""
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        try:
            from labstudio.gui.app_settings import settings

            full = bool(settings().get("csv_full_precision"))
        except Exception:  # noqa: BLE001
            full = False
        f = float(v)
        if full or math.isnan(f) or math.isinf(f):
            return repr(f)
        return format_value(f)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


class DictTableModel(QAbstractTableModel):
    """Table model over a list of dicts; the column set is the ordered union of keys."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[dict] = []
        self.columns: list[str] = []

    def rowCount(self, parent=QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(self.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            v = self.rows[index.row()].get(self.columns[index.column()])
            return format_value(v) if role == Qt.DisplayRole else str(v)
        if role == Qt.TextAlignmentRole:
            v = self.rows[index.row()].get(self.columns[index.column()])
            if isinstance(v, (int, float, np.integer, np.floating)):
                return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.BackgroundRole and self.rows[index.row()].get(WARNING_KEY):
            from PySide6.QtGui import QColor

            return QColor("#fff4ce")
        if role == Qt.ForegroundRole and self.rows[index.row()].get(WARNING_KEY):
            from PySide6.QtGui import QColor

            return QColor("#7a4f00")
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.columns[section] if section < len(self.columns) else None
        return str(section + 1)

    def add_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        new_cols = [k for r in rows for k in r.keys()]
        cols = list(self.columns)
        for c in new_cols:
            if c not in cols:
                cols.append(c)
        self.beginResetModel()
        self.columns = cols
        self.rows.extend(dict(r) for r in rows)
        self.endResetModel()

    def clear(self) -> None:
        self.beginResetModel()
        self.rows, self.columns = [], []
        self.endResetModel()


class ResultsTable(QWidget):
    """ImageJ-like results table with Copy / Export CSV / Clear."""

    changed = Signal()

    def __init__(self, title: str = "Results", parent=None):
        super().__init__(parent)
        self.title = title
        self.provenance: list[dict] = []   # one record per analysis run that wrote rows
        self.model = DictTableModel(self)
        self.view = QTableView()
        self.view.setModel(self.model)
        self.view.setAlternatingRowColors(True)
        self.view.setSelectionBehavior(QTableView.SelectRows)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.horizontalHeader().setDefaultSectionSize(92)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.verticalHeader().setDefaultSectionSize(20)
        self.count_label = QLabel("0 rows")
        self.append_check = QCheckBox("Append")
        self.append_check.setChecked(True)
        self.append_check.setToolTip("Append new results; uncheck to replace the table on every run")
        b_copy = QPushButton("Copy")
        b_copy.setToolTip("Copy the table (tab separated) to the clipboard")
        b_copy.clicked.connect(self.copy_to_clipboard)
        b_csv = QPushButton("Export CSV…")
        b_csv.clicked.connect(self._export_dialog)
        b_clear = QPushButton("Clear")
        b_clear.clicked.connect(self.clear)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.count_label)
        top.addStretch(1)
        for w in (self.append_check, b_copy, b_csv, b_clear):
            top.addWidget(w)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(top)
        lay.addWidget(self.view)

    @property
    def rows(self) -> list[dict]:
        return self.model.rows

    @property
    def columns(self) -> list[str]:
        return self.model.columns

    def add_rows(self, rows: list[dict], replace: bool | None = None,
                 provenance: dict | None = None) -> None:
        if replace is None:
            replace = not self.append_check.isChecked()
        if replace:
            self.model.clear()
            self.provenance = []
        if provenance is not None and not any(p is provenance for p in self.provenance):
            self.provenance.append(provenance)
        self.model.add_rows(rows)
        self.count_label.setText(f"{len(self.model.rows)} rows")
        if len(self.model.rows) <= 3000:
            self.view.resizeColumnsToContents()
            hh = self.view.horizontalHeader()
            for i in range(self.model.columnCount()):
                hh.resizeSection(i, min(max(hh.sectionSize(i), 60), 320))
        self.view.scrollToBottom()
        self.changed.emit()

    def add_warnings(self, warnings: list[str], context: dict | None = None,
                     provenance: dict | None = None) -> None:
        """Append amber warning rows (``{'Warning': text, **context}``)."""
        if warnings:
            self.add_rows([{**(context or {}), WARNING_KEY: str(w)} for w in warnings], replace=False,
                          provenance=provenance)

    @property
    def warnings(self) -> list[str]:
        return [r[WARNING_KEY] for r in self.rows if r.get(WARNING_KEY)]

    def clear(self) -> None:
        self.model.clear()
        self.provenance = []
        self.count_label.setText("0 rows")
        self.changed.emit()

    def to_text(self, sep: str = "\t") -> str:
        lines = [sep.join(self.columns)]
        for r in self.rows:
            lines.append(sep.join(format_value(r.get(c)) for c in self.columns))
        return "\n".join(lines)

    def copy_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self.to_text())

    def provenance_header(self) -> list[str]:
        """'# …' lines (``core.provenance.provenance_lines``) of every analysis in the table."""
        from labstudio.core.provenance import provenance_lines

        out: list[str] = []
        for rec in self.provenance:
            lines = provenance_lines(rec)
            extra = []
            if rec.get("image"):
                extra.append(f"# image: {rec['image']}" + (f" sha256={rec['image_sha256']}"
                                                           if rec.get("image_sha256") else ""))
            if rec.get("calibration"):
                extra.append(f"# calibration: {rec['calibration']} · source: {rec.get('calibration_source')}")
            out += lines[:1] + extra + lines[1:]
        return out

    def export_csv(self, path: str) -> None:
        """Write the table to ``path`` (UTF-8 with BOM so Excel shows µ and ² correctly).

        The file starts with provenance comment lines (analysis, parameters, calibration and its
        source, image file + SHA-256, warnings) for every analysis in the table; numbers use the
        significant figures from Preferences unless 'csv_full_precision' is set."""
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            for line in self.provenance_header():
                fh.write(line.replace("\n", " ") + "\n")
            w = csv.writer(fh)
            w.writerow(self.columns)
            for r in self.rows:
                w.writerow([csv_value(r.get(c)) for c in self.columns])

    def _export_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, f"Export {self.title}", f"{self.title.lower()}.csv",
                                              "CSV (*.csv)")
        if path:
            self.export_csv(path)


class PlotWidget(QWidget):
    """Small matplotlib canvas for line profiles, histograms and size distributions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(5, 2.2), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.data_columns: list[str] = []
        self.data: list[np.ndarray] = []
        self.title_label = QLabel("No plot yet — use Profile, Histogram or a size distribution.")
        self.log_check = QCheckBox("Log y")
        self.log_check.toggled.connect(self._set_log)
        b_csv = QPushButton("Export data CSV…")
        b_csv.clicked.connect(self._export_csv_dialog)
        b_png = QPushButton("Export PNG…")
        b_png.clicked.connect(self._export_png_dialog)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.title_label, 1)
        top.addWidget(self.log_check)
        top.addWidget(b_csv)
        top.addWidget(b_png)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addLayout(top)
        lay.addWidget(self.canvas, 1)

    def _set_log(self, on: bool) -> None:
        self.ax.set_yscale("log" if on else "linear")
        self.canvas.draw_idle()

    def plot_line(self, x, y, xlabel: str, ylabel: str, title: str = "") -> None:
        self.ax.clear()
        self.ax.plot(x, y, color="#1f77b4", lw=1.2)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(alpha=0.3)
        self.title_label.setText(title)
        self.data_columns = [xlabel, ylabel]
        self.data = [np.asarray(x), np.asarray(y)]
        self._set_log(self.log_check.isChecked())

    def plot_hist(self, counts, edges, xlabel: str, ylabel: str = "Count", title: str = "",
                  color: str = "#555555", vlines: list[float] | None = None) -> None:
        self.ax.clear()
        edges = np.asarray(edges, float)
        self.ax.stairs(counts, edges, fill=True, color=color, alpha=0.8)
        for v in vlines or []:
            self.ax.axvline(v, color="#d62728", lw=1)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(alpha=0.3)
        self.title_label.setText(title)
        centres = (edges[:-1] + edges[1:]) / 2
        self.data_columns = [f"{xlabel} (bin centre)", ylabel]
        self.data = [centres, np.asarray(counts)]
        self._set_log(self.log_check.isChecked())

    def export_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(self.data_columns)
            for row in zip(*self.data):
                w.writerow([float(v) for v in row])

    def _export_csv_dialog(self) -> None:
        if not self.data:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export plot data", "plot_data.csv", "CSV (*.csv)")
        if path:
            self.export_csv(path)

    def _export_png_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export plot", "plot.png", "PNG (*.png)")
        if path:
            self.figure.savefig(path, dpi=300)
