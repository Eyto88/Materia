"""Results table of the Plot tab (module B): Tool | Trace | Quantity | Value | Unit.

* Numbers are shown and copied with :func:`app_settings.format_value` (Preferences
  "sig_figs"); CSV uses the same formatting unless ``csv_full_precision`` is set.
* Warning rows (``warning=True`` or a quantity starting with "⚠") are amber.
* Each row may carry the ``provenance`` dict of the tool run that produced it;
  CSV exports start with ``core.provenance.provenance_lines`` header lines plus
  one ``# [k] …`` line per distinct tool run (its parameters and warnings).
"""
from __future__ import annotations

import csv
import json

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from labstudio.core.provenance import provenance, provenance_lines

from .app_settings import format_value, settings

WARN_BG = "#FFE3A3"
WARN_FG = "#5A3B00"


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)


class ResultsTable(QWidget):
    """Tool | Trace | Quantity | Value | Unit with copy / export CSV / clear."""

    HEADERS = ["Tool", "Trace", "Quantity", "Value", "Unit"]
    KEYS = ("tool", "trace", "quantity", "value", "unit")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[dict] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("<b>Results</b>"))
        bar.addStretch(1)
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.setToolTip("Copy the selected rows (or all rows) to the clipboard as tab-separated text "
                                 "— pastes straight into Excel. Numbers use the significant figures set in "
                                 "Preferences.")
        self.btn_csv = QPushButton("Export CSV…")
        self.btn_csv.setToolTip("Save all results to a CSV file with a provenance header (parameters, warnings, "
                                "app version).")
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setToolTip("Remove all rows from the results table.")
        for b in (self.btn_copy, self.btn_csv, self.btn_clear):
            bar.addWidget(b)
        lay.addLayout(bar)
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setMinimumHeight(60)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        for i, w in enumerate((90, 170, 220, 110, 80)):
            self.table.setColumnWidth(i, w)
        lay.addWidget(self.table)
        self.btn_copy.clicked.connect(self.copy_to_clipboard)
        self.btn_csv.clicked.connect(self._export_dialog)
        self.btn_clear.clicked.connect(self.clear)

    # ------------------------------------------------------------------ rows
    @staticmethod
    def normalize(row: dict) -> dict:
        low = {str(k).lower(): v for k, v in row.items()}
        q = str(low.get("quantity", low.get("name", "")))
        out = {"tool": str(low.get("tool", "")), "trace": str(low.get("trace", "")), "quantity": q,
               "value": low.get("value", ""), "unit": str(low.get("unit", "") or "")}
        out["warning"] = bool(low.get("warning", False)) or q.startswith("⚠")
        if low.get("provenance") is not None:
            out["provenance"] = low["provenance"]
        return out

    def add_rows(self, rows: list[dict], provenance_rec: dict | None = None) -> list[dict]:
        """Append rows; ``provenance_rec`` is attached to rows that have none."""
        norm = [self.normalize(r) for r in rows]
        for r in norm:
            if provenance_rec is not None and "provenance" not in r:
                r["provenance"] = provenance_rec
            self.rows.append(r)
            i = self.table.rowCount()
            self.table.insertRow(i)
            v = r["value"]
            for j, txt in enumerate((r["tool"], r["trace"], r["quantity"], format_value(v), r["unit"])):
                it = QTableWidgetItem(txt)
                if j == 3:
                    if _is_number(v):
                        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    it.setToolTip(repr(v) if _is_number(v) else str(v))
                if r["warning"]:
                    it.setBackground(QBrush(QColor(WARN_BG)))
                    it.setForeground(QBrush(QColor(WARN_FG)))
                    if j == 3:
                        it.setToolTip(str(v))
                self.table.setItem(i, j, it)
        if norm:
            self.table.scrollToBottom()
        return norm

    def clear(self) -> None:
        self.rows.clear()
        self.table.setRowCount(0)

    def warnings(self) -> list[str]:
        return [str(r["value"]) for r in self.rows if r.get("warning")]

    # ------------------------------------------------------------------ copy
    def as_text(self, only_selected: bool = False) -> str:
        """Tab-separated text (numbers via format_value)."""
        sel = sorted({i.row() for i in self.table.selectedIndexes()}) if only_selected else []
        rows = [self.rows[i] for i in sel] if sel else self.rows
        out = ["\t".join(self.HEADERS)]
        for r in rows:
            out.append("\t".join(format_value(r[k]) if k == "value" else str(r[k]) for k in self.KEYS))
        return "\n".join(out)

    def copy_to_clipboard(self) -> None:
        QGuiApplication.clipboard().setText(self.as_text(only_selected=True))

    # ------------------------------------------------------------------ CSV
    def provenance_record(self) -> dict:
        """Provenance of the whole table (analysis "Plot tools")."""
        full = bool(settings().get("csv_full_precision"))
        runs = self._runs()
        warns = []
        for r in self.rows:
            if r.get("warning") and str(r["value"]) not in warns:
                warns.append(str(r["value"]))
        for rec in runs:
            for w in rec.get("warnings", []) or []:
                if w not in warns:
                    warns.append(w)
        return provenance("Plot tools", {"rows": len(self.rows), "tool_runs": len(runs),
                                         "sig_figs": settings().sig_figs, "csv_full_precision": full},
                          extra={"warnings": warns})

    def _runs(self) -> list[dict]:
        seen: list[dict] = []
        for r in self.rows:
            p = r.get("provenance")
            if p is not None and not any(p is q for q in seen):
                seen.append(p)
        return seen

    def export_csv(self, path: str) -> None:
        """CSV with provenance header lines; values formatted unless csv_full_precision."""
        full = bool(settings().get("csv_full_precision"))
        lines = provenance_lines(self.provenance_record())
        for k, rec in enumerate(self._runs(), 1):
            desc = f"# [{k}] {rec.get('analysis', '')}"
            ds = rec.get("dataset", {}).get("name") if isinstance(rec.get("dataset"), dict) else None
            if ds:
                desc += f" · dataset '{ds}'"
            if rec.get("source_file"):
                desc += f" · source {rec['source_file']} sha256={rec.get('source_sha256')}"
            desc += " · parameters " + json.dumps(rec.get("parameters", {}), ensure_ascii=False)
            if rec.get("warnings"):
                desc += " · warnings: " + "; ".join(map(str, rec["warnings"]))
            lines.append(desc)
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            for ln in lines:
                fh.write(ln.replace("\n", " ") + "\n")
            w = csv.writer(fh)
            w.writerow(self.HEADERS)
            for r in self.rows:
                v = r["value"]
                if _is_number(v):
                    v = repr(float(v)) if full and not isinstance(v, (int, np.integer)) else format_value(v)
                w.writerow([r["tool"], r["trace"], r["quantity"], v, r["unit"]])

    def _export_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export results", "results.csv", "CSV (*.csv)")
        if path:
            self.export_csv(path)
