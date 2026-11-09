"""Import dialog: preview an instrument export, map its columns to dimensions/units,
enter specimen metadata and build a canonical-unit :class:`Dataset`.

Usable headlessly: construct ``ImportDialog(path)``, tweak widgets programmatically,
then call :meth:`ImportDialog.build` (or :meth:`accept` followed by :meth:`dataset`).

No silent guesses: columns whose unit/quantity was guessed (spelled-out unit word, no unit
in the file, ambiguous M/m prefix ...) are highlighted amber and have an unticked
"Units OK" box. Importing is still possible, but every unconfirmed guess is listed in
``Dataset.metadata['warnings']``. For DSC files without a heating rate the rate field is
highlighted and the default (10 K/min) is flagged unless the user confirms or edits it.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..core import data_io as D
from ..core import units as U
from ..core.dataset import DATASET_KINDS, Dataset

PREVIEW_ROWS = 200
AMBER = "#ffd27f"
AMBER_TEXT = "#9a5b00"

KIND_LABELS = {
    "generic": "Generic (x–y data)",
    "dsc": "DSC",
    "tensile": "Tensile (to failure)",
    "compression": "Compression",
    "cyclic": "Cyclic / superelastic",
}

SECTION_AREA, SECTION_ROUND, SECTION_RECT = 0, 1, 2


def _spin(maximum: float = 1e9, decimals: int = 4, tip: str = "") -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(0.0, maximum)
    sb.setDecimals(decimals)
    sb.setSpecialValueText("—")      # 0 means "not set"
    sb.setKeyboardTracking(False)
    sb.setMinimumWidth(70)
    if tip:
        sb.setToolTip(tip)
    return sb


def _unit_combo(dimension: str, current: str | None = None, tip: str = "") -> QComboBox:
    cb = QComboBox()
    cb.addItems(U.units_for(dimension))
    cb.setCurrentText(current or U.CANONICAL[dimension])
    if tip:
        cb.setToolTip(tip)
    return cb


def _with_unit(spin: QDoubleSpinBox, combo: QComboBox) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(spin, 1)
    lay.addWidget(combo)
    return w


class ImportDialog(QDialog):
    """Preview + column mapping + metadata for one data file.

    Attributes useful for scripting/tests: ``raw`` (RawTable), ``include_checks``, ``name_edits``,
    ``dim_combos``, ``unit_combos``, ``confirm_checks`` (one per source column), ``kind_combo``,
    ``sheet_combo`` (spreadsheets with several sheets), ``section_mode``,
    ``area_spin``/``area_unit``, ``diam_spin``/``diam_unit``, ``width_spin``/``width_unit``,
    ``thick_spin``/``thick_unit``, ``gauge_spin``/``gauge_unit``, ``mass_spin``/``mass_unit``,
    ``rate_spin``/``rate_unit``/``rate_confirm_check``, ``preset_combo``, ``save_defaults_check``,
    ``name_edit``, ``apply_all_check``, ``status_label`` (amber warnings preview).
    """

    COL_INCLUDE, COL_SOURCE, COL_NAME, COL_DIM, COL_UNIT, COL_CONFIRM, COL_PREVIEW = range(7)

    #: application-wide provider of analysis defaults (set once by the analysis panel / main window)
    _default_analysis_defaults_provider: Callable[[], dict] | None = None

    def __init__(self, path: str | os.PathLike, parent: QWidget | None = None, *,
                 raw: D.RawTable | None = None, presets_dir: str | os.PathLike | None = None,
                 show_apply_all: bool = False, sheet: str | int | None = None) -> None:
        super().__init__(parent)
        self.path = str(path)
        self.presets_dir = presets_dir
        self.raw = raw if raw is not None else D.read_table(path, sheet=sheet)
        self._dataset: Dataset | None = None
        self._analysis_defaults: dict[str, Any] = {}
        self._analysis_defaults_provider: Callable[[], dict] | None = None
        self.setWindowTitle(f"Import data — {Path(self.path).name}")
        self.setSizeGripEnabled(True)
        self.resize(1100, 650)

        self.include_checks: list[QCheckBox] = []
        self.name_edits: list[QLineEdit] = []
        self.dim_combos: list[QComboBox] = []
        self.unit_combos: list[QComboBox] = []
        self.confirm_checks: list[QCheckBox] = []
        self._guessed: list[bool] = []
        self._reasons: list[str] = []
        self._sources: list[str] = []
        self._populating = False

        self._build_ui(show_apply_all)
        self._load_raw()

    # ------------------------------------------------------------------ UI
    def _build_ui(self, show_apply_all: bool) -> None:
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.info_label = QLabel()
        self.info_label.setTextFormat(Qt.RichText)
        self.info_label.setWordWrap(True)
        self.info_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        top.addWidget(self.info_label, 1)
        self.sheet_label = QLabel("Sheet:")
        self.sheet_combo = QComboBox()
        self.sheet_combo.setToolTip("This workbook has several sheets — pick the one holding the data.")
        top.addWidget(self.sheet_label)
        top.addWidget(self.sheet_combo)
        root.addLayout(top)

        main = QSplitter(Qt.Horizontal)
        main.setChildrenCollapsible(False)
        root.addWidget(main, 1)

        # ---- left: preamble + preview + mapping
        left = QSplitter(Qt.Vertical)
        left.setChildrenCollapsible(False)
        self.pre_box = QGroupBox("File preamble / instrument metadata")
        self.pre_box.setCheckable(True)
        pre_lay = QVBoxLayout(self.pre_box)
        pre_lay.setContentsMargins(4, 4, 4, 4)
        self.preamble_view = QPlainTextEdit()
        self.preamble_view.setReadOnly(True)
        self.preamble_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.preamble_view.setMinimumHeight(30)
        pre_lay.addWidget(self.preamble_view)
        self.pre_box.toggled.connect(self.preamble_view.setVisible)
        left.addWidget(self.pre_box)

        self.prev_box = QGroupBox("Preview")
        prev_lay = QVBoxLayout(self.prev_box)
        prev_lay.setContentsMargins(4, 4, 4, 4)
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.setMinimumHeight(60)
        self.preview_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        prev_lay.addWidget(self.preview_table)
        left.addWidget(self.prev_box)

        map_box = QGroupBox("Column mapping")
        map_lay = QVBoxLayout(map_box)
        map_lay.setContentsMargins(4, 4, 4, 4)
        self.map_table = QTableWidget(0, 7)
        self.map_table.setHorizontalHeaderLabels(
            ["Import", "Source column", "Name", "Quantity", "Unit in file", "Units OK",
             "First value → canonical"])
        hh = self.map_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setStretchLastSection(True)
        self.map_table.horizontalHeaderItem(self.COL_CONFIRM).setToolTip(
            "Tick once you have checked the quantity and unit of this column. Amber rows were guessed "
            "(spelled-out unit, no unit in the file, ambiguous prefix); unconfirmed guesses are recorded "
            "as warnings on the dataset.")
        self.map_table.verticalHeader().setVisible(False)
        self.map_table.setMinimumHeight(80)
        self.map_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        map_lay.addWidget(self.map_table)
        row = QHBoxLayout()
        self.confirm_all_btn = QPushButton("Confirm all units")
        self.confirm_all_btn.setToolTip("Mark every column's quantity and unit as checked.")
        self.confirm_all_btn.clicked.connect(self.confirm_all)
        row.addWidget(self.confirm_all_btn)
        row.addStretch(1)
        map_lay.addLayout(row)
        left.addWidget(map_box)
        left.setStretchFactor(0, 1)
        left.setStretchFactor(1, 3)
        left.setStretchFactor(2, 3)
        main.addWidget(left)

        # ---- right: kind, metadata, presets (scrollable so the dialog fits small screens)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 0, 4, 0)
        gen_box = QGroupBox("Dataset")
        gf = QFormLayout(gen_box)
        self.name_edit = QLineEdit(Path(self.path).stem)
        self.name_edit.setToolTip("Name shown in the dataset list and plot legends.")
        gf.addRow("Name", self.name_edit)
        self.kind_combo = QComboBox()
        for k in DATASET_KINDS:
            self.kind_combo.addItem(KIND_LABELS.get(k, k), k)
        self.kind_combo.setToolTip("Test type; controls default axes, metadata fields and analysis tab.")
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        gf.addRow("Test type", self.kind_combo)
        rl.addWidget(gen_box)

        self.meta_stack = QStackedWidget()
        self.meta_stack.addWidget(self._build_generic_page())
        self.meta_stack.addWidget(self._build_mech_page())
        self.meta_stack.addWidget(self._build_dsc_page())
        rl.addWidget(self.meta_stack)

        preset_box = QGroupBox("Presets (saved column mappings)")
        pl = QVBoxLayout(preset_box)
        prow = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("Re-use a mapping saved for a given instrument/export format.")
        self.preset_combo.setMinimumContentsLength(8)
        self.preset_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.apply_preset_btn = QPushButton("Apply")
        self.apply_preset_btn.setToolTip("Apply the selected preset's mapping, test type and analysis defaults.")
        self.apply_preset_btn.clicked.connect(lambda: self.apply_preset(self.preset_combo.currentText()))
        self._refresh_presets()
        prow.addWidget(self.preset_combo, 1)
        prow.addWidget(self.apply_preset_btn)
        pl.addLayout(prow)
        self.save_preset_btn = QPushButton("Save current mapping as preset…")
        self.save_preset_btn.setToolTip("Store column names, quantities, units and the test type under a name.")
        self.save_preset_btn.clicked.connect(self._ask_save_preset)
        pl.addWidget(self.save_preset_btn)
        self.save_defaults_check = QCheckBox("Save analysis defaults with preset")
        self.save_defaults_check.setToolTip(
            "Also store the current analysis parameters (DSC prominence/baseline, tensile modulus range, "
            "cyclic settings …) so files imported with this preset start with them.")
        pl.addWidget(self.save_defaults_check)
        self.preset_defaults_label = QLabel("")
        self.preset_defaults_label.setWordWrap(True)
        pl.addWidget(self.preset_defaults_label)
        rl.addWidget(preset_box)
        rl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(right)
        scroll.setMinimumWidth(240)
        main.addWidget(scroll)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 1)
        main.setSizes([760, 320])

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status_label.setStyleSheet(f"color: {AMBER_TEXT};")
        root.addWidget(self.status_label)

        bottom = QHBoxLayout()
        self.apply_all_check = QCheckBox("Apply to all remaining files")
        self.apply_all_check.setToolTip("Import the remaining selected files with this mapping, test type "
                                        "and metadata without showing the dialog again.")
        self.apply_all_check.setVisible(show_apply_all)
        bottom.addWidget(self.apply_all_check)
        bottom.addStretch(1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Import")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        bottom.addWidget(self.buttons)
        root.addLayout(bottom)
        self.sheet_combo.currentTextChanged.connect(self._on_sheet_changed)
        self._update_defaults_check()

    def _build_generic_page(self) -> QWidget:
        w = QGroupBox("Metadata")
        lay = QVBoxLayout(w)
        lab = QLabel("No specimen metadata needed for generic data.")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        return w

    def _build_mech_page(self) -> QWidget:
        w = QGroupBox("Specimen geometry")
        f = QFormLayout(w)
        self.section_mode = QComboBox()
        self.section_mode.addItems(["Area (direct)", "Round (diameter Ø)", "Rectangular (width × thickness)"])
        self.section_mode.setToolTip("How to specify the initial cross-section A₀ used for engineering stress.")
        self.section_mode.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.section_mode.setMinimumContentsLength(10)
        f.addRow("Cross-section", self.section_mode)

        self.section_stack = QStackedWidget()
        # area page
        self.area_spin = _spin(tip="Initial cross-sectional area A₀.")
        self.area_unit = _unit_combo("area", "mm²")
        self.section_stack.addWidget(self._form_page([("Area A₀", _with_unit(self.area_spin, self.area_unit))]))
        # round page
        self.diam_spin = _spin(tip="Wire/rod diameter; A₀ = π·Ø²/4.")
        self.diam_unit = _unit_combo("length", "mm")
        self.section_stack.addWidget(self._form_page([("Diameter Ø", _with_unit(self.diam_spin, self.diam_unit))]))
        # rect page
        self.width_spin = _spin(tip="Specimen width w; A₀ = w·t.")
        self.width_unit = _unit_combo("length", "mm")
        self.thick_spin = _spin(tip="Specimen thickness t; A₀ = w·t.")
        self.thick_unit = _unit_combo("length", "mm")
        self.section_stack.addWidget(self._form_page([
            ("Width w", _with_unit(self.width_spin, self.width_unit)),
            ("Thickness t", _with_unit(self.thick_spin, self.thick_unit))]))
        f.addRow(self.section_stack)
        self.area_label = QLabel("A₀ = —")
        f.addRow("", self.area_label)

        self.gauge_spin = _spin(tip="Initial gauge length L₀ (extensometer or grip-to-grip) for strain.")
        self.gauge_unit = _unit_combo("length", "mm")
        f.addRow("Gauge length L₀", _with_unit(self.gauge_spin, self.gauge_unit))

        self.section_mode.currentIndexChanged.connect(self.section_stack.setCurrentIndex)
        self.section_mode.currentIndexChanged.connect(self._update_area_label)
        for sb in (self.area_spin, self.diam_spin, self.width_spin, self.thick_spin):
            sb.valueChanged.connect(self._update_area_label)
        for cb in (self.area_unit, self.diam_unit, self.width_unit, self.thick_unit):
            cb.currentIndexChanged.connect(self._update_area_label)
        return w

    def _build_dsc_page(self) -> QWidget:
        w = QGroupBox("DSC sample")
        f = QFormLayout(w)
        self.mass_spin = _spin(decimals=4, tip="Sample mass, needed to normalise heat flow (mW → W/g).")
        self.mass_unit = _unit_combo("mass", "mg")
        f.addRow("Sample mass", _with_unit(self.mass_spin, self.mass_unit))
        self.rate_spin = _spin(maximum=1e4, decimals=3,
                               tip="Heating/cooling rate β; converts temperature axis to time for enthalpy. "
                                   "ΔH scales with 1/β, so a wrong rate gives a proportionally wrong ΔH.")
        self.rate_unit = _unit_combo("heating_rate", "K/min")
        f.addRow("Heating rate", _with_unit(self.rate_spin, self.rate_unit))
        self.rate_confirm_check = QCheckBox("Heating rate checked")
        self.rate_confirm_check.setToolTip("The file does not state the heating rate. Tick after checking the "
                                           "value (or type the programme's rate), otherwise the default is "
                                           "flagged as a warning on the dataset.")
        f.addRow("", self.rate_confirm_check)
        self.dsc_note = QLabel("")
        self.dsc_note.setWordWrap(True)
        f.addRow(self.dsc_note)
        self.rate_spin.valueChanged.connect(self._on_rate_edited)
        self.rate_unit.currentIndexChanged.connect(self._on_rate_edited)
        self.rate_confirm_check.toggled.connect(self._update_rate_highlight)
        return w

    @staticmethod
    def _form_page(rows: list[tuple[str, QWidget]]) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(0, 0, 0, 0)
        for label, widget in rows:
            f.addRow(label, widget)
        return w

    # ------------------------------------------------------------------ (re)load raw table
    def _load_raw(self) -> None:
        """Fill every raw-dependent widget (called on construction and on sheet change)."""
        r = self.raw
        delim_name = {"\t": "tab", " ": "whitespace", ",": "comma", ";": "semicolon", "|": "pipe"}.get(
            r.delimiter, repr(r.delimiter))
        src = f"sheet '{r.sheet}'" if r.sheet else (
            f"delimiter: {delim_name} · decimal: '{r.decimal}' · encoding: {r.encoding}")
        self.info_label.setText(
            f"<b>{Path(self.path).name}</b> — {len(r.df)} rows × {len(r.headers)} columns · {src} · "
            f"header line: {'—' if r.header_row is None else r.header_row + 1}"
            f"{'' if r.units_row is None else ' (+ units row)'}")
        self.sheet_combo.blockSignals(True)
        self.sheet_combo.clear()
        self.sheet_combo.addItems(r.sheets)
        if r.sheet:
            self.sheet_combo.setCurrentText(r.sheet)
        self.sheet_combo.blockSignals(False)
        multi = len(r.sheets) > 1
        self.sheet_combo.setVisible(multi)
        self.sheet_label.setVisible(multi)
        self.preamble_view.setPlainText("\n".join(r.meta_lines) or "(no preamble)")
        self.pre_box.setChecked(bool(r.meta_lines))
        self.preamble_view.setVisible(self.pre_box.isChecked())
        self.prev_box.setTitle(f"Preview (first {min(PREVIEW_ROWS, len(r.df))} rows, as read from file)")
        self._fill_preview()
        specs = D.guess_mapping(r)
        self._populate_mapping(specs)
        self.kind_combo.blockSignals(True)
        self.kind_combo.setCurrentIndex(list(DATASET_KINDS).index(D.guess_kind(r, specs)))
        self.kind_combo.blockSignals(False)
        self._prefill_metadata()
        self._on_kind_changed()
        self._update_area_label()

    def set_sheet(self, sheet: str | int) -> None:
        """Re-read the file from another spreadsheet sheet and refresh the whole dialog."""
        self.raw = D.read_table(self.path, sheet=sheet)
        self._load_raw()

    def _on_sheet_changed(self, name: str) -> None:
        if not name or name == self.raw.sheet:
            return
        try:
            self.set_sheet(name)
        except Exception as exc:  # keep the previous sheet, tell the user (no modal: tests stay headless)
            self.sheet_combo.blockSignals(True)
            self.sheet_combo.setCurrentText(self.raw.sheet or "")
            self.sheet_combo.blockSignals(False)
            self.status_label.setText(f"Sheet '{name}': {exc}")

    @property
    def sheet(self) -> str | None:
        return self.raw.sheet

    def _fill_preview(self) -> None:
        r = self.raw
        n = min(PREVIEW_ROWS, len(r.df))
        t = self.preview_table
        t.clear()
        t.setRowCount(n)
        t.setColumnCount(len(r.headers))
        labels = []
        for i, h in enumerate(r.headers):
            u = r.units[i] if i < len(r.units) else None
            labels.append(f"{h}\n{u}" if u else h)
        t.setHorizontalHeaderLabels(labels)
        arr = r.df.iloc[:n].to_numpy(dtype=float) if n else np.empty((0, len(r.headers)))
        for i in range(n):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                t.setItem(i, j, QTableWidgetItem("" if not np.isfinite(v) else f"{v:.6g}"))

    # ------------------------------------------------------------------ mapping
    def _populate_mapping(self, specs: list[D.ColumnSpec]) -> None:
        self._populating = True
        self.map_table.setRowCount(0)
        self.map_table.setRowCount(len(specs))
        for lst in (self.include_checks, self.name_edits, self.dim_combos, self.unit_combos,
                    self.confirm_checks, self._guessed, self._reasons):
            lst.clear()
        self._sources = [s.source for s in specs]
        dims = U.dimensions()
        for row, s in enumerate(specs):
            chk = QCheckBox()
            chk.setChecked(s.include)
            chk.setToolTip("Import this column.")
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(6, 0, 0, 0)
            hl.addWidget(chk)
            self.map_table.setCellWidget(row, self.COL_INCLUDE, holder)
            src = QTableWidgetItem(s.source)
            src.setFlags(src.flags() & ~Qt.ItemIsEditable)
            self.map_table.setItem(row, self.COL_SOURCE, src)
            ed = QLineEdit(s.name)
            ed.setToolTip("Column name inside LabStudio.")
            self.map_table.setCellWidget(row, self.COL_NAME, ed)
            dc = QComboBox()
            for d in dims:
                dc.addItem(U.DIMENSION_LABELS.get(d, d), d)
            dc.setCurrentIndex(dims.index(s.dimension) if s.dimension in dims else dims.index("dimensionless"))
            dc.setToolTip("Physical quantity; values are converted to its canonical unit "
                          "(e.g. force → N, temperature → °C).")
            self.map_table.setCellWidget(row, self.COL_DIM, dc)
            uc = QComboBox()
            uc.setToolTip("Unit the values are written in, in the file.")
            self._fill_units(uc, dc.currentData(), s.unit)
            self.map_table.setCellWidget(row, self.COL_UNIT, uc)
            cf = QCheckBox()
            cf.setChecked(not s.guessed)
            cholder = QWidget()
            cl = QHBoxLayout(cholder)
            cl.setContentsMargins(6, 0, 0, 0)
            cl.addWidget(cf)
            self.map_table.setCellWidget(row, self.COL_CONFIRM, cholder)
            prev = QTableWidgetItem("")
            prev.setFlags(prev.flags() & ~Qt.ItemIsEditable)
            self.map_table.setItem(row, self.COL_PREVIEW, prev)
            self.include_checks.append(chk); self.name_edits.append(ed)
            self.dim_combos.append(dc); self.unit_combos.append(uc)
            self.confirm_checks.append(cf)
            self._guessed.append(bool(s.guessed)); self._reasons.append(s.reason or "")
            dc.currentIndexChanged.connect(lambda _i, r=row: self._on_dim_changed(r))
            uc.currentIndexChanged.connect(lambda _i, r=row: self._on_unit_changed(r))
            cf.toggled.connect(lambda _c, r=row: self._update_row_highlight(r))
            chk.toggled.connect(lambda _c: self._update_status())
            self._update_preview_cell(row)
            self._update_row_highlight(row)
        self._populating = False
        self._update_status()

    @staticmethod
    def _fill_units(combo: QComboBox, dim: str, unit: str | None) -> None:
        combo.blockSignals(True)
        combo.clear()
        units = U.units_for(dim)
        combo.addItems(units)
        combo.setCurrentText(unit if unit in units else U.CANONICAL[dim])
        combo.blockSignals(False)

    def _on_dim_changed(self, row: int) -> None:
        dim = self.dim_combos[row].currentData()
        self._fill_units(self.unit_combos[row], dim, self.unit_combos[row].currentText())
        self._update_preview_cell(row)
        if not self._populating:
            self.confirm_checks[row].setChecked(True)      # the user chose the quantity explicitly

    def _on_unit_changed(self, row: int) -> None:
        self._update_preview_cell(row)
        if not self._populating:
            self.confirm_checks[row].setChecked(True)

    def _update_row_highlight(self, row: int) -> None:
        pending = self._guessed[row] and not self.confirm_checks[row].isChecked()
        reason = self._reasons[row]
        item = self.map_table.item(row, self.COL_SOURCE)
        if item is not None:
            item.setBackground(QBrush(QColor(AMBER)) if pending else QBrush())
            item.setToolTip(f"Guessed: {reason}" if self._guessed[row] and reason else "")
        style = f"background-color: {AMBER};" if pending else ""
        holder = self.map_table.cellWidget(row, self.COL_CONFIRM)
        if holder is not None:
            holder.setStyleSheet(style)
            holder.setToolTip((f"Guessed: {reason}\n" if self._guessed[row] and reason else "")
                              + "Tick when the quantity and unit are checked.")
        self.unit_combos[row].setToolTip(("Guessed — " + reason + "\n" if pending and reason else "")
                                         + "Unit the values are written in, in the file.")
        if not self._populating:
            self._update_status()

    def _update_preview_cell(self, row: int) -> None:
        item = self.map_table.item(row, self.COL_PREVIEW)
        if item is None:
            return
        col = self.raw.df[self._sources[row]]
        finite = col[np.isfinite(col.to_numpy(float))]
        if finite.empty:
            item.setText("(no numeric data)")
            return
        v = float(finite.iloc[0])
        dim = self.dim_combos[row].currentData()
        unit = self.unit_combos[row].currentText()
        try:
            c = float(U.to_canonical(v, unit, dim))
            cu = U.CANONICAL[dim]
            item.setText(f"{v:.6g} {unit} → {c:.6g} {cu}".strip())
        except ValueError as exc:
            item.setText(str(exc))

    def confirm_all(self) -> None:
        """Mark every column's quantity/unit as checked by the user."""
        for cf in self.confirm_checks:
            cf.setChecked(True)

    def unconfirmed(self) -> list[str]:
        """Source names of included columns whose guessed unit is not confirmed yet."""
        return [s.source for s in self.specs() if s.include and s.guessed]

    def specs(self) -> list[D.ColumnSpec]:
        """Current column mapping from the widgets (``guessed`` = guess not confirmed yet)."""
        out = []
        for i, src in enumerate(self._sources):
            pending = self._guessed[i] and not self.confirm_checks[i].isChecked()
            out.append(D.ColumnSpec(source=src, name=self.name_edits[i].text().strip() or src,
                                    dimension=self.dim_combos[i].currentData(),
                                    unit=self.unit_combos[i].currentText(),
                                    include=self.include_checks[i].isChecked(),
                                    guessed=pending, reason=self._reasons[i] if pending else ""))
        return out

    def set_specs(self, specs: list[D.ColumnSpec]) -> None:
        """Set the mapping widgets (specs matched to rows by source name)."""
        by_src = {s.source: s for s in specs}
        dims = U.dimensions()
        self._populating = True
        try:
            for i, src in enumerate(self._sources):
                s = by_src.get(src)
                if s is None:
                    continue
                self.include_checks[i].setChecked(s.include)
                self.name_edits[i].setText(s.name)
                if s.dimension in dims:
                    self.dim_combos[i].setCurrentIndex(dims.index(s.dimension))
                self._fill_units(self.unit_combos[i], self.dim_combos[i].currentData(), s.unit)
                self._update_preview_cell(i)
                self._guessed[i] = bool(getattr(s, "guessed", False))
                self._reasons[i] = getattr(s, "reason", "") or ""
                self.confirm_checks[i].setChecked(not self._guessed[i])
                self._update_row_highlight(i)
        finally:
            self._populating = False
        self._update_status()

    # ------------------------------------------------------------------ kind / metadata
    def kind(self) -> str:
        return self.kind_combo.currentData()

    def set_kind(self, kind: str) -> None:
        self.kind_combo.setCurrentIndex(list(DATASET_KINDS).index(kind))

    def _on_kind_changed(self, *_: Any) -> None:
        k = self.kind()
        self.meta_stack.setCurrentIndex(1 if k in D.MECHANICAL_KINDS else 2 if k == "dsc" else 0)
        self._update_rate_highlight()

    @property
    def rate_from_file(self) -> bool:
        """True if the file states the heating rate."""
        return bool(self.raw.meta.get("heating_rate_K_per_min"))

    @property
    def rate_needs_confirmation(self) -> bool:
        """DSC rate not in the file and not confirmed/edited by the user."""
        return self.kind() == "dsc" and not self.rate_from_file and not self.rate_confirm_check.isChecked()

    def _on_rate_edited(self, *_: Any) -> None:
        if not self._populating and not self.rate_from_file:
            self.rate_confirm_check.setChecked(True)
        self._update_rate_highlight()

    def _update_rate_highlight(self, *_: Any) -> None:
        need = self.rate_needs_confirmation
        self.rate_spin.setStyleSheet(f"QDoubleSpinBox {{ background-color: {AMBER}; }}" if need else "")
        self.rate_confirm_check.setVisible(not self.rate_from_file)
        notes = []
        if self.kind() == "dsc":
            if any(s.include and s.dimension == "heat_flow" for s in self.specs()):
                notes.append("Heat flow is in mW: enter the sample mass to normalise to W/g.")
            if not self.rate_from_file and need:
                notes.append(f"Heating rate not found in the file — the default "
                             f"{D.DEFAULT_HEATING_RATE_K_PER_MIN:g} K/min will be used and flagged as a "
                             f"warning. Enter the programme's rate or tick “Heating rate checked”.")
            elif not self.rate_from_file:
                notes.append("Heating rate not found in the file — the value entered here is used.")
            elif self.raw.meta.get("heating_rate_text"):
                notes.append(f"Heating rate read from: “{self.raw.meta['heating_rate_text']}”.")
        self.dsc_note.setText(" ".join(notes))
        self.dsc_note.setStyleSheet(f"color: {AMBER_TEXT};" if need else "")
        self._update_status()

    def _prefill_metadata(self) -> None:
        m = self.raw.meta
        self._populating = True
        try:
            if m.get("sample_name"):
                self.name_edit.setToolTip(f"Sample in file: {m['sample_name']}")
            self.mass_spin.setValue(float(m["mass_mg"]) if m.get("mass_mg") else 0.0)
            self.rate_unit.setCurrentText("K/min")
            self.rate_confirm_check.setChecked(False)
            if m.get("heating_rate_K_per_min"):
                self.rate_spin.setValue(float(m["heating_rate_K_per_min"]))
            else:
                self.rate_spin.setValue(D.DEFAULT_HEATING_RATE_K_PER_MIN)
            if m.get("gauge_length_mm"):
                self.gauge_spin.setValue(float(m["gauge_length_mm"]))
            if m.get("diameter_mm"):
                self.diam_spin.setValue(float(m["diameter_mm"]))
                self.section_mode.setCurrentIndex(SECTION_ROUND)
            elif m.get("width_mm") and m.get("thickness_mm"):
                self.width_spin.setValue(float(m["width_mm"]))
                self.thick_spin.setValue(float(m["thickness_mm"]))
                self.section_mode.setCurrentIndex(SECTION_RECT)
            elif m.get("area_mm2"):
                self.area_spin.setValue(float(m["area_mm2"]))
                self.section_mode.setCurrentIndex(SECTION_AREA)
        finally:
            self._populating = False

    @staticmethod
    def _mm(spin: QDoubleSpinBox, combo: QComboBox) -> float | None:
        v = spin.value()
        return float(U.convert(v, combo.currentText(), "mm", "length")) if v > 0 else None

    def section_values(self) -> dict[str, float]:
        """Geometry in canonical units from the mechanical form (only keys that are set)."""
        out: dict[str, float] = {}
        mode = self.section_mode.currentIndex()
        if mode == SECTION_AREA:
            if self.area_spin.value() > 0:
                out["area_mm2"] = float(U.convert(self.area_spin.value(), self.area_unit.currentText(),
                                                  "mm²", "area"))
        elif mode == SECTION_ROUND:
            d = self._mm(self.diam_spin, self.diam_unit)
            if d:
                out["diameter_mm"] = d
                out["area_mm2"] = math.pi * d * d / 4.0
        else:
            w = self._mm(self.width_spin, self.width_unit)
            t = self._mm(self.thick_spin, self.thick_unit)
            if w:
                out["width_mm"] = w
            if t:
                out["thickness_mm"] = t
            if w and t:
                out["area_mm2"] = w * t
        g = self._mm(self.gauge_spin, self.gauge_unit)
        if g:
            out["gauge_length_mm"] = g
        return out

    def _update_area_label(self, *_: Any) -> None:
        a = self.section_values().get("area_mm2")
        self.area_label.setText(f"A₀ = {a:.4g} mm²" if a else "A₀ = — (not set)")

    def metadata(self) -> dict[str, Any]:
        """Metadata (canonical units) for the current test type.

        For DSC the heating rate is only included when it came from the file or the user
        confirmed/edited it (``heating_rate_source`` 'header' or 'user'); otherwise
        :func:`data_io.build_dataset` applies the 10 K/min default with a warning.
        """
        k = self.kind()
        meta: dict[str, Any] = {}
        if self.raw.meta.get("sample_name"):
            meta["sample_name"] = self.raw.meta["sample_name"]
        if k in D.MECHANICAL_KINDS:
            meta.update(self.section_values())
        elif k == "dsc":
            if self.mass_spin.value() > 0:
                meta["mass_mg"] = float(U.convert(self.mass_spin.value(), self.mass_unit.currentText(),
                                                  "mg", "mass"))
            if self.rate_spin.value() > 0 and not self.rate_needs_confirmation:
                rate = float(U.convert(self.rate_spin.value(), self.rate_unit.currentText(), "K/min",
                                       "heating_rate"))
                file_rate = self.raw.meta.get("heating_rate_K_per_min")
                meta["heating_rate_K_per_min"] = rate
                meta["heating_rate_source"] = ("header" if file_rate and
                                               math.isclose(rate, float(file_rate), rel_tol=1e-6) else "user")
        if self._analysis_defaults:
            meta["analysis_defaults"] = dict(self._analysis_defaults)
        return meta

    # ------------------------------------------------------------------ warnings preview
    def pending_warnings(self) -> list[str]:
        """Warnings the dataset would carry if imported now (unconfirmed guesses, defaulted rate)."""
        out = D.import_warnings(self.specs())
        if self.rate_needs_confirmation:
            out.append(f"heating rate not found in file — defaulted to {D.DEFAULT_HEATING_RATE_K_PER_MIN:g} K/min")
        return out

    def _update_status(self) -> None:
        if self._populating or not hasattr(self, "status_label"):
            return
        w = self.pending_warnings()
        if not w:
            self.status_label.setText("")
            return
        n_cols = len(self.unconfirmed())
        head = []
        if n_cols:
            head.append(f"{n_cols} column unit(s) guessed and not confirmed (amber)")
        if self.rate_needs_confirmation:
            head.append("heating rate defaulted")
        self.status_label.setText("⚠ " + "; ".join(head) + ". You can import anyway — these will be "
                                  "recorded as warnings on the dataset.")
        self.status_label.setToolTip("\n".join(w))

    # ------------------------------------------------------------------ presets
    def _refresh_presets(self, select: str | None = None) -> None:
        self.preset_combo.clear()
        names = D.list_presets(self.presets_dir)
        self.preset_combo.addItems(names)
        self.apply_preset_btn.setEnabled(bool(names))
        if select and select in names:
            self.preset_combo.setCurrentText(select)

    def set_analysis_defaults_provider(self, provider: Callable[[], dict] | None) -> None:
        """Callable returning the current analysis parameters (dict per analysis), used by
        "Save analysis defaults with preset"."""
        self._analysis_defaults_provider = provider
        self._update_defaults_check()

    @classmethod
    def set_default_analysis_defaults_provider(cls, provider: Callable[[], dict] | None) -> None:
        """Application-wide provider used by every dialog without its own provider."""
        cls._default_analysis_defaults_provider = provider

    def _provider(self) -> Callable[[], dict] | None:
        return self._analysis_defaults_provider or type(self)._default_analysis_defaults_provider

    def _update_defaults_check(self) -> None:
        has = self._provider() is not None
        self.save_defaults_check.setEnabled(has)
        if not has:
            self.save_defaults_check.setChecked(False)

    def analysis_defaults(self) -> dict[str, Any]:
        """Analysis defaults that will be attached to the dataset (from an applied preset)."""
        return dict(self._analysis_defaults)

    def apply_preset(self, name_or_preset: str | D.Preset) -> None:
        """Apply a preset (by name or object) to the mapping, test type, metadata and analysis defaults."""
        if not name_or_preset:
            return
        p = name_or_preset if isinstance(name_or_preset, D.Preset) else \
            D.load_preset(name_or_preset, directory=self.presets_dir)
        self.set_specs(D.apply_preset(self.raw, p))
        if p.kind in DATASET_KINDS:
            self.set_kind(p.kind)
        md = p.metadata or {}
        if md.get("gauge_length_mm") and not self.raw.meta.get("gauge_length_mm"):
            self.gauge_unit.setCurrentText("mm"); self.gauge_spin.setValue(md["gauge_length_mm"])
        if md.get("heating_rate_K_per_min") and not self.rate_from_file:
            self.rate_unit.setCurrentText("K/min"); self.rate_spin.setValue(md["heating_rate_K_per_min"])
        self._analysis_defaults = dict(p.analysis_defaults or {})
        self.preset_defaults_label.setText(
            f"Preset analysis defaults: {', '.join(sorted(self._analysis_defaults))}"
            if self._analysis_defaults else "")

    def save_preset_as(self, name: str, include_metadata: bool = False,
                       include_analysis_defaults: bool | None = None) -> Path:
        """Save the current mapping + test type as a preset called ``name``.

        ``include_analysis_defaults`` (default: the "Save analysis defaults" checkbox) stores the
        dict returned by the analysis-defaults provider.
        """
        md = {}
        if include_metadata:
            md = {k: v for k, v in self.metadata().items() if k in ("gauge_length_mm", "heating_rate_K_per_min")}
        want = self.save_defaults_check.isChecked() if include_analysis_defaults is None \
            else include_analysis_defaults
        ad = None
        prov = self._provider()
        if want and prov is not None:
            ad = dict(prov() or {})
        elif want and self._analysis_defaults:
            ad = dict(self._analysis_defaults)
        path = D.save_preset(name, self.specs(), self.kind(), md, directory=self.presets_dir,
                             analysis_defaults=ad)
        self._refresh_presets(select=name)
        return path

    def _ask_save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name (e.g. 'Instron 5967 tensile'):")
        if ok and name.strip():
            self.save_preset_as(name.strip())

    # ------------------------------------------------------------------ result
    def build(self) -> Dataset:
        """Build the Dataset from the current widget state (no dialog interaction needed)."""
        return D.build_dataset(self.raw, self.specs(), name=self.name_edit.text().strip() or None,
                               kind=self.kind(), metadata=self.metadata())

    def accept(self) -> None:  # noqa: D401 - Qt override
        try:
            self._dataset = self.build()
        except Exception as exc:  # show the problem, keep dialog open
            self._dataset = None
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        super().accept()

    def dataset(self) -> Dataset | None:
        """Dataset built on accept (None if cancelled / not yet accepted)."""
        return self._dataset

    # ------------------------------------------------------------------ batch helper
    @staticmethod
    def import_files(paths: list[str], parent: QWidget | None = None, *,
                     presets_dir: str | os.PathLike | None = None,
                     analysis_defaults_provider: Callable[[], dict] | None = None) -> list[Dataset]:
        """Open one dialog per file; "Apply to all remaining" imports the rest with the same settings.

        With "apply to all", each remaining file keeps its own header metadata (geometry, mass,
        heating rate); a heating rate the user typed is re-used, a header rate is not.
        """
        out: list[Dataset] = []
        template: tuple[list[D.ColumnSpec], str, dict, str | None] | None = None
        paths = list(paths)
        for i, p in enumerate(paths):
            try:
                if template is not None:
                    specs, kind, meta, sheet = template
                    use_sheet = sheet if sheet and sheet in D.list_sheets(p) else None
                    raw = D.read_table(p, sheet=use_sheet)
                    file_meta = D.metadata_from_raw(raw, kind)
                    md = {**meta, **file_meta}
                    if "heating_rate_K_per_min" in file_meta:
                        md.pop("heating_rate_source", None)
                    out.append(D.build_dataset(raw, D.apply_preset(raw, specs), name=Path(p).stem,
                                               kind=kind, metadata=md))
                    continue
                dlg = ImportDialog(p, parent, presets_dir=presets_dir, show_apply_all=i < len(paths) - 1)
                if analysis_defaults_provider is not None:
                    dlg.set_analysis_defaults_provider(analysis_defaults_provider)
            except Exception as exc:
                QMessageBox.warning(parent, "Import failed", f"{Path(p).name}:\n{exc}")
                continue
            if dlg.exec() == QDialog.Accepted and dlg.dataset() is not None:
                out.append(dlg.dataset())
                if dlg.apply_all_check.isChecked():
                    meta = {k: v for k, v in dlg.metadata().items() if k != "sample_name"}
                    if meta.get("heating_rate_source") == "header":
                        # file 1's header rate must not be applied to file 2
                        meta.pop("heating_rate_K_per_min", None)
                        meta.pop("heating_rate_source", None)
                    template = (dlg.specs(), dlg.kind(), meta, dlg.sheet)
            else:
                # cancelling one file skips it; keep going with the others
                continue
        return out
