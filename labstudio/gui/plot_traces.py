"""Left panel of the Plot tab (module B): datasets, traces, trace style, display units.

* Datasets: list with inline rename (→ ``store.rename``), remove (→ ``store.remove``,
  its traces go too), reorder by drag or ▲/▼ (→ ``store.move``), duplicate, context menu.
* Traces: add from dataset X/Y, visibility, multi-select (for "Average selected"),
  style (colour, line, marker, label, panel, secondary axis, error band/bars).
* Display units of the active panel's axes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QPushButton, QSizePolicy, QSpinBox, QToolButton,
    QVBoxLayout, QWidget,
)

from labstudio.core import units as U

from .plot_style import LINESTYLES, MARKERS

if TYPE_CHECKING:  # pragma: no cover
    from .plot_widget import PlotPanel

ERROR_MODES = {"None": "none", "Shaded ±SD band": "band", "Error bars": "bars"}


def _swatch(color: str, ls: str = "-") -> QIcon:
    pm = QPixmap(22, 12)
    pm.fill(Qt.transparent)
    from PySide6.QtGui import QPainter, QPen
    p = QPainter(pm)
    pen = QPen(QColor(color))
    pen.setWidth(3)
    pen.setStyle({"--": Qt.DashLine, "-.": Qt.DashDotLine, ":": Qt.DotLine}.get(ls, Qt.SolidLine))
    p.setPen(pen)
    p.drawLine(1, 6, 21, 6)
    p.end()
    return QIcon(pm)


def _small_combo(cb: QComboBox, n: int = 8) -> QComboBox:
    cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    cb.setMinimumContentsLength(n)
    cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return cb


class TraceManager(QWidget):
    """Left panel: datasets, add traces, list them (visibility), edit style, display units."""

    def __init__(self, panel: "PlotPanel", parent=None):
        super().__init__(parent)
        self.panel = panel
        self._sync = False
        self._ds_sync = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # ---- datasets
        g_ds = QGroupBox("Datasets")
        g_ds.setCheckable(True)
        g_ds.setChecked(True)
        g_ds.setToolTip("Untick to collapse. Double-click a name to rename; drag to reorder; right-click for "
                        "more.")
        vd = QVBoxLayout(g_ds)
        self.ds_list = QListWidget()
        self.ds_list.setToolTip("Datasets in this session. Double-click to rename, drag to reorder, right-click "
                                "for Duplicate / Remove / Plot.")
        self.ds_list.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.ds_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.ds_list.setDefaultDropAction(Qt.MoveAction)
        self.ds_list.setMinimumHeight(70)
        self.ds_list.setMaximumHeight(150)
        self.ds_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ds_list.setTextElideMode(Qt.ElideMiddle)
        self.ds_list.setContextMenuPolicy(Qt.CustomContextMenu)
        vd.addWidget(self.ds_list)
        hd = QGridLayout()
        self.btn_ds_up = QToolButton(text="▲")
        self.btn_ds_up.setToolTip("Move dataset up.")
        self.btn_ds_down = QToolButton(text="▼")
        self.btn_ds_down.setToolTip("Move dataset down.")
        self.btn_ds_ren = QToolButton(text="Rename")
        self.btn_ds_ren.setToolTip("Rename the dataset (also F2 / double-click).")
        self.btn_ds_dup = QToolButton(text="Duplicate")
        self.btn_ds_dup.setToolTip("Add an independent copy of the dataset (data and metadata).")
        self.btn_ds_del = QToolButton(text="Remove")
        self.btn_ds_del.setToolTip("Remove the dataset from the session (its traces are removed from the plot).")
        for b, r, c, cs in ((self.btn_ds_up, 0, 0, 1), (self.btn_ds_down, 0, 1, 1), (self.btn_ds_ren, 0, 2, 2),
                            (self.btn_ds_dup, 1, 0, 2), (self.btn_ds_del, 1, 2, 2)):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            hd.addWidget(b, r, c, 1, cs)
        vd.addLayout(hd)
        g_ds.toggled.connect(lambda on: [w.setVisible(on) for w in (self.ds_list, self.btn_ds_up, self.btn_ds_down,
                                                                    self.btn_ds_ren, self.btn_ds_dup,
                                                                    self.btn_ds_del)])
        lay.addWidget(g_ds)

        # ---- add
        g_add = QGroupBox("Add trace")
        f = QFormLayout(g_add)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.cb_ds = _small_combo(QComboBox(), 10)
        self.cb_ds.setToolTip("Dataset to plot from (imported files and derived curves).")
        self.cb_x = _small_combo(QComboBox(), 10)
        self.cb_x.setToolTip("Column for the horizontal axis — shown as name (canonical unit).")
        self.cb_y = _small_combo(QComboBox(), 10)
        self.cb_y.setToolTip("Column for the vertical axis.")
        self.chk_sec = QCheckBox("Secondary Y axis")
        self.chk_sec.setToolTip("Plot against a second y-axis on the right (e.g. temperature vs time "
                                "together with heat flow).")
        self.btn_add = QPushButton("Add trace")
        self.btn_add.setToolTip("Add the chosen X/Y pair to the active panel.")
        f.addRow("Dataset", self.cb_ds)
        f.addRow("X", self.cb_x)
        f.addRow("Y", self.cb_y)
        f.addRow(self.chk_sec)
        f.addRow(self.btn_add)
        lay.addWidget(g_add)

        # ---- list
        g_list = QGroupBox("Traces")
        vl = QVBoxLayout(g_list)
        self.list = QListWidget()
        self.list.setToolTip("Tick to show/hide a trace; select it to edit its style and to use it with the "
                             "tools. Ctrl/Shift-click to select several (for 'Average selected').")
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.setMinimumHeight(90)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setTextElideMode(Qt.ElideRight)
        vl.addWidget(self.list)
        hb = QHBoxLayout()
        self.btn_up = QToolButton(text="▲")
        self.btn_up.setToolTip("Move trace up (drawn earlier, earlier in legend, lower in a waterfall).")
        self.btn_down = QToolButton(text="▼")
        self.btn_down.setToolTip("Move trace down.")
        self.btn_del = QToolButton(text="Remove")
        self.btn_del.setToolTip("Remove the selected trace from the plot (the dataset is kept).")
        self.btn_clear = QToolButton(text="Clear")
        self.btn_clear.setToolTip("Remove all traces and overlays.")
        for b in (self.btn_up, self.btn_down):
            hb.addWidget(b)
        hb.addStretch(1)
        hb.addWidget(self.btn_del)
        hb.addWidget(self.btn_clear)
        vl.addLayout(hb)
        ha = QHBoxLayout()
        self.btn_avg = QPushButton("Average selected")
        self.btn_avg.setToolTip("Mean ± SD of the selected traces on a common x grid (their overlap range): "
                                "creates a derived dataset (mean, SD, n) and plots it with a shaded ±SD band.")
        self.cb_avg_err = _small_combo(QComboBox(), 6)
        self.cb_avg_err.addItems(["band", "bars"])
        self.cb_avg_err.setToolTip("Show the SD of the average as a shaded band or as error bars.")
        ha.addWidget(self.btn_avg, 1)
        ha.addWidget(self.cb_avg_err)
        vl.addLayout(ha)
        lay.addWidget(g_list, 1)

        # ---- style
        self.g_style = QGroupBox("Selected trace style")
        fs = QFormLayout(self.g_style)
        fs.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.ed_label = QLineEdit()
        self.ed_label.setToolTip("Legend label. Supports matplotlib mathtext, e.g. $\\sigma_{SIM}$.")
        self.btn_color = QPushButton()
        self.btn_color.setToolTip("Pick the line colour.")
        self.cb_ls = _small_combo(QComboBox(), 6)
        self.cb_ls.addItems(list(LINESTYLES))
        self.cb_ls.setToolTip("Line style ('None' to show markers only).")
        self.sp_lw = QDoubleSpinBox(minimum=0.0, maximum=10.0, singleStep=0.25, decimals=2)
        self.sp_lw.setToolTip("Line width in points.")
        self.cb_marker = _small_combo(QComboBox(), 6)
        self.cb_marker.addItems(list(MARKERS))
        self.cb_marker.setToolTip("Marker drawn at each (displayed) data point.")
        self.sp_ms = QDoubleSpinBox(minimum=0.0, maximum=30.0, singleStep=0.5, decimals=1)
        self.sp_ms.setToolTip("Marker size in points.")
        self.sp_alpha = QDoubleSpinBox(minimum=0.05, maximum=1.0, singleStep=0.05, decimals=2)
        self.sp_alpha.setToolTip("Opacity (1 = opaque).")
        self.chk_trace_sec = QCheckBox("On secondary Y axis")
        self.chk_trace_sec.setToolTip("Move this trace to the right-hand y-axis.")
        self.cb_panel = _small_combo(QComboBox(), 6)
        self.cb_panel.setToolTip("Subplot panel this trace is drawn in (set the layout in Customise ▸ Layout).")
        self.cb_err = _small_combo(QComboBox(), 8)
        self.cb_err.addItems(list(ERROR_MODES))
        self.cb_err.setToolTip("Error display from the dataset's '<y> SD' column (e.g. an averaged trace): "
                               "shaded ±SD band or error bars every N points.")
        self.sp_err_every = QSpinBox(minimum=1, maximum=100000)
        self.sp_err_every.setToolTip("Draw an error bar every N data points.")
        self.sp_err_every.setPrefix("every ")
        row = QHBoxLayout()
        row.addWidget(self.btn_color)
        row.addWidget(QLabel("Alpha"))
        row.addWidget(self.sp_alpha)
        er = QHBoxLayout()
        er.addWidget(self.cb_err, 1)
        er.addWidget(self.sp_err_every)
        fs.addRow("Label", self.ed_label)
        fs.addRow("Colour", row)
        fs.addRow("Line", self.cb_ls)
        fs.addRow("Width", self.sp_lw)
        fs.addRow("Marker", self.cb_marker)
        fs.addRow("Marker size", self.sp_ms)
        fs.addRow("Panel", self.cb_panel)
        fs.addRow("Error", er)
        fs.addRow(self.chk_trace_sec)
        lay.addWidget(self.g_style)

        # ---- units
        g_units = QGroupBox("Display units (active panel)")
        g_units.setToolTip("Show an axis in another unit (e.g. stress in GPa, temperature in K). "
                           "Stored data stay in canonical units.")
        fu = QFormLayout(g_units)
        self.cb_unit: dict[str, QComboBox] = {}
        for key, lab in (("x", "X axis"), ("y", "Y axis"), ("y2", "Y2 axis")):
            cb = _small_combo(QComboBox(), 6)
            cb.setToolTip(f"Display unit for the {lab.lower()}.")
            self.cb_unit[key] = cb
            fu.addRow(lab, cb)
            cb.currentTextChanged.connect(lambda txt, k=key: self._unit_changed(k, txt))
        lay.addWidget(g_units)
        lay.addStretch(0)

        # ---- signals
        self.cb_ds.currentIndexChanged.connect(self._ds_changed)
        self.btn_add.clicked.connect(self._add_clicked)
        self.list.currentRowChanged.connect(self._row_changed)
        self.list.itemChanged.connect(self._item_changed)
        self.btn_del.clicked.connect(self._remove_clicked)
        self.btn_clear.clicked.connect(lambda: self.panel.clear())
        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_down.clicked.connect(lambda: self._move(1))
        self.btn_avg.clicked.connect(self._average_clicked)
        self.btn_color.clicked.connect(self._pick_color)
        self.ed_label.editingFinished.connect(lambda: self._style_edit(label=self.ed_label.text()))
        self.cb_ls.currentTextChanged.connect(lambda t: self._style_edit(linestyle=LINESTYLES[t]))
        self.sp_lw.valueChanged.connect(lambda v: self._style_edit(linewidth=v))
        self.cb_marker.currentTextChanged.connect(lambda t: self._style_edit(marker=MARKERS[t]))
        self.sp_ms.valueChanged.connect(lambda v: self._style_edit(markersize=v))
        self.sp_alpha.valueChanged.connect(lambda v: self._style_edit(alpha=v))
        self.chk_trace_sec.toggled.connect(self._sec_toggled)
        self.cb_panel.currentIndexChanged.connect(self._panel_changed)
        self.cb_err.currentTextChanged.connect(self._err_changed)
        self.sp_err_every.valueChanged.connect(self._err_changed)
        self.ds_list.itemChanged.connect(self._ds_item_changed)
        self.ds_list.currentRowChanged.connect(self._ds_row_changed)
        self.ds_list.model().rowsMoved.connect(lambda *_: QTimer.singleShot(0, self._ds_rows_moved))
        self.ds_list.customContextMenuRequested.connect(self._ds_menu)
        self.btn_ds_up.clicked.connect(lambda: self._ds_move(-1))
        self.btn_ds_down.clicked.connect(lambda: self._ds_move(1))
        self.btn_ds_ren.clicked.connect(self._ds_rename_clicked)
        self.btn_ds_dup.clicked.connect(self._ds_duplicate)
        self.btn_ds_del.clicked.connect(self._ds_remove)
        self.refresh_datasets()
        self.refresh_traces()

    # ------------------------------------------------------------------ datasets list
    def current_dataset(self):
        it = self.ds_list.currentItem()
        return self.panel.store.get(it.data(Qt.UserRole)) if it is not None else None

    def _refresh_ds_list(self) -> None:
        cur = self.cb_ds.currentData()
        self._ds_sync = True
        self.ds_list.clear()
        for ds in self.panel.store:
            it = QListWidgetItem(ds.name)
            it.setData(Qt.UserRole, ds.id)
            it.setFlags(it.flags() | Qt.ItemIsEditable | Qt.ItemIsDragEnabled)
            n = sum(1 for t in self.panel.traces if t.ds is ds)
            it.setToolTip(f"{ds.name}\nkind: {ds.kind} · {len(ds.data)} rows · {len(ds.channels)} columns"
                          + (f" · {n} trace(s)" if n else "") + (f"\n{ds.source_path}" if ds.source_path else ""))
            self.ds_list.addItem(it)
            if ds.id == cur:
                self.ds_list.setCurrentItem(it)
        self._ds_sync = False

    def _ds_row_changed(self, row: int) -> None:
        if self._ds_sync:
            return
        it = self.ds_list.item(row)
        if it is not None:
            i = self.cb_ds.findData(it.data(Qt.UserRole))
            if i >= 0:
                self.cb_ds.setCurrentIndex(i)

    def _ds_item_changed(self, it: QListWidgetItem) -> None:
        if self._ds_sync:
            return
        ds = self.panel.store.get(it.data(Qt.UserRole))
        name = it.text().strip()
        if ds is not None and name and name != ds.name:
            # defer: renaming rebuilds this list, which must not happen inside its own signal
            QTimer.singleShot(0, lambda d=ds, n=name: self.panel.rename_dataset(d, n))

    def _ds_rows_moved(self) -> None:
        order = [self.ds_list.item(i).data(Qt.UserRole) for i in range(self.ds_list.count())]
        store_order = [d.id for d in self.panel.store]
        if order == store_order:
            return
        for idx, ds_id in enumerate(order):
            ds = self.panel.store.get(ds_id)
            if ds is not None and self.panel.store.index_of(ds) != idx:
                self.panel.move_dataset(ds, idx)

    def _ds_move(self, d: int) -> None:
        ds = self.current_dataset()
        if ds is None:
            return
        i = self.panel.store.index_of(ds)
        self.panel.move_dataset(ds, max(0, i + d))

    def _ds_rename_clicked(self) -> None:
        it = self.ds_list.currentItem()
        if it is not None:
            self.ds_list.editItem(it)

    def _ds_duplicate(self) -> None:
        ds = self.current_dataset()
        if ds is not None:
            self.panel.duplicate_dataset(ds)

    def _ds_remove(self) -> None:
        ds = self.current_dataset()
        if ds is not None:
            self.panel.remove_dataset(ds)

    def _ds_menu(self, pos) -> None:
        ds = self.current_dataset()
        if ds is None:
            return
        m = QMenu(self)
        acts = [("Plot default X/Y", lambda: self.panel.add_trace(ds, ds.default_x or ds.columns()[0],
                                                                  ds.default_y or ds.columns()[-1])),
                ("Rename", self._ds_rename_clicked), ("Duplicate", self._ds_duplicate),
                ("Move up", lambda: self._ds_move(-1)), ("Move down", lambda: self._ds_move(1)),
                ("Remove", self._ds_remove)]
        for text, fn in acts:
            a = QAction(text, m)
            a.triggered.connect(fn)
            m.addAction(a)
        m.popup(self.ds_list.mapToGlobal(pos))

    # ------------------------------------------------------------------ datasets / columns
    def refresh_datasets(self) -> None:
        cur = self.cb_ds.currentData()
        self.cb_ds.blockSignals(True)
        self.cb_ds.clear()
        for ds in self.panel.store:
            self.cb_ds.addItem(ds.name, ds.id)
        i = self.cb_ds.findData(cur)
        self.cb_ds.setCurrentIndex(i if i >= 0 else self.cb_ds.count() - 1)
        self.cb_ds.blockSignals(False)
        self._ds_changed()
        self._refresh_ds_list()

    def _ds_changed(self, *_):
        ds = self.panel.store.get(self.cb_ds.currentData()) if self.cb_ds.count() else None
        cx, cy = self.cb_x.currentData(), self.cb_y.currentData()
        for cb in (self.cb_x, self.cb_y):
            cb.clear()
        if ds is None:
            self.btn_add.setEnabled(False)
            return
        self.btn_add.setEnabled(True)
        for col in ds.columns():
            self.cb_x.addItem(ds.label(col), col)
            self.cb_y.addItem(ds.label(col), col)
        for cb, keep, default in ((self.cb_x, cx, ds.default_x), (self.cb_y, cy, ds.default_y)):
            i = cb.findData(keep)
            if i < 0:
                i = cb.findData(default)
            if i >= 0:
                cb.setCurrentIndex(i)
        if not ds.default_y and self.cb_y.count() > 1 and self.cb_y.currentIndex() == self.cb_x.currentIndex():
            self.cb_y.setCurrentIndex(1)
        if not self._ds_sync:
            for i in range(self.ds_list.count()):
                if self.ds_list.item(i).data(Qt.UserRole) == ds.id:
                    self._ds_sync = True
                    self.ds_list.setCurrentRow(i)
                    self._ds_sync = False

    def _add_clicked(self):
        ds = self.panel.store.get(self.cb_ds.currentData())
        if ds is None or self.cb_x.currentData() is None:
            return
        self.panel.add_trace(ds, self.cb_x.currentData(), self.cb_y.currentData(),
                             secondary_y=self.chk_sec.isChecked())

    # ------------------------------------------------------------------ trace list
    def refresh_traces(self) -> None:
        self._sync = True
        sel = self.panel._selected
        self.list.clear()
        multi = self.panel.n_panels > 1
        for t in self.panel.traces:
            st = t.style
            tag = ("  [Y2]" if t.secondary_y else "") + (f"  ({chr(97 + t.panel_index)})" if multi else "")
            it = QListWidgetItem(_swatch(st.get("color", "#000"), st.get("linestyle", "-")), t.label + tag)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if t.visible else Qt.Unchecked)
            it.setToolTip(f"{t.ds.name}: {t.ds.label(t.y)} vs {t.ds.label(t.x)} — {len(t)} points")
            self.list.addItem(it)
        if sel in self.panel.traces:
            self.list.setCurrentRow(self.panel.traces.index(sel))
        self.cb_panel.blockSignals(True)
        self.cb_panel.clear()
        for i in range(self.panel.n_panels):
            self.cb_panel.addItem(f"({chr(97 + i)}) panel {i + 1}", i)
        self.cb_panel.blockSignals(False)
        self._sync = False
        self._populate_style()
        self.refresh_units()
        self._refresh_ds_list()

    def _row_changed(self, row: int):
        if self._sync:
            return
        t = self.panel.traces[row] if 0 <= row < len(self.panel.traces) else None
        self.panel._selected = t
        self._populate_style()
        self.panel.tools.on_selection_changed()

    def _item_changed(self, it: QListWidgetItem):
        if self._sync:
            return
        row = self.list.row(it)
        if 0 <= row < len(self.panel.traces):
            self.panel.set_trace_visible(self.panel.traces[row], it.checkState() == Qt.Checked)

    def _populate_style(self):
        t = self.panel.selected_trace()
        self.g_style.setEnabled(t is not None)
        if t is None:
            return
        st = t.style
        widgets = (self.ed_label, self.cb_ls, self.sp_lw, self.cb_marker, self.sp_ms, self.sp_alpha,
                   self.chk_trace_sec, self.cb_panel, self.cb_err, self.sp_err_every)
        for w in widgets:
            w.blockSignals(True)
        self.ed_label.setText(t.label)
        self.btn_color.setStyleSheet(f"background:{st['color']}; min-width:40px; border:1px solid #888;")
        self.btn_color.setText("")
        ls_name = next((k for k, v in LINESTYLES.items() if v == st["linestyle"]), "Solid")
        self.cb_ls.setCurrentText(ls_name)
        self.sp_lw.setValue(st["linewidth"])
        mk = next((k for k, v in MARKERS.items() if v == st["marker"]), "None")
        self.cb_marker.setCurrentText(mk)
        self.sp_ms.setValue(st["markersize"])
        self.sp_alpha.setValue(st["alpha"])
        self.chk_trace_sec.setChecked(t.secondary_y)
        self.cb_panel.setCurrentIndex(t.panel_index)
        self.cb_panel.setEnabled(self.panel.n_panels > 1)
        has_sd = t.sd_column is not None
        self.cb_err.setEnabled(has_sd)
        self.cb_err.setToolTip(self.cb_err.toolTip().split("\n")[0] + ("" if has_sd else
                               "\n(No '<y> SD' column in this dataset.)"))
        self.cb_err.setCurrentText(next(k for k, v in ERROR_MODES.items() if v == t.error_mode))
        self.sp_err_every.setValue(t.error_every)
        self.sp_err_every.setEnabled(has_sd and t.error_mode == "bars")
        for w in widgets:
            w.blockSignals(False)

    def _style_edit(self, **kw):
        t = self.panel.selected_trace()
        if t is not None and not self._sync:
            self.panel.set_trace_style(t, **kw)

    def _pick_color(self):
        t = self.panel.selected_trace()
        if t is None:
            return
        c = QColorDialog.getColor(QColor(t.style["color"]), self, "Trace colour")
        if c.isValid():
            self.panel.set_trace_style(t, color=c.name())

    def _sec_toggled(self, on: bool):
        t = self.panel.selected_trace()
        if t is not None:
            self.panel.set_trace_secondary(t, on)

    def _panel_changed(self, i: int):
        t = self.panel.selected_trace()
        if t is not None and i >= 0 and not self._sync:
            self.panel.set_trace_panel(t, i)

    def _err_changed(self, *_):
        t = self.panel.selected_trace()
        if t is not None and not self._sync:
            self.panel.set_trace_error(t, ERROR_MODES[self.cb_err.currentText()], self.sp_err_every.value())

    def _remove_clicked(self):
        for t in list(self.panel.selected_traces()):
            self.panel.remove_trace(t)

    def _move(self, d: int):
        t = self.panel.selected_trace()
        if t is not None:
            self.panel.move_trace(t, d)

    def _average_clicked(self):
        try:
            self.panel.average_traces(error=self.cb_avg_err.currentText())
        except ValueError as e:
            self.panel.warn(f"Average: {e}")

    # ------------------------------------------------------------------ units
    def refresh_units(self) -> None:
        for key, cb in self.cb_unit.items():
            dim = self.panel.axis_dimension(key)
            cb.blockSignals(True)
            cb.clear()
            if dim and dim in U.dimensions():
                cb.addItems(U.units_for(dim))
                cb.setCurrentText(self.panel.display_unit(key) or "")
                cb.setEnabled(len(U.units_for(dim)) > 1)
            else:
                cb.setEnabled(False)
            cb.blockSignals(False)

    def _unit_changed(self, key: str, txt: str):
        if txt:
            self.panel.set_display_unit(key, txt)
