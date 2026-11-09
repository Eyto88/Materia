"""Customisation panel (right-hand side of the Plot tab) and the figure export dialog."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox,
    QToolBox, QVBoxLayout, QWidget,
)

from . import plot_math as PM
from . import plot_templates as TPL
from .plot_palettes import COLORMAPS, palette_names
from .plot_style import (
    LAYOUT_PRESETS, LEGEND_LOCATIONS, LEGEND_MODES, MATHTEXT_FONTSETS, PANEL_LABEL_POSITIONS, PRESET_NAMES,
    TICK_DIRECTIONS, font_choices,
)

if TYPE_CHECKING:  # pragma: no cover
    from .plot_widget import PlotPanel

CM_PER_IN = 2.54


def _form(parent_lay=None) -> tuple[QWidget, QFormLayout]:
    w = QWidget()
    f = QFormLayout(w)
    f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    f.setLabelAlignment(Qt.AlignRight)
    f.setContentsMargins(6, 6, 6, 6)
    return w, f


def _size_spin(v: float, tip: str) -> QDoubleSpinBox:
    s = QDoubleSpinBox(minimum=4.0, maximum=48.0, singleStep=0.5, decimals=1, value=v)
    s.setSuffix(" pt")
    s.setToolTip(tip)
    return s


class _ColorButton(QPushButton):
    def __init__(self, tip: str):
        super().__init__()
        self.setToolTip(tip)
        self._c = "#ffffff"
        self.setFixedWidth(64)

    def set_color(self, c: str) -> None:
        self._c = c
        self.setStyleSheet(f"background:{c}; border:1px solid #888; min-height:18px;")

    def color(self) -> str:
        return self._c


class _AxisBox(QGroupBox):
    """Label-less limits/scale/invert editor for one axis."""

    def __init__(self, title: str, key: str, panel: "CustomizePanel"):
        super().__init__(title)
        self.key = key
        self.cp = panel
        g = QGridLayout(self)
        g.setContentsMargins(6, 4, 6, 4)
        self.chk_auto = QCheckBox("Auto limits")
        self.chk_auto.setToolTip("Fit the axis to the data automatically. Untick to type limits.")
        self.ed_min = QLineEdit()
        self.ed_max = QLineEdit()
        for e, n in ((self.ed_min, "minimum"), (self.ed_max, "maximum")):
            e.setValidator(QDoubleValidator())
            e.setToolTip(f"Axis {n} in the displayed unit (press Enter to apply).")
            e.setMaximumWidth(90)
        self.cb_scale = QComboBox()
        self.cb_scale.addItems(["linear", "log"])
        self.cb_scale.setToolTip("Linear or logarithmic axis (log ignores non-positive values).")
        self.chk_inv = QCheckBox("Invert")
        self.chk_inv.setToolTip("Reverse the axis direction (e.g. DSC 'exo down' conventions or cooling runs).")
        self.btn_view = QPushButton("From view")
        self.btn_view.setToolTip("Copy the current (zoomed) limits into the fields and fix them.")
        g.addWidget(self.chk_auto, 0, 0, 1, 2)
        g.addWidget(self.btn_view, 0, 2, 1, 2)
        g.addWidget(QLabel("Min"), 1, 0)
        g.addWidget(self.ed_min, 1, 1)
        g.addWidget(QLabel("Max"), 1, 2)
        g.addWidget(self.ed_max, 1, 3)
        g.addWidget(QLabel("Scale"), 2, 0)
        g.addWidget(self.cb_scale, 2, 1)
        g.addWidget(self.chk_inv, 2, 2, 1, 2)
        self.chk_auto.toggled.connect(self._changed)
        self.ed_min.editingFinished.connect(self._changed)
        self.ed_max.editingFinished.connect(self._changed)
        self.cb_scale.currentTextChanged.connect(self._changed)
        self.chk_inv.toggled.connect(self._changed)
        self.btn_view.clicked.connect(self._from_view)

    def axs(self):
        return self.cp.panel._axis_settings(self.key)

    def sync(self) -> None:
        a = self.axs()
        for w in (self.chk_auto, self.ed_min, self.ed_max, self.cb_scale, self.chk_inv):
            w.blockSignals(True)
        self.chk_auto.setChecked(a.auto_limits)
        lo, hi = a.vmin, a.vmax
        axes = self.cp.panel.ax if self.key != "y2" else self.cp.panel._ax2
        if a.auto_limits and axes is not None:  # show the current automatic limits
            lo, hi = sorted(axes.get_xlim() if self.key == "x" else axes.get_ylim())
        self.ed_min.setText(PM.fmt(lo, 5))
        self.ed_max.setText(PM.fmt(hi, 5))
        self.ed_min.setEnabled(not a.auto_limits)
        self.ed_max.setEnabled(not a.auto_limits)
        self.cb_scale.setCurrentText(a.scale)
        self.chk_inv.setChecked(a.invert)
        for w in (self.chk_auto, self.ed_min, self.ed_max, self.cb_scale, self.chk_inv):
            w.blockSignals(False)

    def _changed(self, *_):
        if self.cp._sync:
            return
        a = self.axs()
        a.auto_limits = self.chk_auto.isChecked()
        for attr, ed in (("vmin", self.ed_min), ("vmax", self.ed_max)):
            try:
                setattr(a, attr, float(ed.text().replace(",", ".")))
            except ValueError:
                pass
        a.scale = self.cb_scale.currentText()
        a.invert = self.chk_inv.isChecked()
        self.ed_min.setEnabled(not a.auto_limits)
        self.ed_max.setEnabled(not a.auto_limits)
        self.cp.apply(autoscale=True)

    def _from_view(self):
        p = self.cp.panel
        axes = p.ax if self.key != "y2" else p._ax2
        if axes is None:
            return
        lo, hi = axes.get_xlim() if self.key == "x" else axes.get_ylim()
        a = self.axs()
        a.vmin, a.vmax = sorted((float(lo), float(hi)))
        a.auto_limits = False
        self.sync()
        self.cp.apply(autoscale=True)


class CustomizePanel(QWidget):
    """Right-hand customisation panel; edits ``panel.settings`` and applies live.

    Per-panel items (title, axis labels, limits, units) refer to the panel chosen in
    the *Panel* selector at the top (the PlotPanel's active panel)."""

    def __init__(self, panel: "PlotPanel", parent=None):
        super().__init__(parent)
        self.panel = panel
        self._sync = False
        self.setMinimumWidth(200)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        head = QGridLayout()
        head.addWidget(QLabel("<b>Style preset</b>"), 0, 0)
        self.cb_preset = QComboBox()
        self.cb_preset.addItems(PRESET_NAMES)
        self.cb_preset.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_preset.setMinimumContentsLength(10)
        self.cb_preset.setToolTip("Default: clean sans-serif screen style.\nPublication: serif fonts, inward "
                                  "ticks on all sides, minor ticks, no grid, 300 dpi export.\nPresentation: "
                                  "large fonts and thick lines for slides.\nFor your own looks use Style "
                                  "templates below.")
        head.addWidget(self.cb_preset, 0, 1)
        head.addWidget(QLabel("<b>Panel</b>"), 1, 0)
        self.cb_panel = QComboBox()
        self.cb_panel.setToolTip("Panel whose title, axis labels, limits and units are edited below (and that the "
                                 "tools, span and new traces use). Set the grid in 'Layout'.")
        self.cb_panel.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_panel.setMinimumContentsLength(10)
        head.addWidget(self.cb_panel, 1, 1)
        head.setColumnStretch(1, 1)
        lay.addLayout(head)
        self.toolbox = QToolBox()
        lay.addWidget(self.toolbox, 1)

        # ---- text & fonts
        w, f = _form()
        self.ed_title = QLineEdit()
        self.ed_title.setToolTip("Title of the selected panel (empty = none). Mathtext allowed, e.g. "
                                 "$\\Delta T_{ad}$.")
        self.ed_xl, self.ed_yl, self.ed_y2l = QLineEdit(), QLineEdit(), QLineEdit()
        for e, n in ((self.ed_xl, "x"), (self.ed_yl, "y"), (self.ed_y2l, "secondary y")):
            e.setToolTip(f"Custom {n}-axis label. Leave empty for the automatic label (name and unit).")
        self.cb_font = QComboBox()
        self.cb_font.addItems(font_choices())
        self.cb_font.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_font.setMinimumContentsLength(10)
        self.cb_font.setToolTip("Font family for all text in the figure ('serif'/'sans-serif' pick the best "
                                "installed font, e.g. Times New Roman / Arial).")
        self.cb_math = QComboBox()
        self.cb_math.addItems(MATHTEXT_FONTSETS)
        self.cb_math.setToolTip("Font used for $math$ in labels: 'stix' matches Times-like serif text, "
                                "'dejavusans' matches sans-serif, 'cm' = LaTeX Computer Modern.")
        self.sp_title = _size_spin(12, "Title font size.")
        self.sp_label = _size_spin(11, "Axis label font size.")
        self.sp_tick = _size_spin(10, "Tick label font size.")
        f.addRow("Title", self.ed_title)
        f.addRow("X label", self.ed_xl)
        f.addRow("Y label", self.ed_yl)
        f.addRow("Y2 label", self.ed_y2l)
        f.addRow("Font", self.cb_font)
        f.addRow("Math font", self.cb_math)
        f.addRow("Title size", self.sp_title)
        f.addRow("Label size", self.sp_label)
        f.addRow("Tick size", self.sp_tick)
        self.toolbox.addItem(w, "Text && fonts")

        # ---- axes
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)
        self.axis_boxes = {k: _AxisBox(t, k, self) for k, t in (("x", "X axis"), ("y", "Y axis"),
                                                                 ("y2", "Secondary Y axis"))}
        for b in self.axis_boxes.values():
            v.addWidget(b)
        v.addStretch(1)
        self.toolbox.addItem(w, "Axes && limits")

        # ---- ticks & grid
        w, f = _form()
        self.cb_tickdir = QComboBox()
        self.cb_tickdir.addItems(TICK_DIRECTIONS)
        self.cb_tickdir.setToolTip("Tick direction: outward, inward (common in journals) or crossing the axis.")
        self.chk_allsides = QCheckBox("Ticks on all four sides")
        self.chk_allsides.setToolTip("Mirror ticks on the top and right axes (box style).")
        self.chk_minor = QCheckBox("Minor ticks")
        self.chk_minor.setToolTip("Show minor tick marks between major ticks.")
        self.chk_grid = QCheckBox("Major grid")
        self.chk_grid.setToolTip("Light grid lines at major ticks.")
        self.chk_gridm = QCheckBox("Minor grid")
        self.chk_gridm.setToolTip("Faint dotted grid lines at minor ticks.")
        self.sp_ticklen = QDoubleSpinBox(minimum=0.0, maximum=20.0, singleStep=0.5, decimals=1)
        self.sp_ticklen.setSuffix(" pt")
        self.sp_ticklen.setToolTip("Major tick length (minor ticks are 55 % of it).")
        self.sp_axlw = QDoubleSpinBox(minimum=0.1, maximum=5.0, singleStep=0.1, decimals=2)
        self.sp_axlw.setSuffix(" pt")
        self.sp_axlw.setToolTip("Line width of the axes frame (spines).")
        f.addRow("Direction", self.cb_tickdir)
        f.addRow("Tick length", self.sp_ticklen)
        f.addRow("Frame width", self.sp_axlw)
        for c in (self.chk_allsides, self.chk_minor, self.chk_grid, self.chk_gridm):
            f.addRow(c)
        self.toolbox.addItem(w, "Ticks && grid")

        # ---- legend
        w, f = _form()
        self.chk_leg = QCheckBox("Show legend")
        self.chk_leg.setToolTip("Legend listing each trace (drag it on the plot to move it).")
        self.cb_legmode = QComboBox()
        self.cb_legmode.addItems(LEGEND_MODES)
        self.cb_legmode.setToolTip("auto: a numbered series with more than 12 entries (e.g. 'Cycle 1…50') is "
                                   "collapsed to first ⋮ last (or dropped when a colorbar shows it).\nfull: every "
                                   "entry.\ncolorbar: such series are shown as a colorbar instead.\noff: no legend.")
        self.cb_legloc = QComboBox()
        self.cb_legloc.addItems(LEGEND_LOCATIONS)
        self.cb_legloc.setToolTip("Legend position ('best' avoids the data; 'outside right' puts it beside "
                                  "the axes).")
        self.sp_legcol = QSpinBox(minimum=1, maximum=8)
        self.sp_legcol.setToolTip("Number of legend columns (useful with many traces).")
        self.chk_legframe = QCheckBox("Frame")
        self.chk_legframe.setToolTip("Draw a box behind the legend.")
        self.sp_legsize = _size_spin(9, "Legend font size.")
        f.addRow(self.chk_leg)
        f.addRow("Legend", self.cb_legmode)
        f.addRow("Location", self.cb_legloc)
        f.addRow("Columns", self.sp_legcol)
        f.addRow("Font size", self.sp_legsize)
        f.addRow(self.chk_legframe)
        self.toolbox.addItem(w, "Legend")

        # ---- layout (multi-panel)
        w, f = _form()
        self.cb_layout = QComboBox()
        self.cb_layout.addItems(list(LAYOUT_PRESETS) + ["custom"])
        self.cb_layout.setToolTip("Grid of panels (subplots). Assign each trace to a panel in the trace style "
                                  "box; each panel has its own title, labels and limits.")
        self.sp_rows = QSpinBox(minimum=1, maximum=6)
        self.sp_rows.setToolTip("Rows of panels (custom layout).")
        self.sp_cols = QSpinBox(minimum=1, maximum=6)
        self.sp_cols.setToolTip("Columns of panels (custom layout).")
        rc = QHBoxLayout()
        rc.addWidget(self.sp_rows)
        rc.addWidget(QLabel("×"))
        rc.addWidget(self.sp_cols)
        self.btn_layout = QPushButton("Apply")
        self.btn_layout.setToolTip("Apply the custom rows × columns.")
        rc.addWidget(self.btn_layout)
        self.chk_sharex = QCheckBox("Share x axis")
        self.chk_sharex.setToolTip("All panels use panel (a)'s x limits/unit; inner x tick labels are hidden.")
        self.chk_sharey = QCheckBox("Share y axis")
        self.chk_sharey.setToolTip("All panels use panel (a)'s y limits/unit; inner y tick labels are hidden.")
        self.chk_plabels = QCheckBox("Panel labels (a), (b), …")
        self.chk_plabels.setToolTip("Automatic panel letters when the figure has more than one panel.")
        self.cb_plpos = QComboBox()
        self.cb_plpos.addItems(PANEL_LABEL_POSITIONS)
        self.cb_plpos.setToolTip("Where the panel letter is drawn.")
        self.chk_plbold = QCheckBox("Bold")
        self.chk_plbold.setToolTip("Bold panel letters.")
        f.addRow("Layout", self.cb_layout)
        f.addRow("Rows × cols", rc)
        f.addRow(self.chk_sharex)
        f.addRow(self.chk_sharey)
        f.addRow(self.chk_plabels)
        pl = QHBoxLayout()
        pl.addWidget(self.cb_plpos, 1)
        pl.addWidget(self.chk_plbold)
        f.addRow("Position", pl)
        self.toolbox.addItem(w, "Layout (multi-panel)")

        # ---- colours
        w, f = _form()
        self.cb_palette = QComboBox()
        self.cb_palette.addItems(palette_names())
        self.cb_palette.setToolTip("Categorical palettes (colour-blind safe default, Okabe–Ito, Tableau, greys) "
                                   "or a single-hue / perceptually uniform colormap sampled in N steps.")
        self.sp_steps = QSpinBox(minimum=0, maximum=200)
        self.sp_steps.setSpecialValueText("auto")
        self.sp_steps.setToolTip("Number of colours taken from a colormap ('auto' = number of traces).")
        self.btn_palette = QPushButton("Apply palette to traces")
        self.btn_palette.setToolTip("Re-colour all traces in order; new traces continue with this palette.")
        self.cb_colorby = QComboBox()
        self.cb_colorby.setToolTip("Colour traces by their order in the list, or by a numeric metadata value of "
                                   "their dataset (e.g. anneal temperature), with a colorbar.")
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(list(COLORMAPS))
        self.cb_cmap.setToolTip("Colormap for 'colour traces by'.")
        self.chk_cbar = QCheckBox("Show colorbar")
        self.chk_cbar.setChecked(True)
        self.chk_cbar.setToolTip("Add a colorbar for the values (the legend then hides numbered series in 'auto').")
        self.btn_colorby = QPushButton("Colour traces")
        self.btn_rmcbar = QPushButton("Remove colorbars")
        self.btn_rmcbar.setToolTip("Remove all colorbars from the figure.")
        f.addRow("Palette", self.cb_palette)
        f.addRow("Steps", self.sp_steps)
        f.addRow(self.btn_palette)
        f.addRow(QLabel("<i>Colour traces by</i>"))
        f.addRow("By", self.cb_colorby)
        f.addRow("Colormap", self.cb_cmap)
        f.addRow(self.chk_cbar)
        cbr = QHBoxLayout()
        cbr.addWidget(self.btn_colorby)
        cbr.addWidget(self.btn_rmcbar)
        f.addRow(cbr)
        self.toolbox.addItem(w, "Colours && colormaps")

        # ---- waterfall
        w, f = _form()
        self.sp_offy = QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=4, singleStep=1.0)
        self.sp_offy.setToolTip("Vertical shift between consecutive traces (display only — tools, cursor and "
                                "exports of data use the original values).")
        self.cb_offmode = QComboBox()
        self.cb_offmode.addItems(["absolute", "percent"])
        self.cb_offmode.setToolTip("absolute: in y display units; percent: % of the y range of the panel.")
        self.sp_offx = QDoubleSpinBox(minimum=-1e9, maximum=1e9, decimals=4, singleStep=1.0)
        self.sp_offx.setToolTip("Optional horizontal shift between consecutive traces (x display units).")
        self.chk_offlab = QCheckBox("Labels at the right end of each trace")
        self.chk_offlab.setToolTip("Write each trace's label next to its last point (instead of a legend).")
        f.addRow("Y offset", self.sp_offy)
        f.addRow("Mode", self.cb_offmode)
        f.addRow("X offset", self.sp_offx)
        f.addRow(self.chk_offlab)
        self.toolbox.addItem(w, "Waterfall / offset")

        # ---- templates
        w, f = _form()
        self.cb_tpl = QComboBox()
        self.cb_tpl.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_tpl.setMinimumContentsLength(12)
        self.cb_tpl.setToolTip("Saved looks (fonts, sizes, ticks, legend style, colour cycle, line widths, "
                               "figure size). Built-in: Thesis, Journal single column, Slides.")
        self.btn_tpl_apply = QPushButton("Apply to this figure")
        self.btn_tpl_save = QPushButton("Save current look…")
        self.btn_tpl_save.setToolTip("Save this figure's look as a named template (in ~/.labstudio/styles).")
        self.btn_tpl_del = QPushButton("Delete")
        self.btn_tpl_imp = QPushButton("Import…")
        self.btn_tpl_imp.setToolTip("Import a template file (.json) shared by a labmate.")
        self.btn_tpl_exp = QPushButton("Export…")
        self.btn_tpl_exp.setToolTip("Write the selected template to a .json file to share it.")
        self.chk_tpl_default = QCheckBox("Apply to all new figures")
        self.chk_tpl_default.setToolTip("Use the selected template as the default look for every new plot "
                                        "(stored in Preferences).")
        f.addRow("Template", self.cb_tpl)
        g1 = QHBoxLayout()
        g1.addWidget(self.btn_tpl_apply)
        g1.addWidget(self.btn_tpl_save)
        f.addRow(g1)
        g2 = QHBoxLayout()
        for b in (self.btn_tpl_imp, self.btn_tpl_exp, self.btn_tpl_del):
            g2.addWidget(b)
        f.addRow(g2)
        f.addRow(self.chk_tpl_default)
        self.toolbox.addItem(w, "Style templates")

        # ---- figure
        w, f = _form()
        self.btn_figbg = _ColorButton("Figure (outer) background colour.")
        self.btn_axbg = _ColorButton("Plot area background colour.")
        self.sp_lw = QDoubleSpinBox(minimum=0.2, maximum=10, singleStep=0.25, decimals=2)
        self.sp_lw.setToolTip("Line width applied to all traces (individual widths can be set in the "
                              "trace list).")
        self.btn_lw = QPushButton("Apply to all")
        self.btn_lw.setToolTip("Set every trace to this line width.")
        lw = QHBoxLayout()
        lw.addWidget(self.sp_lw)
        lw.addWidget(self.btn_lw)
        self.sp_ew = QDoubleSpinBox(minimum=2.0, maximum=100.0, decimals=2, singleStep=0.5)
        self.sp_eh = QDoubleSpinBox(minimum=2.0, maximum=100.0, decimals=2, singleStep=0.5)
        for sp_ in (self.sp_ew, self.sp_eh):
            sp_.setSuffix(" cm")
            sp_.setToolTip("Export size (also used by the export dialog and 'Preview at export size').")
        es = QHBoxLayout()
        es.addWidget(self.sp_ew)
        es.addWidget(QLabel("×"))
        es.addWidget(self.sp_eh)
        self.chk_preview = QCheckBox("Preview at export size")
        self.chk_preview.setToolTip("Fix the canvas to the export width × height so what you see on screen is "
                                    "exactly the exported layout (fonts, legend, margins). Scroll if it is larger "
                                    "than the window.")
        self.btn_export = QPushButton("Export figure…")
        self.btn_export.setToolTip("PNG / SVG / PDF with DPI, size in cm and transparent background.")
        f.addRow("Figure bg", self.btn_figbg)
        f.addRow("Axes bg", self.btn_axbg)
        f.addRow("Line width", lw)
        f.addRow("Export size", es)
        f.addRow(self.chk_preview)
        f.addRow(self.btn_export)
        self.toolbox.addItem(w, "Figure && export")

        # ---- signals
        self.cb_preset.currentTextChanged.connect(self._preset_changed)
        self.cb_panel.currentIndexChanged.connect(self._panel_selected)
        for e in (self.ed_title, self.ed_xl, self.ed_yl, self.ed_y2l):
            e.editingFinished.connect(self._text_changed)
        self.cb_font.currentTextChanged.connect(self._text_changed)
        self.cb_math.currentTextChanged.connect(self._text_changed)
        for sp_ in (self.sp_title, self.sp_label, self.sp_tick, self.sp_legsize, self.sp_ticklen, self.sp_axlw):
            sp_.valueChanged.connect(self._text_changed)
        self.cb_tickdir.currentTextChanged.connect(self._text_changed)
        for c in (self.chk_allsides, self.chk_minor, self.chk_grid, self.chk_gridm, self.chk_leg,
                  self.chk_legframe, self.chk_plabels, self.chk_plbold):
            c.toggled.connect(self._text_changed)
        self.cb_legloc.currentTextChanged.connect(self._text_changed)
        self.cb_legmode.currentTextChanged.connect(self._text_changed)
        self.cb_plpos.currentTextChanged.connect(self._text_changed)
        self.sp_legcol.valueChanged.connect(self._text_changed)
        self.cb_layout.currentTextChanged.connect(self._layout_preset)
        self.btn_layout.clicked.connect(self._layout_custom)
        self.chk_sharex.toggled.connect(self._share_changed)
        self.chk_sharey.toggled.connect(self._share_changed)
        self.btn_palette.clicked.connect(self._apply_palette)
        self.btn_colorby.clicked.connect(self._color_by)
        self.btn_rmcbar.clicked.connect(lambda: self.panel.remove_colorbars())
        for wd in (self.sp_offy, self.sp_offx):
            wd.valueChanged.connect(self._offsets_changed)
        self.cb_offmode.currentTextChanged.connect(self._offsets_changed)
        self.chk_offlab.toggled.connect(self._offsets_changed)
        self.btn_tpl_apply.clicked.connect(self._tpl_apply)
        self.btn_tpl_save.clicked.connect(self._tpl_save_dialog)
        self.btn_tpl_del.clicked.connect(self._tpl_delete)
        self.btn_tpl_imp.clicked.connect(self._tpl_import_dialog)
        self.btn_tpl_exp.clicked.connect(self._tpl_export_dialog)
        self.chk_tpl_default.toggled.connect(self._tpl_default_toggled)
        self.cb_tpl.currentTextChanged.connect(self._tpl_selected)
        self.btn_figbg.clicked.connect(lambda: self._pick("fig_facecolor"))
        self.btn_axbg.clicked.connect(lambda: self._pick("ax_facecolor"))
        self.btn_lw.clicked.connect(self._apply_lw)
        self.sp_ew.valueChanged.connect(self._export_size_changed)
        self.sp_eh.valueChanged.connect(self._export_size_changed)
        self.chk_preview.toggled.connect(lambda on: None if self._sync else
                                         self.panel.set_preview_export_size(on))
        self.btn_export.clicked.connect(lambda: self.panel.show_export_dialog())
        panel.tracesChanged.connect(self._update_placeholders)
        self.refresh_templates()
        self.sync_from_settings()

    # ------------------------------------------------------------------
    def apply(self, autoscale: bool = False) -> None:
        self.panel._apply_all(autoscale=autoscale)
        self.panel.settingsChanged.emit()

    def _widgets(self) -> list:
        return [self.cb_preset, self.cb_panel, self.ed_title, self.ed_xl, self.ed_yl, self.ed_y2l, self.cb_font,
                self.cb_math, self.sp_title, self.sp_label, self.sp_tick, self.cb_tickdir, self.chk_allsides,
                self.chk_minor, self.chk_grid, self.chk_gridm, self.sp_ticklen, self.sp_axlw, self.chk_leg,
                self.cb_legmode, self.cb_legloc, self.sp_legcol, self.chk_legframe, self.sp_legsize, self.sp_lw,
                self.cb_layout, self.sp_rows, self.sp_cols, self.chk_sharex, self.chk_sharey, self.chk_plabels,
                self.cb_plpos, self.chk_plbold, self.sp_offy, self.cb_offmode, self.sp_offx, self.chk_offlab,
                self.sp_ew, self.sp_eh, self.chk_preview, self.chk_tpl_default, self.cb_tpl]

    def sync_from_settings(self) -> None:
        """Refresh every widget from ``panel.settings`` (no signals emitted)."""
        p = self.panel
        s = p.settings
        self._sync = True
        ws = self._widgets()
        for w in ws:
            w.blockSignals(True)
        self.cb_preset.setCurrentText(s.preset)
        self.cb_panel.clear()
        for i in range(p.n_panels):
            self.cb_panel.addItem(f"({chr(97 + i)}) panel {i + 1}", i)
        self.cb_panel.setCurrentIndex(p.active_panel)
        self.cb_panel.setEnabled(p.n_panels > 1)
        self.ed_title.setText(s.panel(p.active_panel).title)
        self.ed_xl.setText(p._axis_settings("x").label)
        self.ed_yl.setText(p._axis_settings("y").label)
        self.ed_y2l.setText(p._axis_settings("y2").label)
        if self.cb_font.findText(s.font_family) < 0:
            self.cb_font.addItem(s.font_family)
        self.cb_font.setCurrentText(s.font_family)
        self.cb_math.setCurrentText(s.mathtext_fontset)
        self.sp_title.setValue(s.title_size)
        self.sp_label.setValue(s.label_size)
        self.sp_tick.setValue(s.tick_size)
        self.cb_tickdir.setCurrentText(s.tick_direction)
        self.chk_allsides.setChecked(s.ticks_all_sides)
        self.chk_minor.setChecked(s.minor_ticks)
        self.chk_grid.setChecked(s.grid_major)
        self.chk_gridm.setChecked(s.grid_minor)
        self.sp_ticklen.setValue(s.tick_major_len)
        self.sp_axlw.setValue(s.axes_linewidth)
        self.chk_leg.setChecked(s.legend_on)
        self.cb_legmode.setCurrentText(s.legend_mode)
        self.cb_legloc.setCurrentText(s.legend_loc)
        self.sp_legcol.setValue(int(s.legend_ncol))
        self.chk_legframe.setChecked(s.legend_frame)
        self.sp_legsize.setValue(s.legend_size)
        self.sp_lw.setValue(s.linewidth)
        lay_name = next((k for k, v in LAYOUT_PRESETS.items() if v == (s.layout_rows, s.layout_cols)), "custom")
        self.cb_layout.setCurrentText(lay_name)
        self.sp_rows.setValue(s.layout_rows)
        self.sp_cols.setValue(s.layout_cols)
        self.chk_sharex.setChecked(s.share_x)
        self.chk_sharey.setChecked(s.share_y)
        self.chk_plabels.setChecked(s.panel_labels)
        self.cb_plpos.setCurrentText(s.panel_label_pos)
        self.chk_plbold.setChecked(s.panel_label_bold)
        self.sp_offy.setValue(s.offset_y)
        self.cb_offmode.setCurrentText(s.offset_mode)
        self.sp_offy.setSuffix(" %" if s.offset_mode == "percent" else "")
        self.sp_offx.setValue(s.offset_x)
        self.chk_offlab.setChecked(s.offset_labels)
        self.sp_ew.setValue(s.export_w_cm)
        self.sp_eh.setValue(s.export_h_cm)
        self.chk_preview.setChecked(s.preview_export)
        if s.template and self.cb_tpl.findText(s.template) >= 0:
            self.cb_tpl.setCurrentText(s.template)
        self.chk_tpl_default.setChecked(bool(self.cb_tpl.currentText())
                                        and p.default_template() == self.cb_tpl.currentText())
        self.btn_figbg.set_color(s.fig_facecolor)
        self.btn_axbg.set_color(s.ax_facecolor)
        for w in ws:
            w.blockSignals(False)
        for b in self.axis_boxes.values():
            b.sync()
        self._refresh_colorby()
        self._sync = False
        self._update_placeholders()

    def _refresh_colorby(self) -> None:
        cur = self.cb_colorby.currentText()
        self.cb_colorby.blockSignals(True)
        self.cb_colorby.clear()
        self.cb_colorby.addItem("order")
        self.cb_colorby.addItems(self.panel.metadata_keys())
        if self.cb_colorby.findText(cur) >= 0:
            self.cb_colorby.setCurrentText(cur)
        self.cb_colorby.blockSignals(False)

    def _update_placeholders(self) -> None:
        if not self._sync:
            self._sync = True
            for b in self.axis_boxes.values():
                b.sync()
            self._sync = False
        for key, e in (("x", self.ed_xl), ("y", self.ed_yl), ("y2", self.ed_y2l)):
            lab = self.panel.auto_label(key)
            e.setPlaceholderText(f"auto: {lab}" if lab else "auto")
        self.axis_boxes["y2"].setEnabled(self.panel._ax2 is not None)
        self._refresh_colorby()

    def _preset_changed(self, name: str) -> None:
        if not self._sync:
            self.panel.apply_style_preset(name)

    def _panel_selected(self, i: int) -> None:
        if not self._sync and i >= 0:
            self.panel.set_active_panel(i)

    def _text_changed(self, *_):
        if self._sync:
            return
        p = self.panel
        s = p.settings
        s.panel(p.active_panel).title = self.ed_title.text()
        p._axis_settings("x").label = self.ed_xl.text()
        p._axis_settings("y").label = self.ed_yl.text()
        p._axis_settings("y2").label = self.ed_y2l.text()
        s.font_family = self.cb_font.currentText()
        s.mathtext_fontset = self.cb_math.currentText()
        s.title_size, s.label_size, s.tick_size = self.sp_title.value(), self.sp_label.value(), self.sp_tick.value()
        s.tick_direction = self.cb_tickdir.currentText()
        s.ticks_all_sides = self.chk_allsides.isChecked()
        s.minor_ticks = self.chk_minor.isChecked()
        s.grid_major = self.chk_grid.isChecked()
        s.grid_minor = self.chk_gridm.isChecked()
        s.tick_major_len = self.sp_ticklen.value()
        s.axes_linewidth = self.sp_axlw.value()
        s.legend_on = self.chk_leg.isChecked()
        s.legend_mode = self.cb_legmode.currentText()
        s.legend_loc = self.cb_legloc.currentText()
        s.legend_ncol = self.sp_legcol.value()
        s.legend_frame = self.chk_legframe.isChecked()
        s.legend_size = self.sp_legsize.value()
        s.panel_labels = self.chk_plabels.isChecked()
        s.panel_label_pos = self.cb_plpos.currentText()
        s.panel_label_bold = self.chk_plbold.isChecked()
        self.apply(autoscale=False)

    # ---- layout
    def _layout_preset(self, name: str) -> None:
        if self._sync or name not in LAYOUT_PRESETS:
            return
        r, c = LAYOUT_PRESETS[name]
        self.panel.set_layout(r, c)

    def _layout_custom(self) -> None:
        self.panel.set_layout(self.sp_rows.value(), self.sp_cols.value())

    def _share_changed(self, *_):
        if not self._sync:
            s = self.panel.settings
            self.panel.set_layout(s.layout_rows, s.layout_cols, share_x=self.chk_sharex.isChecked(),
                                  share_y=self.chk_sharey.isChecked())

    # ---- colours
    def _apply_palette(self) -> None:
        try:
            self.panel.apply_palette(self.cb_palette.currentText(), self.sp_steps.value() or None)
        except ValueError as e:
            self.panel.warn(str(e))

    def _color_by(self) -> None:
        try:
            self.panel.color_traces_by(self.cb_colorby.currentText() or "order", self.cb_cmap.currentText(),
                                       colorbar=self.chk_cbar.isChecked())
        except ValueError as e:
            self.panel.warn(str(e))

    # ---- offsets
    def _offsets_changed(self, *_):
        if self._sync:
            return
        mode = self.cb_offmode.currentText()
        self.sp_offy.setSuffix(" %" if mode == "percent" else "")
        self.panel.set_offsets(self.sp_offy.value(), mode=mode, x=self.sp_offx.value(),
                               labels=self.chk_offlab.isChecked())

    # ---- templates
    def refresh_templates(self) -> None:
        cur = self.cb_tpl.currentText()
        self.cb_tpl.blockSignals(True)
        self.cb_tpl.clear()
        self.cb_tpl.addItems(TPL.list_templates())
        if self.cb_tpl.findText(cur) >= 0:
            self.cb_tpl.setCurrentText(cur)
        self.cb_tpl.blockSignals(False)
        self._tpl_selected(self.cb_tpl.currentText())

    def _tpl_selected(self, name: str) -> None:
        self.btn_tpl_del.setEnabled(bool(name) and not TPL.is_builtin(name))
        self.chk_tpl_default.blockSignals(True)
        self.chk_tpl_default.setChecked(bool(name) and self.panel.default_template() == name)
        self.chk_tpl_default.blockSignals(False)

    def _tpl_apply(self) -> None:
        name = self.cb_tpl.currentText()
        if name:
            try:
                self.panel.apply_template(name)
            except (KeyError, ValueError, OSError) as e:
                self.panel.warn(f"Template: {e}")

    def _tpl_save_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, "Save style template", "Template name:",
                                        text=self.panel.settings.template or "My thesis style")
        if ok and name.strip():
            self.save_template(name.strip())

    def save_template(self, name: str) -> str:
        path = self.panel.save_template(name)
        self.refresh_templates()
        self.cb_tpl.setCurrentText(name)
        self.panel.statusMessage.emit(f"Template '{name}' saved to {path}")
        return path

    def _tpl_delete(self) -> None:
        name = self.cb_tpl.currentText()
        if name and TPL.delete_template(name):
            if self.panel.default_template() == name:
                self.panel.set_default_template("")
            self.refresh_templates()

    def _tpl_import_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import style template", "", "Style template (*.json)")
        if path:
            try:
                name = TPL.import_template(path)
            except (OSError, ValueError) as e:
                QMessageBox.warning(self, "Import template", str(e))
                return
            self.refresh_templates()
            self.cb_tpl.setCurrentText(name)

    def _tpl_export_dialog(self) -> None:
        name = self.cb_tpl.currentText()
        if not name:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export style template", name + ".json",
                                              "Style template (*.json)")
        if path:
            TPL.export_template(name, path)

    def _tpl_default_toggled(self, on: bool) -> None:
        if self._sync:
            return
        self.panel.set_default_template(self.cb_tpl.currentText() if on else "")

    # ---- figure
    def _pick(self, attr: str) -> None:
        c = QColorDialog.getColor(QColor(getattr(self.panel.settings, attr)), self, "Background colour")
        if c.isValid():
            setattr(self.panel.settings, attr, c.name())
            self.sync_from_settings()
            self.apply()

    def _apply_lw(self) -> None:
        self.panel.settings.linewidth = self.sp_lw.value()
        for t in self.panel.traces:
            t.line.set_linewidth(self.panel.settings.linewidth)
        self.panel._after_traces_changed(autoscale=False)

    def _export_size_changed(self, *_):
        if self._sync:
            return
        s = self.panel.settings
        s.export_w_cm, s.export_h_cm = self.sp_ew.value(), self.sp_eh.value()
        if s.preview_export:
            self.panel._sync_preview()


# =========================================================================== export


class ExportDialog(QDialog):
    """Figure export: format, DPI, size in cm, transparent background, file path."""

    FORMATS = {"PNG": "png", "SVG": "svg", "PDF": "pdf"}

    def __init__(self, panel: "PlotPanel", parent=None):
        super().__init__(parent)
        self.panel = panel
        self.setWindowTitle("Export figure")
        lay = QVBoxLayout(self)
        f = QFormLayout()
        self.cb_fmt = QComboBox()
        self.cb_fmt.addItems(list(self.FORMATS))
        self.cb_fmt.setToolTip("PNG for documents/slides; SVG/PDF are vector formats (infinitely sharp, "
                               "editable in Inkscape/Illustrator, ideal for LaTeX).")
        self.cb_dpi = QComboBox()
        self.cb_dpi.addItems(["150", "300", "600"])
        self.cb_dpi.setEditable(True)
        self.cb_dpi.setToolTip("Resolution for PNG (and raster parts of PDF/SVG). 300 dpi is the usual journal "
                               "minimum; 600 for line art.")
        self.sp_w = QDoubleSpinBox(minimum=2.0, maximum=100.0, decimals=2, singleStep=0.5)
        self.sp_h = QDoubleSpinBox(minimum=2.0, maximum=100.0, decimals=2, singleStep=0.5)
        for s in (self.sp_w, self.sp_h):
            s.setSuffix(" cm")
        self.sp_w.setToolTip("Printed width. Single journal column ≈ 8.5 cm, double ≈ 17.5 cm.")
        self.sp_h.setToolTip("Printed height.")
        sizes = QHBoxLayout()
        sizes.addWidget(self.sp_w)
        sizes.addWidget(QLabel("×"))
        sizes.addWidget(self.sp_h)
        presets = QHBoxLayout()
        for text, wcm, hcm in (("Single col.", 8.5, 6.5), ("1.5 col.", 12.0, 8.5), ("Double col.", 17.5, 11.0),
                               ("Slide", 25.0, 14.0), ("As on screen", None, None)):
            b = QPushButton(text)
            b.setToolTip("Set the export size" + (f" to {wcm} × {hcm} cm" if wcm else " to the on-screen size"))
            b.clicked.connect(lambda _=False, a=wcm, c=hcm: self._set_size(a, c))
            presets.addWidget(b)
        self.chk_transp = QCheckBox("Transparent background")
        self.chk_transp.setToolTip("No figure/axes background — for placing on coloured slides.")
        self.ed_path = QLineEdit()
        self.ed_path.setToolTip("Output file.")
        self.btn_browse = QPushButton("Browse…")
        pr = QHBoxLayout()
        pr.addWidget(self.ed_path, 1)
        pr.addWidget(self.btn_browse)
        self.lbl_px = QLabel()
        self.lbl_px.setStyleSheet("color:#555;")
        f.addRow("Format", self.cb_fmt)
        f.addRow("DPI", self.cb_dpi)
        f.addRow("Size", sizes)
        f.addRow("", presets)
        f.addRow(self.chk_transp)
        f.addRow("File", pr)
        f.addRow("", self.lbl_px)
        lay.addLayout(f)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._save_clicked)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.cb_dpi.setCurrentText(str(panel.settings.export_dpi))
        self._set_size(panel.settings.export_w_cm, panel.settings.export_h_cm)
        self.ed_path.setText(os.path.join(os.path.expanduser("~"), "figure.png"))
        self.btn_browse.clicked.connect(self._browse)
        self.cb_fmt.currentTextChanged.connect(self._fmt_changed)
        for s in (self.sp_w, self.sp_h):
            s.valueChanged.connect(self._update_px)
        self.cb_dpi.currentTextChanged.connect(self._update_px)
        self._update_px()

    def _set_size(self, wcm: float | None, hcm: float | None) -> None:
        if wcm is None:
            w, h = self.panel.figure.get_size_inches()
            if self.panel.settings.preview_export:
                wcm, hcm = self.panel.settings.export_w_cm, self.panel.settings.export_h_cm
                self.sp_w.setValue(wcm)
                self.sp_h.setValue(hcm)
                return
            wcm, hcm = w * CM_PER_IN, h * CM_PER_IN
        self.sp_w.setValue(wcm)
        self.sp_h.setValue(hcm)

    def dpi(self) -> int:
        try:
            return max(10, int(float(self.cb_dpi.currentText())))
        except ValueError:
            return 300

    def size_in(self) -> tuple[float, float]:
        return self.sp_w.value() / CM_PER_IN, self.sp_h.value() / CM_PER_IN

    def fmt(self) -> str:
        return self.FORMATS[self.cb_fmt.currentText()]

    def _update_px(self, *_):
        w, h = self.size_in()
        d = self.dpi()
        self.lbl_px.setText(f"{round(w * d)} × {round(h * d)} px at {d} dpi")

    def _fmt_changed(self, *_):
        p = self.ed_path.text()
        root, _ext = os.path.splitext(p)
        self.ed_path.setText(root + "." + self.fmt())

    def _browse(self):
        ext = self.fmt()
        path, _ = QFileDialog.getSaveFileName(self, "Export figure", self.ed_path.text(),
                                              f"{ext.upper()} (*.{ext})")
        if path:
            if not path.lower().endswith("." + ext):
                path += "." + ext
            self.ed_path.setText(path)

    def do_export(self, path: str | None = None) -> str:
        """Export with the dialog's current options (no GUI interaction). Returns the path."""
        path = path or self.ed_path.text()
        st = self.panel.settings
        st.export_w_cm, st.export_h_cm = self.sp_w.value(), self.sp_h.value()
        if st.preview_export:
            self.panel._sync_preview()
        self.panel.customize_panel.sync_from_settings()
        self.panel.export_figure(path, self.fmt(), self.dpi(), self.chk_transp.isChecked(), self.size_in())
        self.panel.statusMessage.emit(f"Exported {path}")
        return path

    def _save_clicked(self):
        try:
            self.do_export()
        except Exception as e:  # noqa: BLE001 — surface any I/O error to the user
            QMessageBox.critical(self, "Export failed", str(e))
            return
        self.accept()
