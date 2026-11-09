"""Micrographs tab: an ImageJ-style workspace for SEM and light optical micrographs.

Layout::

    ┌ images list ┐┌ action toolbar ───────────────────────┐┌ calibration ┐
    │             ││tools│ viewer (zoom/pan/ROIs/overlays)  ││ QToolBox:   │
    ├ ROI manager ┤│     │                                  ││ Adjust …    │
    │             │├ status line ───────────────────────────┤│ Annotate    │
    └─────────────┘└ Results | Summary | Plot ──────────────┘└─────────────┘

Public API used by the main window: ``ImageWorkspace(parent=None)``, ``open_image(path)``,
``open_dialog()``, ``IMAGE_EXTENSIONS``, ``add_image``, ``export_image``, ``export_results_csv``,
``panels``; session hooks ``get_state()`` / ``set_state(state, id_map)`` /
``get_state_files()`` / ``set_state_files(files)``.

v1.1: calibration source shown next to the scale and recorded in every result's provenance;
dpi-only images open uncalibrated with a banner; heavy operations run through
``gui.worker.run_task``; burned-in scale bars / labels and the SEM data bar are excluded from
analyses by default; side panels scroll and collapse so the tab fits 1200×680.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGroupBox, QHBoxLayout, QLabel, QLayout, QListWidget,
                               QListWidgetItem, QPushButton, QSpinBox, QSplitter, QTabWidget,
                               QToolBar, QToolBox, QToolButton, QVBoxLayout, QWidget, QCheckBox,
                               QScrollArea, QSizePolicy)

from labstudio.core import image_ops as iops
from labstudio.core.image_ops import Calibration
from labstudio.gui.image_results import PlotWidget, ResultsTable
from labstudio.gui.image_viewer import ImageViewer, Roi, TOOLS

IMAGE_EXTENSIONS: list[str] = list(iops.IMAGE_EXTENSIONS)
UNDO_LIMIT = 25
#: Undo snapshots are also bounded by memory (bytes, all snapshots of one image together).
UNDO_MEMORY_LIMIT = 1 << 30

LENGTH_KEYS = {"Perimeter", "ECD", "Major", "Minor", "Feret", "MinFeret", "X", "Y", "Length",
               "BX", "BY"}
AREA_KEYS = {"Area"}
ANGLE_KEYS = {"Angle", "FeretAngle"}


def unitize(row: dict, cal: Calibration) -> dict:
    """Rename measurement keys to carry units: 'Area' → 'Area (µm²)', 'Feret' → 'Feret (µm)'."""
    out = {}
    for k, v in row.items():
        if k in AREA_KEYS:
            out[f"{k} ({cal.area_unit})"] = v
        elif k in LENGTH_KEYS:
            out[f"{k} ({cal.unit})"] = v
        elif k in ANGLE_KEYS:
            out[f"{k} (°)"] = v
        else:
            out[k] = v
    return out


@dataclass
class Snapshot:
    data: np.ndarray
    mask: np.ndarray | None
    calibration: Calibration
    label: str

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes + (self.mask.nbytes if self.mask is not None else 0))


@dataclass
class ImageDocument:
    """One open image with its processing history, binary mask, ROIs and overlays."""

    name: str
    data: np.ndarray
    calibration: Calibration
    path: str = ""
    metadata: dict = field(default_factory=dict)
    n_pages: int = 1
    page: int = 0
    mask: np.ndarray | None = None
    window: tuple[float, float] = (0.0, 255.0)
    rois: list[Roi] = field(default_factory=list)
    annotations: list[dict] = field(default_factory=list)
    shapes: list[dict] = field(default_factory=list)
    overlay_rgba: np.ndarray | None = None
    show_mask: bool = False
    undo_stack: list[Snapshot] = field(default_factory=list)
    redo_stack: list[Snapshot] = field(default_factory=list)
    original: Snapshot | None = None
    scale_bar: dict | None = None
    data_bar_row: int | None = None        # detected SEM info bar (first row), if not cropped
    print_dpi: float | None = None         # generic dpi tag found in the file (not a scale)

    def __post_init__(self) -> None:
        if self.original is None:
            self.original = Snapshot(self.data, None, self.calibration, "original")
        self._excl_cache: tuple | None = None
        self.reset_window()

    def exclusion_boxes(self) -> list[tuple[int, int, int, int]]:
        """Burned-in scale bars / labels and the SEM data bar (auto-detected, cached per array)."""
        key = (id(self.data), self.data.shape, self.data_bar_row)
        if self._excl_cache is None or self._excl_cache[0] != key:
            try:
                boxes = iops.detect_annotation_regions(self.data, self.data_bar_row)
            except Exception:  # noqa: BLE001 - detection must never block analysis
                boxes = []
            self._excl_cache = (key, boxes)
        return list(self._excl_cache[1])

    def exclusion_mask(self) -> np.ndarray | None:
        boxes = self.exclusion_boxes()
        return iops.regions_mask(self.shape, boxes) if boxes else None

    @property
    def gray(self) -> np.ndarray:
        return iops.to_gray(self.data)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[:2]

    def reset_window(self, auto: bool = False) -> None:
        if auto:
            self.window = iops.auto_contrast_limits(self.data)
        elif self.data.dtype == np.uint8:
            self.window = (0.0, 255.0)
        else:
            g = self.gray
            lo, hi = float(np.min(g)), float(np.max(g))
            self.window = (lo, hi if hi > lo else lo + 1)

    def snapshot(self, label: str) -> Snapshot:
        return Snapshot(self.data, None if self.mask is None else self.mask.copy(), self.calibration,
                        label)

    def push_undo(self, label: str) -> None:
        self.undo_stack.append(self.snapshot(label))
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack.pop(0)
        # memory bound (identical arrays shared between snapshots are counted once)
        while len(self.undo_stack) > 1:
            seen, total = set(), 0
            for snap in self.undo_stack:
                if id(snap.data) not in seen:
                    seen.add(id(snap.data))
                    total += snap.data.nbytes
                total += snap.mask.nbytes if snap.mask is not None else 0
            if total <= UNDO_MEMORY_LIMIT:
                break
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def restore(self, s: Snapshot) -> None:
        shape_changed = s.data.shape[:2] != self.data.shape[:2]
        self.data, self.mask, self.calibration = s.data, s.mask, s.calibration
        if shape_changed:
            self.rois.clear()
            self.shapes.clear()
            self.overlay_rgba = None


class ColumnFlowLayout(QLayout):
    """Lays widgets out top-to-bottom and wraps into further columns when the height runs out
    (the tool column of the workspace), so the tools never force a tall minimum height."""

    def __init__(self, parent=None, spacing: int = 2):
        super().__init__(parent)
        self._items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):  # noqa: N802
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):  # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):  # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations(0)

    def hasHeightForWidth(self):  # noqa: N802
        return False

    def _col_width(self) -> int:
        return max([it.sizeHint().width() for it in self._items] or [0])

    def _layout(self, rect: QRect, apply: bool) -> int:
        x, y = rect.x(), rect.y()
        cw = self._col_width()
        ncols = 1
        for it in self._items:
            h = it.sizeHint().height()
            if y + h > rect.bottom() + 1 and y > rect.y():
                x += cw + self.spacing()
                y = rect.y()
                ncols += 1
            if apply:
                it.setGeometry(QRect(QPoint(x, y), QSize(cw, h)))
            y += h + self.spacing()
        return ncols

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        n = self._layout(rect, True)
        want = n * self._col_width() + (n - 1) * self.spacing()
        par = self.parentWidget()
        if par is not None and par.minimumWidth() != want:
            par.setMinimumWidth(want)

    def sizeHint(self):  # noqa: N802
        h = sum(it.sizeHint().height() + self.spacing() for it in self._items)
        return QSize(self._col_width(), min(h, 400))

    def minimumSize(self):  # noqa: N802
        return QSize(self._col_width(), max([it.sizeHint().height() for it in self._items] or [0]))


class ImageWorkspace(QWidget):
    """The whole Micrographs tab (see module docstring)."""

    IMAGE_EXTENSIONS = IMAGE_EXTENSIONS
    imageOpened = Signal(str)
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.docs: list[ImageDocument] = []
        self._last_opened: ImageDocument | None = None
        self._current = -1
        self.global_calibration: Calibration | None = None
        self._preview_rgba: np.ndarray | None = None
        self._current_roi_id: int | None = None
        self._build_ui()
        self._update_actions()

    # ================================================================= UI construction
    def _build_ui(self) -> None:
        from labstudio.gui import image_panels as P

        self.viewer = ImageViewer()
        self.viewer.mouseMoved.connect(self._on_mouse_moved)
        self.viewer.roiCreated.connect(self._on_roi_created)
        self.viewer.clickedAt.connect(self._on_viewer_click)
        self.viewer.arrowCreated.connect(self._on_arrow)
        self.viewer.zoomChanged.connect(lambda z: self._update_status())

        # ---- left: images + ROI manager
        self.image_list = QListWidget()
        self.image_list.setToolTip("Open images — click to switch")
        self.image_list.currentRowChanged.connect(self.set_current)
        b_open = QPushButton("Open…")
        b_open.clicked.connect(self.open_dialog)
        b_close = QPushButton("Close")
        b_close.clicked.connect(lambda: self.close_image(self._current))
        img_box = QGroupBox("Images")
        il = QVBoxLayout(img_box)
        il.setContentsMargins(4, 4, 4, 4)
        il.addWidget(self.image_list)
        hb = QHBoxLayout()
        hb.addWidget(b_open)
        hb.addWidget(b_close)
        il.addLayout(hb)

        self.roi_list = QListWidget()
        self.roi_list.setToolTip("ROI manager — select a ROI to make it current")
        self.roi_list.currentRowChanged.connect(self._on_roi_selected)
        roi_box = QGroupBox("ROI manager")
        rl = QVBoxLayout(roi_box)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.addWidget(self.roi_list)
        g = QHBoxLayout()
        b_meas = QPushButton("Measure")
        b_meas.setToolTip("Measure the selected ROI (M)")
        b_meas.clicked.connect(lambda: self.measure_roi())
        b_all = QPushButton("All")
        b_all.setToolTip("Measure all ROIs")
        b_all.clicked.connect(self.measure_all_rois)
        b_del = QPushButton("Delete")
        b_del.clicked.connect(self.delete_current_roi)
        for b in (b_meas, b_all, b_del):
            g.addWidget(b)
        rl.addLayout(g)
        g2 = QHBoxLayout()
        b_prof = QPushButton("Profile")
        b_prof.setToolTip("Plot the intensity profile along the selected line ROI")
        b_prof.clicked.connect(lambda: self.plot_profile())
        b_clr = QPushButton("Clear")
        b_clr.clicked.connect(self.clear_rois)
        g2.addWidget(b_prof)
        g2.addWidget(b_clr)
        rl.addLayout(g2)
        lw = QHBoxLayout()
        lw.addWidget(QLabel("Line width"))
        self.line_width = QSpinBox()
        self.line_width.setRange(1, 200)
        self.line_width.setSuffix(" px")
        self.line_width.setToolTip("Width of line selections for profiles / line measurements "
                                   "(values averaged perpendicular to the line)")
        lw.addWidget(self.line_width)
        rl.addLayout(lw)

        left_split = QSplitter(Qt.Vertical)
        left_split.addWidget(img_box)
        left_split.addWidget(roi_box)
        left_split.setSizes([200, 400])
        for lw_ in (self.image_list, self.roi_list):
            lw_.setMinimumHeight(40)
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setFrameShape(QFrame.NoFrame)
        left.setWidget(left_split)
        left.setMinimumWidth(120)
        left_split.setMinimumWidth(110)
        self.left_panel = left

        # ---- centre: toolbars + viewer + status + banner
        self.action_bar = QToolBar("Actions")
        self.action_bar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._actions: dict[str, QAction] = {}

        def act(key, text, slot, tip, shortcut=None, bar=self.action_bar):
            a = QAction(text, self)
            a.setToolTip(tip + (f"  ({shortcut})" if shortcut else ""))
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
                a.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            self.addAction(a)
            bar.addAction(a)
            self._actions[key] = a
            return a

        act("open", "Open", self.open_dialog, "Open micrograph(s)")
        act("export", "Export image", self.export_dialog, "Export image as PNG/TIFF with or without overlays")
        self.action_bar.addSeparator()
        act("undo", "Undo", self.undo, "Undo last image operation", "Ctrl+Z")
        act("redo", "Redo", self.redo, "Redo", "Ctrl+Y")
        act("revert", "Revert", self.revert, "Revert to the original image as loaded")
        self.action_bar.addSeparator()
        act("fit", "Fit", self.viewer.fit, "Fit image to window", "F")
        act("100", "1:1", self.viewer.zoom_100, "Zoom 100 % (one image pixel per screen pixel)")
        act("zin", "+", lambda: self.viewer.zoom_by(1.25), "Zoom in (mouse wheel)")
        act("zout", "−", lambda: self.viewer.zoom_by(0.8), "Zoom out (mouse wheel)")
        self.action_bar.addSeparator()
        act("measure", "Measure", lambda: self.measure_roi(), "Measure the current ROI", "M")
        a = act("overlays", "Overlays", lambda on: self.viewer.set_overlays_visible(on),
                "Show/hide overlays, ROIs and annotations")
        a.setCheckable(True)
        a.setChecked(True)
        act("clear_ov", "Clear overlay", self.clear_analysis_overlay, "Remove analysis overlays")
        self.action_bar.addSeparator()
        a = act("toggle_left", "◧ Images", lambda on: self.left_panel.setVisible(on),
                "Show/hide the image list and ROI manager")
        a.setCheckable(True)
        a.setChecked(True)
        a = act("toggle_right", "Panels ◨", lambda on: self.right_panel.setVisible(on),
                "Show/hide the calibration and tool panels")
        a.setCheckable(True)
        a.setChecked(True)

        self.tool_bar = QWidget()   # wrapping column of tool buttons
        self.tool_bar.setLayout(ColumnFlowLayout(spacing=1))
        self.tool_bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Ignored)
        self._tool_group = QActionGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_actions: dict[str, QAction] = {}
        tool_defs = [
            ("pan", "Pan", "Pan / zoom (drag; wheel zooms; middle button pans in any tool)"),
            ("line", "Line", "Straight line: length and angle (drag)"),
            ("polyline", "Segmented", "Segmented line (click vertices, double-click / Enter to finish)"),
            ("angle", "Angle", "Angle tool: click 3 points (arm, vertex, arm)"),
            ("rect", "Rect", "Rectangle ROI (drag)"),
            ("ellipse", "Ellipse", "Elliptical ROI (drag)"),
            ("polygon", "Polygon", "Polygon ROI (click vertices, double-click / Enter to close)"),
            ("freehand", "Freehand", "Freehand ROI (drag)"),
            ("point", "Points", "Multi-point counter (click points, double-click / Enter to finish)"),
            ("scale", "Set scale", "Draw a line along the scale bar, then enter its known length"),
            ("profile", "Profile", "Draw a line to plot the intensity profile"),
            ("text", "Text", "Click to place the text set in the Annotate panel"),
            ("arrow", "Arrow", "Drag to draw an arrow annotation"),
        ]
        for key, text, tip in tool_defs:
            a = QAction(text, self)
            a.setCheckable(True)
            a.setToolTip(tip)
            a.triggered.connect(lambda _=False, k=key: self.set_tool(k))
            self._tool_group.addAction(a)
            tb = QToolButton()
            tb.setDefaultAction(a)
            tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
            tb.setAutoRaise(True)
            self.tool_bar.layout().addWidget(tb)
            self._tool_actions[key] = a
        self._tool_actions["pan"].setChecked(True)

        self.banner = QFrame()
        self.banner.setFrameShape(QFrame.StyledPanel)
        self.banner.setStyleSheet("QFrame { background: #fff4ce; } QLabel { color: #5c4400; }")
        bl = QHBoxLayout(self.banner)
        bl.setContentsMargins(6, 2, 6, 2)
        self.banner_label = QLabel("")
        self.banner_label.setWordWrap(True)
        self.banner_btn = QPushButton("Crop")
        self.banner_btn.clicked.connect(self._banner_accept)
        self.banner_btn2 = QPushButton("")
        self.banner_btn2.clicked.connect(self._banner_secondary)
        self.banner_btn2.hide()
        self.page_spin = QSpinBox()
        self.page_spin.setPrefix("page ")
        self.page_spin.setVisible(False)
        b_dis = QPushButton("Dismiss")
        b_dis.clicked.connect(self._banner_dismiss)
        bl.addWidget(self.banner_label, 1)
        bl.addWidget(self.page_spin)
        bl.addWidget(self.banner_btn)
        bl.addWidget(self.banner_btn2)
        bl.addWidget(b_dis)
        self.banner.hide()
        self._banner_action = None
        self._banner_action2 = None
        self._banner_queue: list[tuple] = []

        self.status_label = QLabel("Open a micrograph (TIFF/PNG/JPG/BMP) to start.")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_label.setMinimumWidth(80)
        self.status_label.setWordWrap(False)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.pixel_label = QLabel("")
        self.pixel_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.pixel_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        view_row = QHBoxLayout()
        view_row.setContentsMargins(0, 0, 0, 0)
        view_row.setSpacing(2)
        view_row.addWidget(self.tool_bar)
        view_row.addWidget(self.viewer, 1)
        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(2)
        cl.addWidget(self.action_bar)
        cl.addWidget(self.banner)
        cl.addLayout(view_row, 1)
        srow = QHBoxLayout()
        srow.addWidget(self.status_label, 1)
        srow.addWidget(self.pixel_label)
        cl.addLayout(srow)

        # ---- bottom: results / summary / plot
        self.results = ResultsTable("Results")
        self.summary = ResultsTable("Summary")
        self.plot = PlotWidget()
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.addTab(self.results, "Results")
        self.bottom_tabs.addTab(self.summary, "Summary")
        self.bottom_tabs.addTab(self.plot, "Plot")

        centre_split = QSplitter(Qt.Vertical)
        centre_split.addWidget(centre)
        centre_split.addWidget(self.bottom_tabs)
        centre_split.setStretchFactor(0, 3)
        centre_split.setStretchFactor(1, 1)
        centre_split.setSizes([620, 240])
        self.bottom_tabs.setMinimumHeight(80)

        # ---- right: calibration + toolbox
        self.cal_box = self._build_calibration_box()
        self.toolbox = QToolBox()
        self.panels = {
            "adjust": P.AdjustPanel(self),
            "process": P.ProcessPanel(self),
            "threshold": P.ThresholdPanel(self),
            "particles": P.ParticlesPanel(self),
            "grain": P.GrainSizePanel(self),
            "phase": P.PhasePanel(self),
            "histogram": P.HistogramPanel(self),
            "annotate": P.AnnotatePanel(self),
        }
        titles = {"adjust": "Adjust", "process": "Process", "threshold": "Threshold && Binary",
                  "particles": "Analyze particles", "grain": "Grain size (ASTM E112)",
                  "phase": "Phase fraction", "histogram": "Histogram", "annotate": "Annotate && scale bar"}
        for k, pnl in self.panels.items():
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setFrameShape(QFrame.NoFrame)
            sc.setWidget(pnl)
            self.toolbox.addItem(sc, titles[k])
        self.toolbox.currentChanged.connect(self._on_panel_changed)
        right_inner = QWidget()
        rlay = QVBoxLayout(right_inner)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.addWidget(self.cal_box)
        rlay.addWidget(self.toolbox, 1)
        self.toolbox.setMinimumHeight(260)
        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setFrameShape(QFrame.NoFrame)
        right.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right.setWidget(right_inner)
        right.setMinimumWidth(200)
        self.right_panel = right

        main = QSplitter(Qt.Horizontal)
        main.addWidget(left)
        main.addWidget(centre_split)
        main.addWidget(right)
        main.setStretchFactor(1, 1)
        main.setCollapsible(1, False)
        main.setSizes([185, 925, 370])
        self.main_split = main
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(main)

    def _build_calibration_box(self) -> QGroupBox:
        box = QGroupBox("Spatial calibration")
        f = QFormLayout(box)
        f.setContentsMargins(6, 4, 6, 4)
        f.setVerticalSpacing(3)
        self.cal_label = QLabel("Uncalibrated (pixels)")
        self.cal_label.setWordWrap(True)
        f.addRow(self.cal_label)
        self.cal_source_label = QLabel("source: uncalibrated")
        self.cal_source_label.setWordWrap(True)
        self.cal_source_label.setToolTip("Where the scale came from. Only SEM metadata (FEI/Zeiss/JEOL/"
                                         "Hitachi), ImageJ/OME metadata, a LabStudio export or "
                                         "'Set scale' from the scale bar are trusted.")
        f.addRow(self.cal_source_label)
        self.scale_px = QDoubleSpinBox()
        self.scale_px.setRange(0, 1e6)
        self.scale_px.setDecimals(2)
        self.scale_px.setSuffix(" px")
        self.scale_px.setToolTip("Distance in pixels (filled in by drawing a line with the 'Set scale' "
                                 "tool along the micron bar). Use 1 px to type a pixel size directly.")
        self.scale_dist = QDoubleSpinBox()
        self.scale_dist.setRange(0, 1e9)
        self.scale_dist.setDecimals(4)
        self.scale_dist.setValue(10)
        self.scale_dist.setToolTip("Known physical length of that distance")
        self.scale_unit = QComboBox()
        self.scale_unit.addItems(["nm", "µm", "mm"])
        self.scale_unit.setCurrentText("µm")
        row = QHBoxLayout()
        row.addWidget(self.scale_px)
        row.addWidget(QLabel("="))
        f.addRow(row)
        row1 = QHBoxLayout()
        row1.addWidget(self.scale_dist)
        row1.addWidget(self.scale_unit)
        f.addRow(row1)
        row2 = QHBoxLayout()
        self.scale_global = QCheckBox("Global")
        self.scale_global.setToolTip("Apply this scale to all open images and to new images without "
                                     "metadata (ImageJ 'Global')")
        b_set = QPushButton("Set scale")
        b_set.clicked.connect(self.set_scale_from_fields)
        b_rm = QPushButton("Remove")
        b_rm.setToolTip("Remove calibration (pixels)")
        b_rm.clicked.connect(lambda: self.set_calibration(Calibration()))
        self.disp_unit = QComboBox()
        self.disp_unit.addItems(["nm", "µm", "mm"])
        self.disp_unit.setToolTip("Display unit for calibrated measurements")
        self.disp_unit.activated.connect(self._change_display_unit)
        row2.addWidget(b_set)
        row2.addWidget(b_rm)
        f.addRow(row2)
        row3 = QHBoxLayout()
        row3.addWidget(self.scale_global)
        row3.addWidget(QLabel("unit"))
        row3.addWidget(self.disp_unit)
        f.addRow(row3)
        return box

    def _on_panel_changed(self, index: int) -> None:
        thr = self.panels["threshold"]
        if self.toolbox.widget(index).widget() is thr:
            thr.update_preview()
        elif self._preview_rgba is not None:
            self.set_preview(None)

    # ================================================================= documents
    @property
    def doc(self) -> ImageDocument | None:
        return self.docs[self._current] if 0 <= self._current < len(self.docs) else None

    def require_doc(self) -> ImageDocument | None:
        d = self.doc
        if d is None:
            self.status("Open an image first.")
        return d

    @property
    def cal(self) -> Calibration:
        return self.doc.calibration if self.doc else Calibration()

    def open_dialog(self) -> list[ImageDocument]:
        """File dialog to open one or more micrographs."""
        pat = " ".join(f"*{e}" for e in IMAGE_EXTENSIONS)
        paths, _ = QFileDialog.getOpenFileNames(self, "Open micrograph", "",
                                                f"Images ({pat});;All files (*)")
        return [self.open_image(p) for p in paths]

    def open_image(self, path: str, page: int = 0) -> ImageDocument | None:
        """Open an image file (calibration from trusted SEM/ImageJ/OME metadata when available).

        Loading runs through ``run_task`` (busy dialog for big files); returns the document
        immediately in synchronous mode (tests), else None (the image appears when loaded).
        """
        from labstudio.gui.worker import run_task

        def load():
            li = iops.load_image(path, page=page)
            bar = iops.detect_data_bar(li.data)
            return li, bar

        return run_task(self, f"Loading {os.path.basename(path)}…", load,
                        on_done=lambda res: self._opened(path, *res),
                        on_error=lambda tb, exc: self.status(f"Could not open {path}: {exc}")) and \
            self._last_opened

    def _opened(self, path: str, li, bar) -> ImageDocument:
        cal = li.calibration
        if not cal.calibrated and self.global_calibration is not None:
            cal = self.global_calibration
        doc = ImageDocument(name=li.name if li.n_pages == 1 else f"{li.name} [p{li.page + 1}]",
                            data=li.data, calibration=cal, path=path, metadata=li.metadata,
                            n_pages=li.n_pages, page=li.page)
        doc.data_bar_row = bar
        doc.print_dpi = li.metadata.get("print_dpi")
        self._add_doc(doc)
        self._last_opened = doc
        msg = [f"Opened {doc.name}: {doc.shape[1]}×{doc.shape[0]} px, {doc.data.dtype}"]
        msg.append(f"scale: {cal.describe()} (source: {cal.source_label})" if cal.calibrated
                   else "uncalibrated")
        self.status("; ".join(msg))
        self._banner_queue = []
        if doc.print_dpi and not cal.calibrated:
            self._queue_banner(iops.print_resolution_banner(doc.print_dpi), "Set scale",
                               lambda: self.set_tool("scale"), secondary=("Use dpi anyway",
                                                                          self.use_print_resolution))
        if bar is not None:
            self._queue_banner(f"SEM data bar detected (rows {bar}–{doc.shape[0]}); it is excluded "
                               "from analyses. Crop it from the image?", "Crop data bar",
                               lambda r=bar: self.crop_data_bar(r))
        elif li.n_pages > 1:
            self.page_spin.setRange(1, li.n_pages)
            self.page_spin.setValue(li.page + 1)
            self._queue_banner(f"Multi-page TIFF with {li.n_pages} pages — showing page {li.page + 1}.",
                               "Load page", lambda: self.open_image(path, self.page_spin.value() - 1),
                               page=True)
        self._next_banner()
        self.imageOpened.emit(path)
        return doc

    def use_print_resolution(self) -> None:
        """'Use dpi anyway': calibrate from the file's print resolution (flagged in every result)."""
        doc = self.require_doc()
        if doc is None or not doc.print_dpi:
            return
        self.set_calibration(iops.calibration_from_print_resolution(doc.print_dpi), global_=False)
        self.status(f"Calibrated from {doc.print_dpi:g} dpi — results will carry a warning.")

    def add_image(self, data: np.ndarray, name: str = "untitled",
                  calibration: Calibration | None = None) -> ImageDocument:
        """Add an in-memory array as a new image."""
        doc = ImageDocument(name=name, data=np.asarray(data), calibration=calibration or
                            (self.global_calibration or Calibration()))
        self._add_doc(doc)
        return doc

    def _add_doc(self, doc: ImageDocument) -> None:
        self.docs.append(doc)
        it = QListWidgetItem(doc.name)
        it.setToolTip(doc.path or doc.name)
        self.image_list.addItem(it)
        self.image_list.setCurrentRow(len(self.docs) - 1)

    def close_image(self, index: int) -> None:
        if not (0 <= index < len(self.docs)):
            return
        self.docs.pop(index)
        self.image_list.blockSignals(True)
        self.image_list.takeItem(index)
        self.image_list.blockSignals(False)
        self._current = -1
        if self.docs:
            self.image_list.setCurrentRow(min(index, len(self.docs) - 1))
            self.set_current(self.image_list.currentRow())
        else:
            self.viewer.set_image(None)
            self.viewer.clear_rois()
            self.viewer.clear_overlays()
            self.viewer.set_annotations([])
            self.viewer.set_scale_bar(None)
            self.roi_list.clear()
            self._update_actions()
            self._update_cal_label()

    def set_current(self, index: int) -> None:
        if index == self._current or not (0 <= index < len(self.docs)):
            return
        self._current = index
        self._preview_rgba = None
        self._banner_queue = []
        self.banner.hide()
        self.refresh(fit=True, rebuild_items=True)
        for p in self.panels.values():
            if hasattr(p, "on_document_changed"):
                p.on_document_changed()

    # ================================================================= display
    def display_array(self, doc: ImageDocument | None = None) -> np.ndarray | None:
        doc = doc or self.doc
        if doc is None:
            return None
        if doc.show_mask and doc.mask is not None:
            return doc.mask.astype(np.uint8) * 255
        return iops.apply_window(doc.data, *doc.window)

    def refresh(self, fit: bool = False, rebuild_items: bool = False) -> None:
        """Redraw the current document (image, overlays, ROIs, annotations, scale bar)."""
        doc = self.doc
        if doc is None:
            return
        self.viewer.set_image(self.display_array(doc), fit=fit)
        self.viewer.set_overlay(self._preview_rgba if self._preview_rgba is not None else doc.overlay_rgba)
        self.viewer.set_shapes(doc.shapes)
        self.viewer.set_annotations(doc.annotations)
        self.viewer.set_scale_bar(self._scale_bar_spec(doc))
        if rebuild_items:
            self.viewer.clear_rois()
            for r in doc.rois:
                self.viewer.add_roi_item(r)
            self._rebuild_roi_list()
        self._update_actions()
        self._update_cal_label()
        self._update_status()

    @property
    def overlay_step(self) -> int:
        """Subsampling for RGBA overlays (1, or the display-pyramid step for images > 4k px)."""
        doc = self.doc
        if doc is None:
            return 1
        from labstudio.gui.image_viewer import PYRAMID_LIMIT

        return max(1, int(math.ceil(max(doc.shape) / PYRAMID_LIMIT)))

    def overlay_view(self, arr: np.ndarray) -> np.ndarray:
        """``arr[::step, ::step]`` — the resolution at which overlays are built."""
        st = self.overlay_step
        return arr[::st, ::st] if st > 1 else arr

    # ================================================================= provenance
    def image_sha256(self, doc: ImageDocument | None = None) -> str | None:
        doc = doc or self.doc
        if doc is None:
            return None
        if doc.path and os.path.isfile(doc.path):
            from labstudio.core.provenance import file_sha256

            return file_sha256(doc.path)
        return hashlib.sha256(np.ascontiguousarray(doc.data).tobytes()).hexdigest()

    def make_provenance(self, analysis: str, params: dict | None = None,
                        warnings: list[str] | None = None, doc: ImageDocument | None = None) -> dict:
        """Provenance record for an analysis on ``doc`` (calibration + its source, image file +
        SHA-256, parameters, warnings)."""
        from labstudio.core.provenance import provenance

        doc = doc or self.doc
        c = doc.calibration if doc else Calibration()
        warns = list(warnings or [])
        if c.from_print_resolution and not any("dpi" in w for w in warns):
            warns.append("Calibration taken from a print resolution (dpi), not from the microscope.")
        extra = {"image": doc.name if doc else "", "image_sha256": self.image_sha256(doc),
                 "calibration": c.describe(), "calibration_source": c.source_label,
                 "image_shape": list(doc.shape) if doc else None, "warnings": warns}
        return provenance(analysis, params or {}, source_path=(doc.path or None) if doc else None,
                          extra=extra)

    def row_context(self, doc: ImageDocument | None = None) -> dict:
        """Columns appended to every results row: scale and its source."""
        doc = doc or self.doc
        c = doc.calibration if doc else Calibration()
        return {"Scale": c.short(), "Scale source": c.source_label}

    def report_warnings(self, warnings: list[str], analysis: str, provenance: dict | None = None,
                        to_results: bool = False) -> None:
        """Warning rows (amber) in Summary (and Results), amber status line."""
        if not warnings:
            return
        ctx = {"Image": self.doc.name if self.doc else "", "Analysis": analysis}
        self.summary.add_warnings(warnings, ctx, provenance)
        if to_results:
            self.results.add_warnings(warnings, ctx, provenance)
        self.status_label.setStyleSheet("color: #7a4f00;")
        self.status_label.setText("⚠ " + " | ".join(warnings))
        self.statusMessage.emit("Warning: " + " | ".join(warnings))

    def set_preview(self, rgba: np.ndarray | None) -> None:
        """Transient overlay (e.g. red threshold preview) shown above the analysis overlay."""
        self._preview_rgba = rgba
        doc = self.doc
        self.viewer.set_overlay(rgba if rgba is not None else (doc.overlay_rgba if doc else None))

    def set_analysis_overlay(self, shapes: list[dict] | None = None,
                             rgba: np.ndarray | None = None) -> None:
        doc = self.doc
        if doc is None:
            return
        doc.shapes = list(shapes or [])
        doc.overlay_rgba = rgba
        self.viewer.set_shapes(doc.shapes)
        if self._preview_rgba is None:
            self.viewer.set_overlay(rgba)

    def clear_analysis_overlay(self) -> None:
        self._preview_rgba = None
        self.set_analysis_overlay([], None)

    def _scale_bar_spec(self, doc: ImageDocument) -> dict | None:
        sb = doc.scale_bar
        if not sb or not doc.calibration.calibrated:
            return None
        spec = dict(sb)
        spec.update(iops.resolve_scalebar_style(sb, doc.shape, doc.calibration))
        return spec

    def _update_cal_label(self) -> None:
        c = self.cal
        self.cal_label.setText(c.describe())
        self.cal_label.setStyleSheet("" if c.calibrated else "color: #b00020;")
        self.cal_source_label.setText(f"source: {c.source_label}")
        warn = (not c.calibrated) or c.from_print_resolution
        self.cal_source_label.setStyleSheet("color: #7a4f00;" if warn else "color: #2e7d32;")
        if c.calibrated:
            self.disp_unit.setCurrentText(c.unit)

    def _update_status(self) -> None:
        doc = self.doc
        if doc is None:
            self.pixel_label.setText("")
            return
        z = self.viewer.zoom_factor() * 100
        self._status_suffix = f"{doc.shape[1]}×{doc.shape[0]} px · {doc.data.dtype} · {z:.0f} %"
        if not self.pixel_label.text().startswith("x="):
            self.pixel_label.setText(self._status_suffix)

    def _on_mouse_moved(self, x: float, y: float) -> None:
        doc = self.doc
        if doc is None or math.isnan(x):
            self.pixel_label.setText(getattr(self, "_status_suffix", ""))
            return
        c = doc.calibration
        ix, iy = int(x), int(y)
        v = doc.data[iy, ix]
        vs = ", ".join(str(int(t)) if np.issubdtype(doc.data.dtype, np.integer) else f"{t:.4g}"
                       for t in np.atleast_1d(v))
        if c.calibrated:
            pos = f"x={c.length(x):.4g}, y={c.length(y):.4g} {c.unit}"
        else:
            pos = f"x={ix}, y={iy} px"
        extra = ""
        if doc.mask is not None:
            extra = f" · mask={'1' if doc.mask[iy, ix] else '0'}"
        self.pixel_label.setText(f"{pos} · value={vs}{extra} · {self.viewer.zoom_factor() * 100:.0f} %")

    def status(self, msg: str) -> None:
        self.status_label.setStyleSheet("")
        self.status_label.setText(msg)
        self.statusMessage.emit(msg)

    def _show_banner(self, text: str, button: str, action, page: bool = False,
                     secondary: tuple | None = None) -> None:
        self.banner_label.setText(text)
        self.banner_btn.setText(button)
        self.page_spin.setVisible(page)
        self._banner_action = action
        if secondary:
            self.banner_btn2.setText(secondary[0])
            self._banner_action2 = secondary[1]
            self.banner_btn2.show()
        else:
            self._banner_action2 = None
            self.banner_btn2.hide()
        self.banner.show()

    def _queue_banner(self, text, button, action, page=False, secondary=None) -> None:
        self._banner_queue.append((text, button, action, page, secondary))

    def _next_banner(self) -> None:
        if self._banner_queue:
            self._show_banner(*self._banner_queue.pop(0))
        else:
            self.banner.hide()

    @property
    def banner_text(self) -> str:
        return self.banner_label.text() if not self.banner.isHidden() else ""

    def _banner_dismiss(self) -> None:
        self._next_banner()

    def _banner_accept(self) -> None:
        act = self._banner_action
        self._next_banner()
        if act:
            act()

    def _banner_secondary(self) -> None:
        act = self._banner_action2
        self._next_banner()
        if act:
            act()

    # ================================================================= operations + undo
    def apply_op(self, label: str, fn, reset_window: bool = True, keep_mask: bool = False,
                 calibration: Calibration | None = None, then=None) -> bool | None:
        """Apply ``fn(data) -> new_data`` to the current image with undo support.

        The computation runs through ``gui.worker.run_task`` (worker thread + busy dialog in the
        app, synchronous in tests). Returns True/False in synchronous mode, None when the result
        arrives later; ``then()`` is called after a successful application in both cases.
        """
        from labstudio.gui.worker import run_task

        doc = self.require_doc()
        if doc is None:
            return False
        data = doc.data
        state = {"ok": False}

        def done(new):
            if self.doc is not doc:
                return
            self._finish_op(doc, label, new, reset_window, keep_mask, calibration)
            state["ok"] = True
            if then is not None:
                then()

        run_task(self, f"{label}…", lambda: fn(data), on_done=done,
                 on_error=lambda tb, exc: self.status(f"{label} failed: {exc}"))
        from labstudio.gui.worker import _sync_mode

        return state["ok"] if _sync_mode() else None

    def _finish_op(self, doc, label, new, reset_window, keep_mask, calibration) -> None:
        doc.push_undo(label)
        new = np.asarray(new)
        shape_changed = new.shape[:2] != doc.data.shape[:2]
        doc.data = new
        if calibration is not None:
            doc.calibration = calibration
        if shape_changed or not keep_mask:
            doc.mask = None if shape_changed else doc.mask
        if shape_changed:
            doc.rois.clear()
            doc.shapes.clear()
            doc.overlay_rgba = None
            self._preview_rgba = None
            doc.data_bar_row = None
        if reset_window:
            doc.reset_window()
        self.refresh(fit=shape_changed, rebuild_items=shape_changed)
        self.status(f"{label} applied.")
        for p in self.panels.values():
            if hasattr(p, "on_document_changed"):
                p.on_document_changed()

    def set_mask(self, mask: np.ndarray | None, label: str = "Set mask", show: bool = True) -> None:
        doc = self.require_doc()
        if doc is None:
            return
        doc.push_undo(label)
        doc.mask = None if mask is None else np.asarray(mask, bool)
        doc.show_mask = show and mask is not None
        self._preview_rgba = None
        self.refresh()
        self.status(f"{label}: {int(doc.mask.sum()) if doc.mask is not None else 0} foreground px")

    def mask_op(self, label: str, fn) -> bool:
        doc = self.require_doc()
        if doc is None:
            return False
        if doc.mask is None:
            self.status("No binary mask — create one with Threshold ▸ Apply first.")
            return False
        try:
            new = np.asarray(fn(doc.mask), bool)
        except Exception as exc:
            self.status(f"{label} failed: {exc}")
            return False
        doc.push_undo(label)
        doc.mask = new
        doc.show_mask = True
        self.refresh()
        self.status(f"{label}: {int(new.sum())} foreground px")
        return True

    def undo(self) -> None:
        doc = self.doc
        if doc is None or not doc.undo_stack:
            self.status("Nothing to undo.")
            return
        s = doc.undo_stack.pop()
        doc.redo_stack.append(doc.snapshot(s.label))
        doc.restore(s)
        doc.reset_window()
        self.refresh(fit=True, rebuild_items=True)
        self.status(f"Undid: {s.label}")
        self._notify_panels()

    def redo(self) -> None:
        doc = self.doc
        if doc is None or not doc.redo_stack:
            self.status("Nothing to redo.")
            return
        s = doc.redo_stack.pop()
        doc.undo_stack.append(doc.snapshot(s.label))
        doc.restore(s)
        doc.reset_window()
        self.refresh(fit=True, rebuild_items=True)
        self.status(f"Redid: {s.label}")
        self._notify_panels()

    def revert(self) -> None:
        """Revert to the image as originally loaded (undoable)."""
        doc = self.require_doc()
        if doc is None:
            return
        doc.push_undo("Revert")
        doc.restore(Snapshot(doc.original.data.copy(), None, doc.original.calibration, "original"))
        doc.mask = None
        doc.show_mask = False
        doc.reset_window()
        self.refresh(fit=True, rebuild_items=True)
        self.status("Reverted to original.")
        self._notify_panels()

    def _notify_panels(self) -> None:
        for p in self.panels.values():
            if hasattr(p, "on_document_changed"):
                p.on_document_changed()

    def _update_actions(self) -> None:
        d = self.doc
        self._actions["undo"].setEnabled(bool(d and d.undo_stack))
        self._actions["redo"].setEnabled(bool(d and d.redo_stack))
        for k in ("revert", "export", "measure", "fit", "100", "zin", "zout"):
            self._actions[k].setEnabled(d is not None)

    def crop_data_bar(self, row: int | None = None) -> bool:
        doc = self.require_doc()
        if doc is None:
            return False
        if row is None:
            row = iops.detect_data_bar(doc.data)
            if row is None:
                self.status("No SEM data bar detected.")
                return False
        self._banner_queue = [b for b in self._banner_queue if b[1] != "Crop data bar"]
        if self.banner_btn.text() == "Crop data bar":
            self._next_banner()
        return self.apply_op("Crop data bar", lambda d: d[:row].copy(), reset_window=False)

    # ================================================================= calibration
    def set_calibration(self, cal: Calibration, global_: bool | None = None) -> None:
        if global_ is None:
            global_ = self.scale_global.isChecked()
        if global_:
            self.global_calibration = cal if cal.calibrated else None
            for d in self.docs:
                d.calibration = cal
        elif self.doc is not None:
            self.doc.calibration = cal
        self.refresh()
        self._notify_panels()
        self.status(f"Scale: {cal.describe()} (source: {cal.source_label})")

    def set_scale_from_fields(self) -> None:
        try:
            cal = Calibration.from_known_distance(self.scale_px.value(), self.scale_dist.value(),
                                                  self.scale_unit.currentText())
        except ValueError as exc:
            self.status(str(exc))
            return
        self.set_calibration(cal)

    def _change_display_unit(self) -> None:
        doc = self.doc
        if doc is None or not doc.calibration.calibrated:
            return
        doc.calibration = doc.calibration.to_unit(self.disp_unit.currentText())
        self.refresh()

    # ================================================================= tools & ROIs
    def set_tool(self, tool: str) -> None:
        self.viewer.setTool(tool)
        if tool in self._tool_actions:
            self._tool_actions[tool].setChecked(True)
        hints = {"scale": "Draw a line along the scale bar; then type its length and press Set scale.",
                 "polygon": "Click vertices; double-click or Enter to close (Esc cancels).",
                 "polyline": "Click vertices; double-click or Enter to finish.",
                 "point": "Click to add points; double-click or Enter to finish the counter.",
                 "angle": "Click arm end, vertex, other arm end.",
                 "text": "Click where the text should go (text/size/colour in the Annotate panel)."}
        if tool in hints:
            self.status(hints[tool])

    def _on_roi_created(self, tool: str, roi: Roi) -> None:
        self.add_roi(roi)
        if tool == "scale":
            L = iops.polygon_perimeter(np.asarray(roi.points), closed=False)
            self.scale_px.setValue(L)
            self.scale_dist.setFocus()
            self.status(f"Scale line = {L:.2f} px. Enter the known length and press 'Set scale'.")
        elif tool == "profile":
            self.plot_profile(roi)
        elif roi.kind == "point":
            self.status(f"{len(roi.points)} points counted ({roi.name}).")

    def add_roi(self, roi: Roi) -> Roi:
        doc = self.require_doc()
        if doc is None:
            return roi
        doc.rois.append(roi)
        self.viewer.add_roi_item(roi)
        self._rebuild_roi_list(select=roi.id)
        return roi

    def _roi_summary(self, roi: Roi) -> str:
        c = self.cal
        if roi.kind in ("line", "polyline"):
            L = c.length(iops.polygon_perimeter(np.asarray(roi.points), closed=False))
            return f"{roi.name}  L={L:.4g} {c.unit}"
        if roi.kind == "angle" and len(roi.points) == 3:
            p = roi.points
            return f"{roi.name}  {iops.angle_between(p[0], p[1], p[2]):.1f}°"
        if roi.kind == "point":
            return f"{roi.name}  n={len(roi.points)}"
        A = c.area(iops.polygon_area(roi.vertices()))
        return f"{roi.name}  A={A:.4g} {c.area_unit}"

    def _rebuild_roi_list(self, select: int | None = None) -> None:
        doc = self.doc
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        if doc is not None:
            for r in doc.rois:
                it = QListWidgetItem(self._roi_summary(r))
                it.setData(Qt.UserRole, r.id)
                self.roi_list.addItem(it)
        self.roi_list.blockSignals(False)
        if select is not None and doc is not None:
            for i, r in enumerate(doc.rois):
                if r.id == select:
                    self.roi_list.setCurrentRow(i)
        elif doc is not None and doc.rois:
            self.roi_list.setCurrentRow(len(doc.rois) - 1)
        else:
            self._current_roi_id = None

    def _on_roi_selected(self, row: int) -> None:
        doc = self.doc
        if doc is None or not (0 <= row < len(doc.rois)):
            self._current_roi_id = None
            self.viewer.select_roi(None)
            return
        self._current_roi_id = doc.rois[row].id
        self.viewer.select_roi(self._current_roi_id)

    def current_roi(self, kinds: tuple[str, ...] | None = None) -> Roi | None:
        """The selected ROI (optionally only of given kinds; falls back to the latest such ROI)."""
        doc = self.doc
        if doc is None or not doc.rois:
            return None
        sel = next((r for r in doc.rois if r.id == self._current_roi_id), None)
        if sel is not None and (kinds is None or sel.kind in kinds):
            return sel
        if kinds is None:
            return doc.rois[-1]
        for r in reversed(doc.rois):
            if r.kind in kinds:
                return r
        return None

    def current_area_roi(self) -> Roi | None:
        return self.current_roi(("rect", "ellipse", "polygon", "freehand"))

    def roi_mask(self, roi: Roi) -> np.ndarray:
        v = roi.vertices()
        return iops.polygon_mask(self.doc.shape, v[:, 0], v[:, 1])

    def delete_current_roi(self) -> None:
        doc = self.doc
        r = self.current_roi()
        if doc is None or r is None:
            return
        doc.rois = [x for x in doc.rois if x.id != r.id]
        self.viewer.remove_roi_item(r.id)
        self._rebuild_roi_list()

    def clear_rois(self) -> None:
        doc = self.doc
        if doc is None:
            return
        doc.rois.clear()
        self.viewer.clear_rois()
        self._rebuild_roi_list()

    # ================================================================= measurement
    def measure_roi(self, roi: Roi | None = None) -> list[dict]:
        """ImageJ 'Measure' on a ROI (or the whole image if there is none) → Results."""
        doc = self.require_doc()
        if doc is None:
            return []
        roi = roi or self.current_roi()
        c = doc.calibration
        base = {"Image": doc.name}
        rows: list[dict] = []
        if roi is None:
            st = iops.intensity_stats(doc.data)
            h, w = doc.shape
            rows.append({**base, "ROI": "image", "Type": "image", "Area": c.area(h * w),
                         "Mean": st["mean"], "StdDev": st["std"], "Min": st["min"], "Max": st["max"],
                         "Median": st["median"]})
        elif roi.kind in ("line", "polyline"):
            m = iops.measure_line(doc.data, roi.points, c, width=self.line_width.value())
            m.pop("Length (px)", None)
            rows.append({**base, "ROI": roi.name, "Type": roi.kind, **m})
        elif roi.kind == "angle":
            p = roi.points
            ang = iops.angle_between(p[0], p[1], p[2]) if len(p) >= 3 else float("nan")
            rows.append({**base, "ROI": roi.name, "Type": "angle", "Angle (°)": ang,
                         "X": c.length(p[1][0]), "Y": c.length(p[1][1])})
        elif roi.kind == "point":
            g = doc.gray
            for i, (x, y) in enumerate(roi.points, start=1):
                ix = int(min(max(x, 0), doc.shape[1] - 1))
                iy = int(min(max(y, 0), doc.shape[0] - 1))
                rows.append({**base, "ROI": roi.name, "Type": "point", "Counter": i,
                             "X": c.length(x), "Y": c.length(y), "Mean": float(g[iy, ix])})
            self.summary.add_rows([{"Image": doc.name, "Analysis": "Point count", "Quantity":
                                    f"Count ({roi.name})", "Value": len(roi.points), "Unit": ""}])
        else:
            m = iops.measure_polygon(doc.data, roi.vertices(), c)
            m.pop("Area (px²)", None)
            if roi.kind == "ellipse":  # exact ellipse geometry instead of the 180-gon
                (x0, y0), (x1, y1) = roi.points[:2]
                a, b = abs(x1 - x0) / 2, abs(y1 - y0) / 2
                m["Area"] = c.area(math.pi * a * b)
                h = ((a - b) / (a + b)) ** 2 if a + b else 0
                m["Perimeter"] = c.length(math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h))))
            rows.append({**base, "ROI": roi.name, "Type": roi.kind, **m})
        rows = [{**unitize(r, c), **self.row_context(doc)} for r in rows]
        warns = [] if c.calibrated else ["Image is uncalibrated: measurements are in pixels. Set "
                                         "the scale from the scale bar."]
        prov = self.make_provenance("Measure", {"roi": roi.name if roi else "image",
                                                "kind": roi.kind if roi else "image",
                                                "points": roi.points if roi else None,
                                                "line_width_px": self.line_width.value()}, warns, doc)
        self.results.add_rows(rows, provenance=prov)
        if warns:
            self.report_warnings(warns, "Measure", prov, to_results=True)
        self.bottom_tabs.setCurrentWidget(self.results)
        self.status(f"Measured {roi.name if roi else 'image'}.")
        return rows

    def measure_all_rois(self) -> None:
        doc = self.require_doc()
        if doc is None:
            return
        for r in list(doc.rois):
            self.measure_roi(r)

    def plot_profile(self, roi: Roi | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        """Plot the intensity profile along a line ROI (calibrated distance)."""
        doc = self.require_doc()
        if doc is None:
            return None
        roi = roi or self.current_roi(("line", "polyline"))
        if roi is None or roi.kind not in ("line", "polyline"):
            self.status("Draw a line (Line or Profile tool) first.")
            return None
        c = doc.calibration
        xs, ys = [], []
        offset = 0.0
        for a, b in zip(roi.points[:-1], roi.points[1:]):
            d, v = iops.line_profile(doc.data, a, b, width=self.line_width.value(), cal=c)
            xs.append(d + offset)
            ys.append(v)
            offset += float(d[-1]) if len(d) else 0.0
        x, y = np.concatenate(xs), np.concatenate(ys)
        self.plot.plot_line(x, y, f"Distance ({c.unit})", "Grey value",
                            f"Profile {roi.name} — {doc.name} (width {self.line_width.value()} px)")
        self.bottom_tabs.setCurrentWidget(self.plot)
        return x, y

    def add_results(self, rows: list[dict], replace: bool | None = None,
                    provenance: dict | None = None) -> None:
        ctx = self.row_context()
        self.results.add_rows([{**r, **{k: v for k, v in ctx.items() if k not in r}} for r in rows],
                              replace=replace, provenance=provenance)
        self.bottom_tabs.setCurrentWidget(self.results)

    def add_summary(self, rows: list[dict], analysis: str, provenance: dict | None = None) -> None:
        doc = self.doc
        name = doc.name if doc else ""
        self.summary.add_rows([{"Image": name, "Analysis": analysis, **r} for r in rows],
                              provenance=provenance)

    def export_results_csv(self, path: str) -> None:
        """Results table → CSV, starting with provenance comment lines."""
        self.results.export_csv(path)

    def export_summary_csv(self, path: str) -> None:
        self.summary.export_csv(path)

    # ================================================================= annotation
    def _on_viewer_click(self, tool: str, x: float, y: float) -> None:
        if tool == "text":
            self.panels["annotate"].place_text(x, y)

    def _on_arrow(self, x0, y0, x1, y1) -> None:
        self.panels["annotate"].place_arrow((x0, y0), (x1, y1))

    def add_annotation(self, shape: dict) -> None:
        doc = self.require_doc()
        if doc is None:
            return
        doc.annotations.append(shape)
        self.viewer.set_annotations(doc.annotations)

    # ================================================================= export
    def export_dialog(self) -> None:
        doc = self.require_doc()
        if doc is None:
            return
        stem = os.path.splitext(doc.name)[0]
        path, flt = QFileDialog.getSaveFileName(
            self, "Export image", f"{stem}_overlay.png",
            "PNG with overlays (*.png);;TIFF with overlays (*.tif);;"
            "PNG as displayed, no overlays (*.png);;TIFF raw data + calibration, no overlays (*.tif)")
        if path:
            self.export_image(path, with_overlays="with overlays" in flt)

    def export_image(self, path: str, with_overlays: bool = True, include_rois: bool = False) -> None:
        """Save the current image.

        With overlays: the view (display contrast + overlays, annotations, scale bar) rendered at
        native resolution. Without overlays: TIFF keeps the raw data (8/16-bit) with an
        ImageJ-readable calibration; PNG/JPG save the displayed (contrast-adjusted) image.
        """
        doc = self.require_doc()
        if doc is None:
            return
        ext = os.path.splitext(path)[1].lower()
        if with_overlays:
            from labstudio.gui.image_viewer import qimage_to_array

            arr = qimage_to_array(self.viewer.render_image(True, include_rois))
            self._save_array(path, arr, ext, None)
        elif ext in (".tif", ".tiff"):
            self._save_array(path, doc.data, ext, doc.calibration)
        else:
            self._save_array(path, self.display_array(doc), ext, None)
        self.status(f"Exported {path}")

    # ================================================================= scale-bar styles
    def apply_scale_bar_style(self, style: dict, all_images: bool = False) -> int:
        """Apply a scale-bar style (preset) to the current image or to all open images."""
        targets = self.docs if all_images else ([self.doc] if self.doc else [])
        n = 0
        for d in targets:
            d.scale_bar = dict(style)
            n += 1
        self.refresh()
        self._notify_panels()
        return n

    # ================================================================= session
    def get_state(self) -> dict:
        """JSON-serialisable workspace state for a ``.lstudio`` session (arrays go to
        :meth:`get_state_files`)."""
        images = []
        for i, d in enumerate(self.docs):
            images.append({
                "name": d.name, "path": d.path, "page": d.page, "n_pages": d.n_pages,
                "file": f"image_{i}.npz", "calibration": d.calibration.to_dict(),
                "window": [float(d.window[0]), float(d.window[1])],
                "rois": [{"kind": r.kind, "points": [list(p) for p in r.points], "name": r.name}
                         for r in d.rois],
                "has_mask": d.mask is not None, "show_mask": bool(d.show_mask),
                "annotations": _jsonable(d.annotations), "scale_bar": _jsonable(d.scale_bar),
                "data_bar_row": d.data_bar_row, "print_dpi": d.print_dpi,
                "metadata": {k: v for k, v in d.metadata.items() if k != "raw"},
            })
        return {"version": 1, "active": self._current, "images": images,
                "global_calibration": self.global_calibration.to_dict() if self.global_calibration else None,
                "overlays_visible": bool(self._actions["overlays"].isChecked()),
                "results": _jsonable(self.results.rows), "summary": _jsonable(self.summary.rows),
                "results_provenance": _jsonable(self.results.provenance),
                "summary_provenance": _jsonable(self.summary.provenance),
                "line_width": self.line_width.value()}

    def get_state_files(self) -> dict[str, bytes]:
        """Current (processed) pixel data + packed-bit mask of every image as compressed .npz."""
        out = {}
        for i, d in enumerate(self.docs):
            buf = io.BytesIO()
            arrays = {"data": d.data}
            if d.mask is not None:
                arrays["mask_bits"] = np.packbits(d.mask.astype(bool), axis=None)
                arrays["mask_shape"] = np.array(d.mask.shape, np.int64)
            np.savez_compressed(buf, **arrays)
            out[f"image_{i}.npz"] = buf.getvalue()
        return out

    def set_state_files(self, files: dict[str, bytes]) -> None:
        self._pending_files = dict(files or {})

    def set_state(self, state: dict, id_map: dict | None = None) -> None:
        """Restore images, calibrations (+ sources), ROIs, masks, annotations, scale bars and
        results from :meth:`get_state` (+ files given to :meth:`set_state_files`)."""
        from labstudio.gui.image_viewer import Roi

        files = getattr(self, "_pending_files", {}) or {}
        while self.docs:
            self.close_image(len(self.docs) - 1)
        gc = state.get("global_calibration")
        self.global_calibration = Calibration.from_dict(gc) if gc else None
        for e in state.get("images", []):
            data = mask = None
            blob = files.get(e.get("file", ""))
            if blob is not None:
                with np.load(io.BytesIO(blob)) as z:
                    data = z["data"]
                    if "mask_bits" in z:
                        shp = tuple(int(v) for v in z["mask_shape"])
                        mask = np.unpackbits(z["mask_bits"], count=int(np.prod(shp))).reshape(shp).astype(bool)
            elif e.get("path") and os.path.isfile(e["path"]):
                data = iops.load_image(e["path"], e.get("page", 0)).data
            if data is None:
                self.status(f"Session: image data for {e.get('name')} missing — skipped.")
                continue
            original = None
            if e.get("path") and os.path.isfile(e["path"]):
                try:
                    od = iops.load_image(e["path"], e.get("page", 0)).data
                    original = Snapshot(od, None, Calibration.from_dict(e.get("calibration")), "original")
                except Exception:  # noqa: BLE001
                    original = None
            doc = ImageDocument(name=e.get("name", "image"), data=data,
                                calibration=Calibration.from_dict(e.get("calibration")),
                                path=e.get("path", ""), metadata=e.get("metadata") or {},
                                n_pages=e.get("n_pages", 1), page=e.get("page", 0), original=original)
            doc.mask = mask
            doc.show_mask = bool(e.get("show_mask")) and mask is not None
            if e.get("window"):
                doc.window = tuple(float(v) for v in e["window"])
            doc.rois = [Roi(r["kind"], [tuple(p) for p in r["points"]], name=r.get("name", ""))
                        for r in e.get("rois", [])]
            doc.annotations = [_tuplify(a) for a in e.get("annotations", [])]
            doc.scale_bar = e.get("scale_bar")
            doc.data_bar_row = e.get("data_bar_row")
            doc.print_dpi = e.get("print_dpi")
            self._add_doc(doc)
        self.results.clear()
        self.summary.clear()
        self.results.add_rows(state.get("results", []), replace=True)
        self.summary.add_rows(state.get("summary", []), replace=True)
        self.results.provenance = list(state.get("results_provenance", []))
        self.summary.provenance = list(state.get("summary_provenance", []))
        if "line_width" in state:
            self.line_width.setValue(int(state["line_width"]))
        vis = state.get("overlays_visible", True)
        self._actions["overlays"].setChecked(vis)
        self.viewer.set_overlays_visible(vis)
        act = state.get("active", len(self.docs) - 1)
        if self.docs:
            act = min(max(0, int(act)), len(self.docs) - 1)
            self._current = -1
            self.image_list.setCurrentRow(act)
            self.set_current(act)
            self.refresh(fit=True, rebuild_items=True)
        self._pending_files = {}

    @staticmethod
    def _save_array(path: str, arr: np.ndarray, ext: str, cal: Calibration | None) -> None:
        if ext in (".tif", ".tiff"):
            import tifffile

            kw = {}
            if cal is not None and cal.calibrated:
                unit = "micron" if cal.unit == "µm" else cal.unit
                kw = {"imagej": arr.dtype in (np.uint8, np.uint16, np.float32) and arr.ndim == 2,
                      "resolution": (cal.px_per_unit, cal.px_per_unit)}
                if kw["imagej"]:
                    kw["metadata"] = {"unit": unit, "labstudio_cal_source": cal.source_label}
                else:
                    kw.pop("imagej")
            tifffile.imwrite(path, arr, **kw)
        else:
            from PIL import Image

            a = arr if arr.dtype == np.uint8 else iops.to_8bit(arr)
            Image.fromarray(a).save(path)


def _jsonable(v):
    """numpy / tuples → JSON-friendly Python values."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _tuplify(a: dict) -> dict:
    """Annotation dict from JSON: coordinate lists back to tuples."""
    out = dict(a)
    for k in ("pos", "p0", "p1"):
        if k in out and isinstance(out[k], list):
            out[k] = tuple(out[k])
    return out
