"""Right-hand option panels of the Micrographs workspace (one QWidget per QToolBox page).

Every panel exposes plain methods (``run()``, ``apply()``, ...) that the buttons call, so the
functionality can be driven programmatically (tests) without dialogs.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QPushButton,
                               QSlider, QSpinBox, QVBoxLayout, QWidget)

from labstudio.core import image_ops as iops
from labstudio.core import micro_analysis as ma
from labstudio.gui.worker import run_task

COLOURS = ["white", "black", "yellow", "red", "cyan", "lime", "magenta", "orange"]
PHASE_COLOURS = [(30, 120, 255), (255, 60, 60), (40, 200, 80), (255, 200, 0), (180, 80, 255)]


# ------------------------------------------------------------------ small widget helpers

def dspin(value=0.0, lo=0.0, hi=1e6, step=1.0, decimals=2, suffix="", tip="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    if tip:
        s.setToolTip(tip)
    s.setKeyboardTracking(False)
    return s


def ispin(value=1, lo=0, hi=100000, suffix="", tip="") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    if tip:
        s.setToolTip(tip)
    s.setKeyboardTracking(False)
    return s


def button(text, slot, tip="") -> QPushButton:
    b = QPushButton(text)
    b.clicked.connect(lambda: slot())
    if tip:
        b.setToolTip(tip)
    return b


def group(title: str) -> tuple[QGroupBox, QFormLayout]:
    g = QGroupBox(title)
    f = QFormLayout(g)
    f.setContentsMargins(6, 4, 6, 4)
    f.setVerticalSpacing(3)
    f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return g, f


def hrow(*widgets) -> QHBoxLayout:
    h = QHBoxLayout()
    h.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        h.addWidget(w)
    return h


def label_colours(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.integers(60, 256, size=(n + 1, 3)).astype(np.uint8)
    c[0] = 0
    return c


class _Panel(QWidget):
    def __init__(self, ws):
        super().__init__()
        self.ws = ws
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(4, 4, 4, 4)
        self.lay.setSpacing(6)

    def finish(self) -> None:
        self.lay.addStretch(1)

    def on_document_changed(self) -> None:  # pragma: no cover - overridden
        pass


# ======================================================================================
class AdjustPanel(_Panel):
    """Brightness/contrast, display range, gamma, invert, 8-bit, equalisation, CLAHE."""

    STEPS = 1000

    def __init__(self, ws):
        super().__init__(ws)
        g, f = group("Display range (brightness / contrast)")
        self.min_slider = QSlider(Qt.Horizontal)
        self.max_slider = QSlider(Qt.Horizontal)
        self.bri_slider = QSlider(Qt.Horizontal)
        self.con_slider = QSlider(Qt.Horizontal)
        for s in (self.min_slider, self.max_slider):
            s.setRange(0, self.STEPS)
        for s in (self.bri_slider, self.con_slider):
            s.setRange(-100, 100)
        self.min_slider.setToolTip("Lowest grey value displayed as black")
        self.max_slider.setToolTip("Highest grey value displayed as white")
        self.bri_slider.setToolTip("Shift the display window (does not change data until Apply)")
        self.con_slider.setToolTip("Narrow / widen the display window")
        self.min_label, self.max_label = QLabel("0"), QLabel("255")
        for l in (self.min_label, self.max_label):
            l.setMinimumWidth(56)
        f.addRow("Minimum", hrow(self.min_slider, self.min_label))
        f.addRow("Maximum", hrow(self.max_slider, self.max_label))
        f.addRow("Brightness", self.bri_slider)
        f.addRow("Contrast", self.con_slider)
        self.saturated = dspin(0.35, 0, 20, 0.05, 2, " %", "Auto: percentage of pixels saturated "
                                                              "(half at each end), as in ImageJ")
        f.addRow("Saturated", self.saturated)
        f.addRow(hrow(button("Auto", self.auto, "Auto contrast (saturated %)"),
                      button("Reset", self.reset, "Full data range"),
                      button("Apply", self.apply_lut, "Bake the display range into the data "
                                                       "(converts to 8-bit)")))
        self.min_slider.valueChanged.connect(self._minmax_changed)
        self.max_slider.valueChanged.connect(self._minmax_changed)
        self.bri_slider.valueChanged.connect(self._bc_changed)
        self.con_slider.valueChanged.connect(self._bc_changed)
        self.lay.addWidget(g)

        g2, f2 = group("Intensity transforms")
        self.gamma = dspin(1.0, 0.05, 5, 0.05, 2, tip="out = max·(in/max)^γ")
        f2.addRow("Gamma γ", hrow(self.gamma, button("Apply", self.apply_gamma)))
        f2.addRow(hrow(button("Invert", lambda: self.ws.apply_op("Invert", iops.invert)),
                       button("8-bit", lambda: self.ws.apply_op("8-bit", lambda d: iops.to_8bit(
                           d, *self.ws.doc.window)), "Convert to 8-bit grey using the display range"),
                       button("Equalize", lambda: self.ws.apply_op("Equalize histogram",
                                                                   iops.equalize_histogram))))
        self.clahe_clip = dspin(0.01, 0.001, 1, 0.005, 3, tip="CLAHE clip limit (higher = more contrast)")
        self.clahe_tile = ispin(64, 8, 2048, " px", "CLAHE block size")
        f2.addRow("CLAHE clip", self.clahe_clip)
        f2.addRow("CLAHE block", hrow(self.clahe_tile, button("Apply", self.apply_clahe)))
        self.lay.addWidget(g2)
        self.finish()
        self._busy = False

    def _range(self):
        d = self.ws.doc
        if d is None:
            return 0.0, 255.0
        if d.data.dtype == np.uint8:
            return 0.0, 255.0
        g = d.gray
        lo, hi = float(np.min(g)), float(np.max(g))
        return lo, hi if hi > lo else lo + 1

    def _to_val(self, s):
        lo, hi = self._range()
        return lo + (hi - lo) * s / self.STEPS

    def _to_slider(self, v):
        lo, hi = self._range()
        return int(round((v - lo) / (hi - lo) * self.STEPS))

    def on_document_changed(self) -> None:
        d = self.ws.doc
        if d is None:
            return
        self._busy = True
        self.min_slider.setValue(self._to_slider(d.window[0]))
        self.max_slider.setValue(self._to_slider(d.window[1]))
        self.bri_slider.setValue(0)
        self.con_slider.setValue(0)
        self._busy = False
        self._labels()

    def _labels(self):
        d = self.ws.doc
        if d:
            self.min_label.setText(f"{d.window[0]:.5g}")
            self.max_label.setText(f"{d.window[1]:.5g}")

    def set_window(self, lo: float, hi: float) -> None:
        d = self.ws.require_doc()
        if d is None:
            return
        d.window = (float(lo), float(max(hi, lo + 1e-9)))
        self.ws.refresh()
        self.on_document_changed()

    def _minmax_changed(self):
        if self._busy or self.ws.doc is None:
            return
        lo, hi = self._to_val(self.min_slider.value()), self._to_val(self.max_slider.value())
        if hi <= lo:
            hi = lo + (self._range()[1] - self._range()[0]) / self.STEPS
        self.ws.doc.window = (lo, hi)
        self.ws.refresh()
        self._labels()

    def _bc_changed(self):
        if self._busy or self.ws.doc is None:
            return
        lo0, hi0 = self._range()
        lo, hi = iops.brightness_contrast_window(self.bri_slider.value() / 100,
                                                 self.con_slider.value() / 100, lo0, hi0)
        self.ws.doc.window = (lo, hi)
        self._busy = True
        self.min_slider.setValue(self._to_slider(lo))
        self.max_slider.setValue(self._to_slider(hi))
        self._busy = False
        self.ws.refresh()
        self._labels()

    def auto(self):
        d = self.ws.require_doc()
        if d:
            self.set_window(*iops.auto_contrast_limits(d.data, self.saturated.value()))

    def reset(self):
        d = self.ws.require_doc()
        if d:
            d.reset_window()
            self.ws.refresh()
            self.on_document_changed()

    def apply_lut(self):
        d = self.ws.require_doc()
        if d:
            lo, hi = d.window
            self.ws.apply_op("Apply LUT", lambda a: iops.apply_window(a, lo, hi))

    def apply_gamma(self):
        g = self.gamma.value()
        self.ws.apply_op(f"Gamma {g:g}", lambda a: iops.gamma_correct(a, g))

    def apply_clahe(self):
        c, t = self.clahe_clip.value(), self.clahe_tile.value()
        self.ws.apply_op("CLAHE", lambda a: iops.clahe(a, c, t))


# ======================================================================================
class ProcessPanel(_Panel):
    """Filters, FFT band-pass, rolling-ball background, crop/rotate/flip/resize."""

    def __init__(self, ws):
        super().__init__(ws)
        g, f = group("Filters")
        self.sigma = dspin(2.0, 0.1, 200, 0.5, 2, " px", "Gaussian standard deviation")
        f.addRow("Gaussian σ", hrow(self.sigma, button("Apply", lambda: self._op(
            f"Gaussian σ={self.sigma.value():g}", iops.gaussian_blur, self.sigma.value()))))
        self.median_r = dspin(2.0, 0.5, 50, 0.5, 1, " px", "Median filter radius (circular kernel); "
                                                             "removes salt-and-pepper / SEM shot noise")
        f.addRow("Median r", hrow(self.median_r, button("Apply", lambda: self._op(
            f"Median r={self.median_r.value():g}", iops.median_filter, self.median_r.value()))))
        self.mean_r = dspin(2.0, 0.5, 50, 0.5, 1, " px", "Mean filter radius (circular kernel)")
        f.addRow("Mean r", hrow(self.mean_r, button("Apply", lambda: self._op(
            f"Mean r={self.mean_r.value():g}", iops.mean_filter, self.mean_r.value()))))
        self.us_r = dspin(2.0, 0.1, 100, 0.5, 1, " px", "Unsharp mask Gaussian radius")
        self.us_w = dspin(0.6, 0.1, 0.9, 0.05, 2, tip="Unsharp mask weight (0.1–0.9)")
        f.addRow("Unsharp", hrow(self.us_r, self.us_w, button("Apply", lambda: self._op(
            "Unsharp mask", iops.unsharp_mask, self.us_r.value(), self.us_w.value()))))
        f.addRow(hrow(button("Sharpen", lambda: self._op("Sharpen", iops.sharpen)),
                      button("Find edges", lambda: self._op("Find edges", iops.find_edges),
                             "Sobel gradient magnitude")))
        self.lay.addWidget(g)

        g, f = group("FFT band-pass")
        self.bp_large = dspin(40, 0, 10000, 5, 1, " px", "Filter large structures down to (px) — removes "
                                                       "shading. 0 = off")
        self.bp_small = dspin(3, 0, 1000, 1, 1, " px", "Filter small structures up to (px) — removes noise. "
                                                    "0 = off")
        f.addRow("Large", self.bp_large)
        f.addRow("Small", hrow(self.bp_small, button("Apply", lambda: self._op(
            "FFT band-pass", iops.fft_bandpass, self.bp_large.value(), self.bp_small.value()))))
        self.lay.addWidget(g)

        g, f = group("Subtract background (rolling ball)")
        self.ball = dspin(50, 1, 2000, 5, 1, " px", "Rolling-ball radius: at least the size of the largest "
                                                  "feature that is not background")
        self.light_bg = QCheckBox("Light background")
        self.light_bg.setToolTip("Tick for bright-field LOM (features darker than background)")
        f.addRow("Radius", self.ball)
        f.addRow(hrow(self.light_bg, button("Subtract", self.subtract_background),
                      button("Show bg", self.show_background, "Open the background estimate as a new image")))
        self.lay.addWidget(g)

        g, f = group("Geometry")
        f.addRow(hrow(button("Crop to ROI", self.crop_to_roi, "Crop to the bounding box of the area ROI"),
                      button("Crop data bar", lambda: self.ws.crop_data_bar(),
                             "Detect and remove the SEM info bar at the bottom")))
        f.addRow(hrow(button("Rotate 90° L", lambda: self.ws.apply_op("Rotate 90° left",
                                                                      lambda d: iops.rotate90(d, 1))),
                      button("Rotate 90° R", lambda: self.ws.apply_op("Rotate 90° right",
                                                                      lambda d: iops.rotate90(d, -1)))))
        self.angle = dspin(0, -360, 360, 0.5, 2, "°", "Counter-clockwise angle")
        self.enlarge = QCheckBox("Enlarge")
        self.enlarge.setToolTip("Enlarge canvas so nothing is clipped")
        f.addRow("Rotate", hrow(self.angle, self.enlarge, button("Apply", self.rotate)))
        f.addRow(hrow(button("Flip horizontal", lambda: self.ws.apply_op("Flip H", lambda d: iops.flip(d, True))),
                      button("Flip vertical", lambda: self.ws.apply_op("Flip V", lambda d: iops.flip(d, False)))))
        self.rs_w = ispin(512, 1, 100000, " px")
        self.rs_h = ispin(512, 1, 100000, " px")
        self.rs_keep = QCheckBox("Keep aspect")
        self.rs_keep.setChecked(True)
        self.rs_w.valueChanged.connect(self._rs_w_changed)
        f.addRow("Size W×H", hrow(self.rs_w, self.rs_h))
        f.addRow(hrow(self.rs_keep, button("Resize", self.resize,
                                           "Resample (calibration updated accordingly)")))
        self.lay.addWidget(g)
        self.finish()

    def on_document_changed(self) -> None:
        d = self.ws.doc
        if d is None:
            return
        self.rs_w.blockSignals(True)
        self.rs_w.setValue(d.shape[1])
        self.rs_h.setValue(d.shape[0])
        self.rs_w.blockSignals(False)

    def _rs_w_changed(self, w):
        d = self.ws.doc
        if d is not None and self.rs_keep.isChecked():
            self.rs_h.setValue(max(1, int(round(w * d.shape[0] / d.shape[1]))))

    def _op(self, label, fn, *args):
        self.ws.apply_op(label, lambda d: fn(d, *args), reset_window=False)

    def subtract_background(self):
        r, light = self.ball.value(), self.light_bg.isChecked()
        self.ws.apply_op(f"Subtract background r={r:g}",
                         lambda d: iops.subtract_background(d, r, light))

    def show_background(self):
        d = self.ws.require_doc()
        if d is None:
            return
        data, r, light = d.data, self.ball.value(), self.light_bg.isChecked()
        run_task(self.ws, "Estimating background…",
                 lambda: np.abs(iops.subtract_background(data, r, light, return_background=True)[1]
                                ).astype(np.float32),
                 on_done=lambda bg: self.ws.add_image(bg, f"Background of {d.name}", d.calibration),
                 on_error=lambda tb, e: self.ws.status(f"Background failed: {e}"))

    def crop_to_roi(self):
        d = self.ws.require_doc()
        if d is None:
            return
        roi = self.ws.current_area_roi()
        if roi is None:
            self.ws.status("Draw a rectangle (or any area ROI) to crop to.")
            return
        x0, y0, x1, y1 = roi.bounds()
        self.ws.apply_op("Crop", lambda a: iops.crop(a, math.floor(x0), math.floor(y0),
                                                     math.ceil(x1), math.ceil(y1)), reset_window=False)

    def rotate(self):
        a, e = self.angle.value(), self.enlarge.isChecked()
        self.ws.apply_op(f"Rotate {a:g}°", lambda d: iops.rotate(d, a, e), reset_window=False)

    def resize(self):
        d = self.ws.require_doc()
        if d is None:
            return
        w, h = self.rs_w.value(), self.rs_h.value()
        f = w / d.shape[1]
        cal = d.calibration
        new_cal = iops.Calibration(cal.px_per_unit * f, cal.unit, cal.source) if cal.calibrated else cal
        self.ws.apply_op(f"Resize {w}×{h}", lambda a: iops.resize(a, w, h), reset_window=False,
                         calibration=new_cal)


# ======================================================================================
class ThresholdPanel(_Panel):
    """Manual / automatic threshold with red preview → binary mask; binary morphology."""

    STEPS = 1000

    def __init__(self, ws):
        super().__init__(ws)
        g, f = group("Threshold")
        self.method = QComboBox()
        self.method.addItems(["Manual"] + iops.THRESHOLD_METHODS)
        self.method.setCurrentText("Otsu")
        self.method.setToolTip("Automatic method (skimage); MaxEntropy = Kapur maximum entropy")
        self.dark = QCheckBox("Objects are dark")
        self.dark.setChecked(True)
        self.dark.setToolTip("Select pixels darker than the threshold (e.g. pores, boundaries, "
                             "precipitates darker than matrix)")
        f.addRow("Method", hrow(self.method, button("Auto", self.auto)))
        f.addRow(self.dark)
        self.lo_slider = QSlider(Qt.Horizontal)
        self.hi_slider = QSlider(Qt.Horizontal)
        for s in (self.lo_slider, self.hi_slider):
            s.setRange(0, self.STEPS)
            s.valueChanged.connect(self._sliders_changed)
        self.lo_label, self.hi_label = QLabel("–"), QLabel("–")
        for l in (self.lo_label, self.hi_label):
            l.setMinimumWidth(56)
        f.addRow("Lower", hrow(self.lo_slider, self.lo_label))
        f.addRow("Upper", hrow(self.hi_slider, self.hi_label))
        self.preview = QCheckBox("Red preview")
        self.preview.setChecked(True)
        self.preview.toggled.connect(lambda _: self.update_preview())
        self.show_mask = QCheckBox("Show mask")
        self.show_mask.setToolTip("Display the binary mask instead of the image")
        self.show_mask.toggled.connect(self._toggle_mask)
        f.addRow(hrow(self.preview, self.show_mask))
        f.addRow(hrow(button("Apply → mask", self.apply, "Create the binary mask from the selected range"),
                      button("Clear mask", lambda: self.ws.set_mask(None, "Clear mask"))))
        self.info = QLabel("")
        self.info.setWordWrap(True)
        f.addRow(self.info)
        self.lay.addWidget(g)

        g, f = group("Binary (operates on the mask)")
        self.iters = ispin(1, 1, 100, tip="Iterations")
        self.radius = ispin(1, 1, 50, " px", "Structuring element radius (1 = 3×3)")
        f.addRow("Iterations / radius", hrow(self.iters, self.radius))
        grid = QGridLayout()
        ops = [("Erode", lambda m: iops.binary_erode(m, self.iters.value(), self.radius.value())),
               ("Dilate", lambda m: iops.binary_dilate(m, self.iters.value(), self.radius.value())),
               ("Open", lambda m: iops.binary_open(m, self.iters.value(), self.radius.value())),
               ("Close", lambda m: iops.binary_close(m, self.iters.value(), self.radius.value())),
               ("Fill holes", iops.fill_holes),
               ("Outline", iops.outline),
               ("Skeletonize", iops.skeletonize),
               ("Invert mask", lambda m: ~m)]
        for i, (name, fn) in enumerate(ops):
            grid.addWidget(button(name, lambda n=name, fn=fn: self.ws.mask_op(n, fn)), i // 3, i % 3)
        f.addRow(grid)
        self.min_px = ispin(20, 1, 10 ** 8, " px", "Remove objects smaller than this (pixels)")
        f.addRow("Remove small", hrow(self.min_px, button("Apply", lambda: self.ws.mask_op(
            "Remove small", lambda m: iops.remove_small(m, self.min_px.value())))))
        self.ws_dist = ispin(5, 1, 500, " px", "Minimum distance between particle centres for watershed seeds")
        f.addRow("Watershed", hrow(self.ws_dist, button("Separate", lambda: self.ws.mask_op(
            "Watershed", lambda m: iops.watershed_separate(m, self.ws_dist.value())),
            "Split touching particles (distance-transform watershed)")))
        self.lay.addWidget(g)
        self.finish()
        self._range_cache = (0.0, 255.0)
        self._busy = False

    # -- helpers
    def _range(self):
        d = self.ws.doc
        if d is None:
            return 0.0, 255.0
        if d.data.dtype == np.uint8:
            return 0.0, 255.0
        g = d.gray
        lo, hi = float(np.min(g)), float(np.max(g))
        return lo, hi if hi > lo else lo + 1

    def values(self) -> tuple[float, float]:
        lo, hi = self._range()
        a = lo + (hi - lo) * self.lo_slider.value() / self.STEPS
        b = lo + (hi - lo) * self.hi_slider.value() / self.STEPS
        return min(a, b), max(a, b)

    def set_values(self, a: float, b: float) -> None:
        lo, hi = self._range()
        self._busy = True
        self.lo_slider.setValue(int(round((a - lo) / (hi - lo) * self.STEPS)))
        self.hi_slider.setValue(int(round((b - lo) / (hi - lo) * self.STEPS)))
        self._busy = False
        self._sliders_changed()

    def on_document_changed(self) -> None:
        d = self.ws.doc
        if d is None:
            return
        self.show_mask.blockSignals(True)
        self.show_mask.setChecked(d.show_mask)
        self.show_mask.blockSignals(False)
        if self.lo_slider.value() == self.hi_slider.value():  # never set → start from Otsu
            try:
                lo, hi, _ = iops.auto_threshold(d.data, "Otsu", self.dark.isChecked())
            except ValueError:
                lo, hi = self._range()
            self.set_values(lo, hi)

    def _toggle_mask(self, on: bool) -> None:
        d = self.ws.doc
        if d is None:
            return
        d.show_mask = on and d.mask is not None
        self.ws.refresh()

    def _sliders_changed(self):
        if self._busy:
            return
        a, b = self.values()
        self.lo_label.setText(f"{a:.5g}")
        self.hi_label.setText(f"{b:.5g}")
        if self.method.currentText() != "Manual" and self.sender() in (self.lo_slider, self.hi_slider):
            self.method.setCurrentText("Manual")
        self.update_preview()

    def current_mask(self) -> np.ndarray | None:
        d = self.ws.doc
        if d is None:
            return None
        a, b = self.values()
        return iops.threshold_range(d.data, a, b)

    def update_preview(self) -> None:
        d = self.ws.doc
        if d is None:
            return
        if not self.preview.isChecked() or not self.isVisible():
            self.ws.set_preview(None)
            return
        a, b = self.values()
        m = iops.threshold_range(self.ws.overlay_view(d.data), a, b)
        rgba = np.zeros(m.shape + (4,), np.uint8)
        rgba[m] = (255, 0, 0, 150)
        self.ws.set_preview(rgba)
        frac = 100.0 * m.mean()
        self.info.setText(f"Selected: {frac:.2f} % of the image")

    def showEvent(self, ev):  # noqa: N802
        super().showEvent(ev)
        self.update_preview()

    def hideEvent(self, ev):  # noqa: N802
        super().hideEvent(ev)
        if self.ws.doc is not None:
            self.ws.set_preview(None)

    # -- actions
    def auto(self, method: str | None = None) -> tuple[float, float] | None:
        d = self.ws.require_doc()
        if d is None:
            return None
        m = method or self.method.currentText()
        if m == "Manual":
            m = "Otsu"
        try:
            lo, hi, _ = iops.auto_threshold(d.data, m, self.dark.isChecked())
        except ValueError as exc:
            self.ws.status(str(exc))
            return None
        self.method.blockSignals(True)
        self.method.setCurrentText(m)
        self.method.blockSignals(False)
        self.set_values(lo, hi)
        self.ws.status(f"{m} threshold: {hi if self.dark.isChecked() else lo:.5g}")
        return lo, hi

    def apply(self) -> np.ndarray | None:
        m = self.current_mask()
        if m is None:
            return None
        a, b = self.values()
        self.ws.set_mask(m, f"Threshold {a:.4g}–{b:.4g}")
        self.show_mask.blockSignals(True)
        self.show_mask.setChecked(True)
        self.show_mask.blockSignals(False)
        return m


# ======================================================================================
class ParticlesPanel(_Panel):
    """ImageJ 'Analyze Particles' on the binary mask."""

    def __init__(self, ws):
        super().__init__(ws)
        g, f = group("Analyze particles (mask foreground)")
        self.min_area = dspin(0, 0, 1e12, 1, 4, tip="Minimum particle area in calibrated unit²")
        self.max_area = dspin(0, 0, 1e12, 1, 4, tip="Maximum particle area in unit² (0 = infinity)")
        self.area_unit = QLabel("px²")
        f.addRow("Size min", hrow(self.min_area, self.area_unit))
        f.addRow("Size max", hrow(self.max_area, QLabel("0 = ∞")))
        self.min_circ = dspin(0.0, 0, 1, 0.05, 2, tip="Circularity 4π·area/perimeter²")
        self.max_circ = dspin(1.0, 0, 1, 0.05, 2)
        f.addRow("Circularity", hrow(self.min_circ, QLabel("–"), self.max_circ))
        self.perim = QComboBox()
        self.perim.addItems(list(ma.PERIMETER_METHODS))
        self.perim.setToolTip("ImageJ (traced polygon): same perimeter and circularity as ImageJ "
                              "'Analyze Particles' (large discs ≈ 0.90) — use to compare with "
                              "literature.\nCrofton (unbiased): closer to the true perimeter of "
                              "smooth outlines (discs ≈ 1.0), but differs from ImageJ.")
        f.addRow("Perimeter", self.perim)
        self.excl = QCheckBox("Exclude on edges")
        self.excl.setChecked(True)
        self.holes = QCheckBox("Include holes")
        self.holes.setToolTip("Interior holes count as particle area")
        f.addRow(hrow(self.excl, self.holes))
        self.excl_labels = QCheckBox("Exclude scale bar / labels")
        self.excl_labels.setChecked(True)
        self.excl_labels.setToolTip("Auto-detected burned-in scale bars, label boxes/text near the "
                                    "corners and the SEM data bar are left out of the analysis "
                                    "(and of the analysed area)")
        f.addRow(hrow(self.excl_labels, button("Preview", self.preview_excluded,
                                               "Show the excluded regions on the image")))
        self.outlines = QCheckBox("Outlines")
        self.outlines.setChecked(True)
        self.numbers = QCheckBox("Labels")
        self.fill = QCheckBox("Colour fill")
        f.addRow(hrow(self.outlines, self.numbers, self.fill))
        self.bins = ispin(20, 2, 500, tip="Bins of the size-distribution histogram")
        self.dist_metric = QComboBox()
        self.dist_metric.addItems(["ECD", "Area", "Feret", "AR", "Circ."])
        f.addRow("Histogram", hrow(self.dist_metric, self.bins))
        f.addRow(button("Analyze", self.run, "Measure every particle; results → Results table, "
                                              "summary → Summary tab"))
        self.info = QLabel("")
        self.info.setWordWrap(True)
        f.addRow(self.info)
        self.lay.addWidget(g)
        self.finish()
        self.last: ma.ParticleResult | None = None

    def on_document_changed(self) -> None:
        self.area_unit.setText(self.ws.cal.area_unit)

    def excluded_boxes(self) -> list:
        d = self.ws.doc
        if d is None or not self.excl_labels.isChecked():
            return []
        return d.exclusion_boxes()

    def preview_excluded(self) -> list:
        boxes = self.excluded_boxes() if self.excl_labels.isChecked() else (
            self.ws.doc.exclusion_boxes() if self.ws.doc else [])
        self.ws.set_analysis_overlay(excluded_shapes(boxes), None)
        self.ws.status(f"{len(boxes)} region(s) excluded (scale bar / labels / data bar)" if boxes
                       else "No burned-in scale bar, labels or data bar detected.")
        return boxes

    def params(self) -> dict:
        return {"min_area": self.min_area.value(), "max_area": self.max_area.value() or math.inf,
                "min_circ": self.min_circ.value(), "max_circ": self.max_circ.value(),
                "exclude_edges": self.excl.isChecked(), "include_holes": self.holes.isChecked(),
                "perimeter": ma.PERIMETER_METHODS[self.perim.currentText()]}

    def run(self) -> ma.ParticleResult | None:
        d = self.ws.require_doc()
        if d is None:
            return None
        if d.mask is None:
            self.ws.status("No binary mask — use Threshold ▸ Apply first.")
            return None
        c = d.calibration
        prm = self.params()
        boxes = self.excluded_boxes()
        mask, data = d.mask, d.data

        def compute():
            excl = iops.regions_mask(mask.shape, boxes) if boxes else None
            return ma.analyze_particles(mask, c, intensity=data, exclude_regions=excl, **prm)

        return run_task(self.ws, "Analysing particles…", compute,
                        on_done=lambda res: self._show(d, res, boxes),
                        on_error=lambda tb, e: self.ws.status(f"Particle analysis failed: {e}"))

    def _show(self, d, res: ma.ParticleResult, boxes) -> None:
        self.last = res
        c = d.calibration
        from labstudio.gui.image_workspace import unitize

        prm = dict(res.params)
        prm["excluded_regions"] = [list(b) for b in boxes]
        prov = self.ws.make_provenance("Analyze particles", prm, res.warnings, d)
        rows = [{"Image": d.name, "Analysis": "Particles", **unitize(r, c)} for r in res.rows]
        self.ws.add_results(rows, provenance=prov)
        u, a = c.unit, c.area_unit
        s = res.summary
        self.ws.add_summary([
            {"Quantity": "Count", "Value": s["Count"], "Unit": ""},
            {"Quantity": "Total area", "Value": s["Total area"], "Unit": a},
            {"Quantity": "Average size", "Value": s["Average size"], "Unit": a},
            {"Quantity": "Area fraction", "Value": s["Area fraction (%)"], "Unit": "%"},
            {"Quantity": "Mean ECD", "Value": s["Mean ECD"], "Unit": u},
            {"Quantity": "Mean circularity", "Value": s["Mean circularity"], "Unit": ""},
            {"Quantity": "Number density", "Value": s["Number density (1/unit²)"], "Unit": f"1/{a}"},
            {"Quantity": "Analysed area", "Value": s["Analysed area"], "Unit": a},
            {"Quantity": "Excluded regions (scale bar / labels)", "Value": len(boxes), "Unit": ""},
        ], "Analyze particles", provenance=prov)
        shapes = []
        if self.outlines.isChecked():
            shapes.append({"kind": "paths", "paths": res.contours, "color": "#00e5ff", "width": 1.2})
        if self.numbers.isChecked():
            fs = max(8, int(min(d.shape) * 0.018))
            for r in res.rows:
                shapes.append({"kind": "text", "text": str(r["Label"]), "color": "yellow", "size": fs,
                               "pos": (c.to_px(r["X"]), c.to_px(r["Y"]))})
        if boxes:
            shapes += excluded_shapes(boxes)
        rgba = None
        if self.fill.isChecked() and res.count:
            col = label_colours(res.count)
            lab = self.ws.overlay_view(res.labels)
            rgba = np.zeros(lab.shape + (4,), np.uint8)
            rgba[..., :3] = col[lab]
            rgba[..., 3] = np.where(lab > 0, 120, 0)
        self.ws.set_analysis_overlay(shapes, rgba)
        d.show_mask = False
        self.ws.refresh()
        metric = self.dist_metric.currentText()
        if res.count:
            vals = res.column(metric)
            vals = vals[np.isfinite(vals)]
            counts, edges = np.histogram(vals, bins=self.bins.value())
            unit = {"ECD": u, "Area": a, "Feret": u}.get(metric, "")
            self.ws.plot.plot_hist(counts, edges, f"{metric} ({unit})" if unit else metric, "Count",
                                   f"Particle size distribution — {d.name} (n={res.count})",
                                   color="#1f77b4")
        from labstudio.gui.app_settings import format_value as fv

        self.info.setText(f"{s['Count']} particles, area fraction {fv(s['Area fraction (%)'])} %, "
                          f"mean ECD {fv(s['Mean ECD'])} {u}"
                          + (f" · {len(boxes)} region(s) excluded" if boxes else ""))
        self.ws.status(self.info.text())
        self.ws.report_warnings(res.warnings, "Analyze particles", prov)
        self.ws.bottom_tabs.setCurrentWidget(self.ws.results)


def excluded_shapes(boxes) -> list[dict]:
    """Overlay shapes (red dashed, hatched) for excluded regions."""
    return [{"kind": "rect", "rect": tuple(float(v) for v in b), "color": "#ff1744", "width": 1.5,
             "dashed": True, "fill": "#ff1744", "alpha": 140, "hatch": True} for b in boxes]


# ======================================================================================
class GrainSizePanel(_Panel):
    """ASTM E112 linear intercept, Jeffries planimetric and grain-area distribution."""

    SOURCES = ["Auto: dark boundaries (etched)", "Auto: bright boundaries",
               "Mask = grain boundaries", "Mask = grains"]
    PATTERNS = {"Grid (horizontal + vertical)": "grid", "Horizontal lines": "horizontal",
                "Vertical lines": "vertical", "Three concentric circles": "circles"}

    def __init__(self, ws):
        super().__init__(ws)
        g, f = group("Grain boundaries")
        self.source = QComboBox()
        self.source.addItems(self.SOURCES)
        self.source.setToolTip("Where the boundary map comes from. 'Auto' enhances thin dark/bright "
                               "lines with a top-hat and thresholds them.")
        f.addRow("Source", self.source)
        self.sigma = dspin(0.5, 0, 20, 0.5, 1, " px", "Gaussian smoothing before boundary detection")
        self.tophat = dspin(6, 0, 200, 1, 0, " px", "Top-hat radius (> boundary width; removes grain "
                                                    "shading). 0 = threshold raw intensity")
        f.addRow("Smoothing / top-hat", hrow(self.sigma, self.tophat))
        self.method = QComboBox()
        self.method.addItems(iops.THRESHOLD_METHODS)
        self.speck = ispin(20, 0, 100000, " px", "Remove boundary fragments smaller than this")
        f.addRow("Threshold / specks", hrow(self.method, self.speck))
        self.thin = QCheckBox("Thin boundaries to 1 px")
        self.thin.setChecked(True)
        self.thin.setToolTip("Skeletonise so boundary width does not shorten intercepts")
        self.min_grain = ispin(10, 1, 10 ** 7, " px", "Regions smaller than this are not counted as grains")
        f.addRow(self.thin)
        f.addRow("Min grain", self.min_grain)
        f.addRow(button("Preview boundaries → mask", self.preview_boundaries,
                        "Put the detected boundary map into the binary mask for checking/editing"))
        self.lay.addWidget(g)

        g, f = group("Linear intercept (ASTM E112 / ISO 643)")
        self.pattern = QComboBox()
        self.pattern.addItems(list(self.PATTERNS))
        self.n_lines = ispin(8, 1, 200, tip="Number of lines per direction")
        f.addRow("Test lines", hrow(self.pattern, self.n_lines))
        self.triple = QCheckBox("Triple-junction crossing counts 1.5")
        self.triple.setToolTip("E112 convention for a test line passing exactly through a triple point")
        f.addRow(self.triple)
        f.addRow(hrow(button("Intercept", self.run_intercept), button("Planimetric", self.run_planimetric),
                      button("Distribution", self.run_distribution)))
        f.addRow(button("Run all three", self.run_all))
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        f.addRow(self.info)
        self.lay.addWidget(g)
        self.finish()
        self.last_intercept: ma.InterceptResult | None = None
        self.last_planimetric: ma.PlanimetricResult | None = None
        self.last_distribution: ma.GrainDistribution | None = None

    # ---------------------------------------------------------------- computation (pure)
    def _settings(self) -> dict:
        return {"source": self.source.currentText(), "sigma": self.sigma.value(),
                "tophat_radius": self.tophat.value(), "threshold": self.method.currentText(),
                "speck_px": self.speck.value(), "thin": self.thin.isChecked(),
                "min_grain_px": self.min_grain.value(), "n_lines": self.n_lines.value(),
                "pattern": self.PATTERNS[self.pattern.currentText()],
                "triple_points": self.triple.isChecked()}

    def _field_rows(self, d) -> int | None:
        """Rows analysed: the SEM data bar (if detected and not cropped) is excluded."""
        r = d.data_bar_row
        return int(r) if r is not None and 0 < r < d.shape[0] else None

    def _boundary_fn(self, d):
        """Pure function computing the (unthinned) boundary map of the analysed field, or None
        (with a status message) if impossible."""
        src = self.source.currentIndex()
        rows = self._field_rows(d)
        st = self._settings()
        if src >= 2:
            if d.mask is None:
                self.ws.status("No binary mask — threshold first or choose an 'Auto' source.")
                return None
            m = d.mask if rows is None else d.mask[:rows]
            return (lambda: m.copy()) if src == 2 else (lambda: ~m)
        data = d.data if rows is None else d.data[:rows]
        return lambda: ma.boundaries_from_image(data, dark_boundaries=(src == 0),
                                                method=st["threshold"], sigma=st["sigma"],
                                                tophat_radius=st["tophat_radius"], thin=False,
                                                min_size=st["speck_px"])

    def boundary_map(self) -> np.ndarray | None:
        d = self.ws.require_doc()
        if d is None:
            return None
        fn = self._boundary_fn(d)
        if fn is None:
            return None
        try:
            return fn()
        except ValueError as exc:
            self.ws.status(f"Boundary detection failed: {exc}")
            return None

    def preview_boundaries(self):
        d = self.ws.require_doc()
        fn = self._boundary_fn(d) if d is not None else None
        if fn is None:
            return
        thin = self.thin.isChecked()
        h, w = d.shape

        def compute():
            b = fn()
            b = ma.thin_boundaries(b) if thin else b
            if b.shape != (h, w):
                full = np.zeros((h, w), bool)
                full[:b.shape[0]] = b
                b = full
            return b

        run_task(self.ws, "Detecting grain boundaries…", compute,
                 on_done=lambda b: self.ws.set_mask(b, "Grain boundaries"),
                 on_error=lambda tb, e: self.ws.status(f"Boundary detection failed: {e}"))

    def _g(self, G):
        return "n/a (calibrate first)" if G is None else f"{G:.2f}"

    def _compute(self, which: tuple[str, ...], b=None):
        d = self.ws.require_doc()
        if d is None:
            return None
        fn = (lambda: b) if b is not None else self._boundary_fn(d)
        if fn is None:
            return None
        st = self._settings()
        cal = d.calibration

        def compute():
            bnd = fn()
            bt = ma.thin_boundaries(bnd) if st["thin"] else bnd      # thin once, reuse
            out = {}
            del bnd
            if "distribution" in which:
                dist = ma.grain_area_distribution(bt, cal, exclude_edges=True,
                                                  min_grain_px=st["min_grain_px"])
                dist.labels = np.zeros((0, 0), np.int32)   # not displayed: free ~4 B/px
                out["distribution"] = dist
            if "planimetric" in which:
                out["planimetric"] = ma.planimetric(bt, cal, st["min_grain_px"], thin=False)
            if "intercept" in which:
                out["intercept"] = ma.linear_intercept(bt, cal, st["n_lines"], st["pattern"],
                                                       triple_points=st["triple_points"], thin=False)
            return out

        return run_task(self.ws, "Measuring grain size…", compute,
                        on_done=lambda out: self._show(d, which, out, st),
                        on_error=lambda tb, e: self.ws.status(f"Grain size failed: {e}"))

    def _field_warnings(self, d) -> list[str]:
        w = []
        rows = self._field_rows(d)
        boxes = [bx for bx in d.exclusion_boxes() if rows is None or bx[1] < rows and bx != (0, rows, d.shape[1], d.shape[0])]
        if rows is not None:
            w.append(f"SEM data bar (rows {rows}–{d.shape[0]}) excluded from the analysed field.")
        if boxes:
            w.append(f"{len(boxes)} burned-in scale bar / label region(s) lie inside the analysed "
                     "field and may add false boundaries — crop or draw an ROI.")
        return w

    def _show(self, d, which, out, st) -> None:
        if self.ws.doc is not d:
            return
        fw = self._field_warnings(d)
        shapes, rgba = [], None
        info = []
        if "distribution" in out:
            res = out["distribution"]
            self.last_distribution = res
            prov = self.ws.make_provenance("Grain-area distribution", st, res.warnings + fw, d)
            self.ws.add_summary(res.summary(), "Grain-area distribution", provenance=prov)
            rows = [{"Image": d.name, "Analysis": "Grain distribution", "Grain": i + 1,
                     f"Area ({d.calibration.area_unit})": float(a), f"ECD ({d.calibration.unit})": float(e)}
                    for i, (a, e) in enumerate(zip(res.areas, res.ecd))]
            self.ws.add_results(rows, provenance=prov)
            self.ws.report_warnings(res.warnings, "Grain-area distribution", prov)
            if len(res.ecd):
                counts, edges = np.histogram(res.ecd, bins=max(5, min(40, int(np.sqrt(len(res.ecd))) + 1)))
                self.ws.plot.plot_hist(counts, edges, f"Grain ECD ({d.calibration.unit})", "Count",
                                       f"Grain size distribution — {d.name} (n={len(res.ecd)}, edge grains excluded)",
                                       color="#2ca02c")
        if "planimetric" in out:
            res = out["planimetric"]
            self.last_planimetric = res
            prov = self.ws.make_provenance("Planimetric (Jeffries, E112)", st, res.warnings + fw, d)
            self.ws.add_summary(res.rows(), "Planimetric (Jeffries, E112)", provenance=prov)
            self.ws.report_warnings(res.warnings, "Planimetric (Jeffries, E112)", prov)
            rgba = self._planimetric_rgba(res)
            info.append(f"Planimetric: {res.n_inside} inside + ½·{res.n_intercepted} + ¼·{res.n_corner} "
                        f"= {res.n_equivalent:g} grains → N_A = {res.N_A:.4g} /{res.unit}², "
                        f"G = {self._g(res.G)}")
        if "intercept" in out:
            res = out["intercept"]
            self.last_intercept = res
            prov = self.ws.make_provenance("Linear intercept (E112)", st, res.warnings + fw, d)
            self.ws.add_summary(res.rows(), "Linear intercept (E112)", provenance=prov)
            self.ws.report_warnings(res.warnings, "Linear intercept (E112)", prov)
            shapes = self._intercept_shapes(res)
            u = res.unit
            info.insert(0, f"Intercept: P = {res.intersections:g} on L_T = {res.total_length:.4g} {u} → "
                           f"l̄ = {res.mean_intercept:.4g} {u}, G = {self._g(res.G)}")
        if fw:
            self.ws.summary.add_warnings(fw, {"Image": d.name, "Analysis": "Grain size"})
        if "planimetric" in out or "intercept" in out:
            self.ws.set_analysis_overlay(shapes, rgba)
            d.show_mask = False
            self.ws.refresh()
            self.info.setText("\n".join(info))
            warns = [w for k in ("intercept", "planimetric") if k in out for w in out[k].warnings]
            if warns:
                self.ws.status_label.setStyleSheet("color: #7a4f00;")
                self.ws.status_label.setText("⚠ " + " | ".join(info + warns))
            else:
                self.ws.status(" | ".join(info))
            self.ws.bottom_tabs.setCurrentWidget(self.ws.summary)

    # ---------------------------------------------------------------- buttons
    def run_intercept(self, b=None, draw=True) -> ma.InterceptResult | None:
        out = self._compute(("intercept",), b)
        return out["intercept"] if out else None

    def _intercept_shapes(self, res: ma.InterceptResult) -> list[dict]:
        d = self.ws.doc
        r = max(2.0, min(d.shape) * 0.006)
        shapes = [{"kind": "polyline", "pts": ln, "color": "#00e5ff", "width": 1.5} for ln in res.lines]
        shapes.append({"kind": "points", "pts": res.hits + 0.5, "color": "#ff1744", "size": r,
                       "width": max(1.0, r / 2.5)})
        return shapes

    def run_planimetric(self, b=None, draw=True) -> ma.PlanimetricResult | None:
        out = self._compute(("planimetric",), b)
        return out["planimetric"] if out else None

    def _planimetric_rgba(self, res: ma.PlanimetricResult) -> np.ndarray:
        lab = self.ws.overlay_view(res.labels)
        lut = np.zeros((int(res.labels.max()) + 1, 4), np.uint8)
        lut[res.inside_labels] = (0, 200, 83, 70)
        lut[res.edge_labels] = (255, 196, 0, 90)
        return lut[lab]

    def run_distribution(self, b=None) -> ma.GrainDistribution | None:
        out = self._compute(("distribution",), b)
        return out["distribution"] if out else None

    def run_all(self):
        """Distribution + planimetric + intercept in one background task (boundaries computed
        and thinned once)."""
        return self._compute(("distribution", "planimetric", "intercept"))


# ======================================================================================
class PhasePanel(_Panel):
    """Area fraction of phases by thresholds or multi-level Otsu."""

    def __init__(self, ws):
        super().__init__(ws)
        g, f = group("Phase fraction")
        self.mode = QComboBox()
        self.mode.addItems(["Multi-level Otsu", "Manual thresholds"])
        self.n = ispin(2, 2, 5, tip="Number of phases (intensity classes)")
        f.addRow("Method", hrow(self.mode, self.n))
        self.thr = QLineEdit()
        self.thr.setPlaceholderText("e.g. 85, 170")
        self.thr.setToolTip("Comma-separated ascending grey-value thresholds (Manual mode)")
        f.addRow("Thresholds", self.thr)
        self.use_roi = QCheckBox("Only inside area ROI")
        f.addRow(self.use_roi)
        self.excl_labels = QCheckBox("Exclude scale bar / labels")
        self.excl_labels.setChecked(True)
        self.excl_labels.setToolTip("Leave burned-in scale bars, label boxes and the SEM data bar "
                                    "out of the fractions")
        f.addRow(self.excl_labels)
        f.addRow(button("Compute", self.run))
        self.info = QLabel("")
        self.info.setWordWrap(True)
        f.addRow(self.info)
        self.lay.addWidget(g)
        self.finish()
        self.last: ma.PhaseResult | None = None

    def run(self) -> ma.PhaseResult | None:
        d = self.ws.require_doc()
        if d is None:
            return None
        thr = None
        if self.mode.currentIndex() == 1:
            try:
                thr = [float(t) for t in self.thr.text().replace(";", ",").split(",") if t.strip()]
            except ValueError:
                self.ws.status("Thresholds must be numbers separated by commas.")
                return None
            if not thr:
                self.ws.status("Enter at least one threshold.")
                return None
        mask = None
        if self.use_roi.isChecked():
            roi = self.ws.current_area_roi()
            if roi is not None:
                mask = self.ws.roi_mask(roi)
        boxes = d.exclusion_boxes() if self.excl_labels.isChecked() else []
        if boxes:
            keep = ~iops.regions_mask(d.shape, boxes)
            mask = keep if mask is None else (mask & keep)
        data, n, cal = d.data, self.n.value(), d.calibration
        res = run_task(self.ws, "Computing phase fractions…",
                       lambda: ma.phase_fractions(data, n, thr, cal, mask),
                       on_done=lambda r: self._show(d, r, mask, boxes, thr),
                       on_error=lambda tb, e: self.ws.status(f"Phase fraction failed: {e}"))
        return res

    def _show(self, d, res, mask, boxes, thr):
        self.last = res
        warns = [] if d.calibration.calibrated else ["Image is uncalibrated: phase areas are in px²."]
        prov = self.ws.make_provenance("Phase fraction", {"mode": self.mode.currentText(),
                                                          "n_phases": self.n.value(),
                                                          "thresholds": res.thresholds,
                                                          "roi_only": self.use_roi.isChecked(),
                                                          "excluded_regions": [list(b) for b in boxes]},
                                       warns, d)
        rows = [{"Image": d.name, "Analysis": "Phase fraction", **r} for r in res.rows()]
        self.ws.add_results(rows, provenance=prov)
        self.ws.add_summary([{"Quantity": f"Phase {i + 1} area fraction", "Value": fr, "Unit": "%"}
                             for i, fr in enumerate(res.fractions)], "Phase fraction", provenance=prov)
        self.ws.report_warnings(warns, "Phase fraction", prov)
        cls = self.ws.overlay_view(res.classes)
        mv = self.ws.overlay_view(mask) if mask is not None else None
        rgba = np.zeros(cls.shape + (4,), np.uint8)
        for i in range(len(res.fractions)):
            sel = cls == i
            if mv is not None:
                sel &= mv
            rgba[sel] = PHASE_COLOURS[i % len(PHASE_COLOURS)] + (110,)
        self.ws.set_analysis_overlay(excluded_shapes(boxes), rgba)
        hs = ma.histogram_stats(d.data, 256, mask)
        self.ws.plot.plot_hist(hs.counts, hs.edges, "Grey value", "Count",
                               f"Phase thresholds — {d.name}", vlines=res.thresholds)
        from labstudio.gui.app_settings import format_value as fv

        self.info.setText(" · ".join(f"P{i + 1}: {fv(f)} %" for i, f in enumerate(res.fractions)) +
                          "  (blue, red, green, yellow… dark → bright)")
        self.ws.status(self.info.text())
        return res


# ======================================================================================
class HistogramPanel(_Panel):
    def __init__(self, ws):
        super().__init__(ws)
        g, f = group("Intensity histogram")
        self.bins = ispin(256, 2, 65536, tip="Number of bins")
        self.log = QCheckBox("Log count")
        f.addRow("Bins", hrow(self.bins, self.log))
        self.use_roi = QCheckBox("Only inside area ROI")
        f.addRow(self.use_roi)
        f.addRow(button("Show histogram", self.run, "Plot in the Plot tab (export data CSV there); "
                                                     "statistics → Summary"))
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        f.addRow(self.info)
        self.lay.addWidget(g)
        self.finish()
        self.last: ma.HistogramStats | None = None

    def run(self) -> ma.HistogramStats | None:
        d = self.ws.require_doc()
        if d is None:
            return None
        mask = None
        if self.use_roi.isChecked():
            roi = self.ws.current_area_roi()
            if roi is not None:
                mask = self.ws.roi_mask(roi)
        hs = ma.histogram_stats(d.data, self.bins.value(), mask)
        self.last = hs
        self.ws.plot.log_check.setChecked(self.log.isChecked())
        self.ws.plot.plot_hist(hs.counts, hs.edges, "Grey value", "Count",
                               f"Histogram — {d.name}" + (" (ROI)" if mask is not None else ""))
        self.ws.add_summary(hs.rows(), "Histogram", provenance=self.ws.make_provenance(
            "Histogram", {"bins": self.bins.value(), "roi_only": mask is not None}, [], d))
        self.info.setText(f"n={hs.n}  mean={hs.mean:.4g}  sd={hs.std:.4g}\nmin={hs.min:.4g}  "
                          f"max={hs.max:.4g}  median={hs.median:.4g}  mode={hs.mode:.4g}")
        self.ws.bottom_tabs.setCurrentWidget(self.ws.plot)
        return hs


# ======================================================================================
class AnnotatePanel(_Panel):
    """Scale bar (overlay / burn-in) with style presets, text, arrows and image export."""

    def __init__(self, ws):
        super().__init__(ws)
        g, f = group("Scale bar")
        self.sb_show = QCheckBox("Show scale bar overlay")
        self.sb_show.toggled.connect(lambda _: self.update_scale_bar())
        f.addRow(self.sb_show)
        self.preset = QComboBox()
        self.preset.setToolTip("Named scale-bar styles (built-in + your own, saved in "
                               "~/.labstudio/styles or $LABSTUDIO_HOME/styles)")
        self.preset.activated.connect(lambda _: self.apply_preset(self.preset.currentText()))
        f.addRow("Preset", self.preset)
        f.addRow(hrow(button("Save as…", self._save_preset_dialog, "Save the current style as a preset"),
                      button("Delete", self._delete_preset, "Delete the selected user preset"),
                      button("Apply to all", self.apply_to_all, "Use this scale-bar style on every open image")))
        self.sb_len = dspin(0, 0, 1e9, 1, 3, tip="Bar length in the calibration unit (0 = automatic nice length)")
        self.sb_unit = QLabel("")
        f.addRow("Length", hrow(self.sb_len, self.sb_unit))
        self.sb_frac = dspin(0.2, 0.02, 0.9, 0.05, 2, tip="Automatic length ≈ this fraction of the image width")
        f.addRow("Auto fraction", self.sb_frac)
        self.sb_pos = QComboBox()
        self.sb_pos.addItems(["lower right", "lower left", "upper right", "upper left"])
        self.sb_col = QComboBox()
        self.sb_col.addItems(COLOURS)
        f.addRow("Position", self.sb_pos)
        f.addRow("Colour", self.sb_col)
        self.sb_font = ispin(0, 0, 1000, " px", "Label font size in image pixels (0 = auto)")
        self.sb_thick = ispin(0, 0, 500, " px", "Bar thickness in image pixels (0 = auto)")
        f.addRow("Font / thickness", hrow(self.sb_font, self.sb_thick))
        self.sb_bg = QCheckBox("Box")
        self.sb_bg.setChecked(True)
        self.sb_box_col = QComboBox()
        self.sb_box_col.addItems(["auto"] + COLOURS)
        self.sb_box_col.setToolTip("Background box colour (auto = contrast with the bar)")
        f.addRow(hrow(self.sb_bg, self.sb_box_col))
        self.sb_label = QCheckBox("Label")
        self.sb_label.setChecked(True)
        self.sb_unit_text = QLineEdit("")
        self.sb_unit_text.setPlaceholderText("unit text (default µm…)")
        self.sb_unit_text.setToolTip("Override the unit shown in the label, e.g. 'μm' or 'micron'")
        f.addRow(hrow(self.sb_label, self.sb_unit_text))
        for w in (self.sb_len, self.sb_font, self.sb_thick, self.sb_frac):
            w.valueChanged.connect(lambda _: self.update_scale_bar())
        for w in (self.sb_pos, self.sb_col, self.sb_box_col):
            w.currentIndexChanged.connect(lambda _: self.update_scale_bar())
        for w in (self.sb_bg, self.sb_label):
            w.toggled.connect(lambda _: self.update_scale_bar())
        self.sb_unit_text.editingFinished.connect(self.update_scale_bar)
        f.addRow(button("Burn in scale bar", self.burn_in, "Draw the scale bar permanently into the "
                                                            "pixels (undoable)"))
        self.lay.addWidget(g)

        g, f = group("Text and arrows")
        self.text = QLineEdit("B2 austenite")
        self.text.setToolTip("Text placed with the Text tool")
        f.addRow("Text", self.text)
        self.t_size = ispin(24, 4, 1000, " px", "Font size in image pixels")
        self.t_col = QComboBox()
        self.t_col.addItems(COLOURS)
        self.t_col.setCurrentText("yellow")
        f.addRow("Size / colour", hrow(self.t_size, self.t_col))
        self.a_width = dspin(3, 0.5, 100, 0.5, 1, " px", "Arrow line width in image pixels")
        f.addRow("Arrow width", self.a_width)
        f.addRow(hrow(button("Text tool", lambda: self.ws.set_tool("text")),
                      button("Arrow tool", lambda: self.ws.set_tool("arrow")),
                      button("Clear", self.clear_annotations)))
        self.lay.addWidget(g)

        g, f = group("Export")
        self.incl_rois = QCheckBox("Include ROIs")
        f.addRow(self.incl_rois)
        f.addRow(hrow(button("Export image…", self.ws.export_dialog,
                             "PNG/TIFF with overlays (as shown) or without (raw data)"),
                      button("Burn in overlays", self.burn_overlays,
                             "Render overlays/annotations into the pixels (RGB, undoable)")))
        self.lay.addWidget(g)
        self.finish()
        self._loading = False
        self.refresh_presets()

    # ---------------------------------------------------------------- presets
    def refresh_presets(self, select: str | None = None) -> None:
        cur = select or self.preset.currentText()
        self.preset.blockSignals(True)
        self.preset.clear()
        self.preset.addItems(iops.list_scalebar_presets())
        if cur:
            i = self.preset.findText(cur)
            if i >= 0:
                self.preset.setCurrentIndex(i)
        self.preset.blockSignals(False)

    def style(self) -> dict:
        """Current scale-bar style (the keys of :data:`iops.SCALEBAR_STYLE_KEYS`)."""
        L = self.sb_len.value() or None
        box = self.sb_box_col.currentText()
        return {"position": self.sb_pos.currentText(), "length_rule": "fixed" if L else "auto",
                "length": L, "fraction": self.sb_frac.value(), "color": self.sb_col.currentText(),
                "font_px": self.sb_font.value() or None, "thickness": self.sb_thick.value() or None,
                "background": self.sb_bg.isChecked(), "box_color": None if box == "auto" else box,
                "unit_text": self.sb_unit_text.text().strip(), "show_label": self.sb_label.isChecked()}

    def set_style(self, st: dict) -> None:
        """Load a style into the widgets (and the current image's overlay if shown)."""
        self._loading = True
        try:
            self.sb_len.setValue(float(st.get("length") or 0) if st.get("length_rule", "auto") != "auto" else 0)
            self.sb_frac.setValue(float(st.get("fraction") or 0.2))
            self.sb_pos.setCurrentText(st.get("position", "lower right"))
            col = st.get("color", "white")
            if self.sb_col.findText(col) < 0:
                self.sb_col.addItem(col)
            self.sb_col.setCurrentText(col)
            d = self.ws.doc
            h = d.shape[0] if d is not None else 0
            fpx = st.get("font_px") or (int(round(h * st["font_frac"])) if st.get("font_frac") and h else 0)
            th = st.get("thickness") or (int(round(h * st["thickness_frac"])) if st.get("thickness_frac") and h else 0)
            self.sb_font.setValue(int(fpx or 0))
            self.sb_thick.setValue(int(th or 0))
            self.sb_bg.setChecked(bool(st.get("background", True)))
            box = st.get("box_color") or "auto"
            if self.sb_box_col.findText(box) < 0:
                self.sb_box_col.addItem(box)
            self.sb_box_col.setCurrentText(box)
            self.sb_unit_text.setText(st.get("unit_text") or "")
            self.sb_label.setChecked(bool(st.get("show_label", True)))
        finally:
            self._loading = False
        self.update_scale_bar()

    def apply_preset(self, name: str) -> dict:
        st = iops.load_scalebar_preset(name)
        self.set_style(st)
        i = self.preset.findText(name)
        if i >= 0:
            self.preset.setCurrentIndex(i)
        return st

    def save_preset(self, name: str) -> str:
        path = iops.save_scalebar_preset(name, self.style())
        self.refresh_presets(select=name)
        self.ws.status(f"Saved scale-bar preset '{name}' → {path}")
        return path

    def _save_preset_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, "Save scale-bar preset", "Preset name:")
        if ok and name.strip():
            self.save_preset(name.strip())

    def _delete_preset(self) -> None:
        name = self.preset.currentText()
        if iops.delete_scalebar_preset(name):
            self.refresh_presets()
            self.ws.status(f"Deleted preset '{name}'.")
        else:
            self.ws.status("Built-in presets cannot be deleted.")

    def apply_to_all(self) -> int:
        """Show the current scale-bar style on every open (calibrated) image."""
        self.sb_show.blockSignals(True)
        self.sb_show.setChecked(True)
        self.sb_show.blockSignals(False)
        n = self.ws.apply_scale_bar_style(self.style(), all_images=True)
        self.ws.status(f"Scale-bar style applied to {n} image(s).")
        return n

    # ---------------------------------------------------------------- scale bar
    def on_document_changed(self) -> None:
        d = self.ws.doc
        if d is None:
            return
        self.sb_unit.setText(d.calibration.unit)
        self.sb_show.blockSignals(True)
        self.sb_show.setChecked(d.scale_bar is not None)
        self.sb_show.blockSignals(False)

    def spec(self) -> dict:
        st = self.style()
        st["show_label"] = st.get("show_label", True)
        st["font_px"] = st.get("font_px")
        return st

    def update_scale_bar(self) -> None:
        if getattr(self, "_loading", False):
            return
        d = self.ws.doc
        if d is None:
            return
        if self.sb_show.isChecked() and not d.calibration.calibrated:
            self.ws.status("Set the scale first (Set scale tool or SEM metadata).")
        d.scale_bar = self.spec() if self.sb_show.isChecked() else None
        self.ws.refresh()

    def burn_in(self) -> None:
        d = self.ws.require_doc()
        if d is None:
            return
        if not d.calibration.calibrated:
            self.ws.status("Set the scale first.")
            return
        lo, hi = d.window
        cal = d.calibration
        r = iops.resolve_scalebar_style(self.style(), d.shape, cal)

        def fn(a):
            base = iops.apply_window(a, lo, hi)
            return iops.burn_scale_bar(base, cal, r["length"], r["position"], r["color"],
                                       r["font_px"], r["thickness"], r["background"],
                                       bool(r["label"]), int(r["margin"]), r["box_color"],
                                       self.style().get("unit_text") or None)

        def done():
            d.scale_bar = None
            self.sb_show.blockSignals(True)
            self.sb_show.setChecked(False)
            self.sb_show.blockSignals(False)
            self.ws.refresh()

        self.ws.apply_op("Burn in scale bar", fn, then=done)

    def place_text(self, x: float, y: float) -> None:
        self.ws.add_annotation({"kind": "text", "text": self.text.text(), "pos": (x, y),
                                "size": self.t_size.value(), "color": self.t_col.currentText()})

    def place_arrow(self, p0, p1) -> None:
        self.ws.add_annotation({"kind": "arrow", "p0": p0, "p1": p1, "width": self.a_width.value(),
                                "color": self.t_col.currentText()})

    def clear_annotations(self) -> None:
        d = self.ws.doc
        if d is not None:
            d.annotations.clear()
            self.ws.refresh()

    def burn_overlays(self) -> None:
        d = self.ws.require_doc()
        if d is None:
            return
        from labstudio.gui.image_viewer import qimage_to_array

        arr = qimage_to_array(self.ws.viewer.render_image(True, self.incl_rois.isChecked()))

        def done():
            d.annotations.clear()
            d.shapes.clear()
            d.overlay_rgba = None
            d.scale_bar = None
            self.ws.refresh()

        self.ws.apply_op("Burn in overlays", lambda a: arr, then=done)
