"""QGraphicsView-based micrograph viewer: zoom/pan, pixel read-out, ROI drawing and overlays.

Coordinates exchanged with the rest of the app are *image pixel coordinates* (x = column,
y = row, pixel centres at integer + 0.5 in scene space are not used: scene (0, 0) is the
top-left corner of pixel (0, 0), so a pixel index is ``floor(scene coordinate)``).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap,
                           QPolygonF, QTransform)
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem,
                               QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
                               QGraphicsView)

#: ROI kinds and whether they enclose an area.
ROI_KINDS = {"line": False, "polyline": False, "angle": False, "rect": True, "ellipse": True,
             "polygon": True, "freehand": True, "point": False}

#: Drawing tools offered by the viewer (tool name → ROI kind emitted, or None for special tools).
TOOLS = {"pan": None, "line": "line", "polyline": "polyline", "angle": "angle", "rect": "rect",
         "ellipse": "ellipse", "polygon": "polygon", "freehand": "freehand", "point": "point",
         "scale": "line", "profile": "line", "text": None, "arrow": None}

_roi_ids = itertools.count(1)

#: Images larger than this (pixels along either side) are displayed from a downsampled level
#: plus a full-resolution tile of the visible region (keeps QPixmap memory bounded).
PYRAMID_LIMIT = 4096

_FONT_FAMILY: list[str] = []


def annotation_font_family() -> str:
    """Family name of the bundled DejaVu Sans (registered with Qt once) so overlays and exported
    images render µ, ², ° identically on every platform; '' if unavailable."""
    if not _FONT_FAMILY:
        fam = ""
        try:
            from PySide6.QtGui import QFontDatabase

            from labstudio.core.image_ops import annotation_font_path

            path = annotation_font_path()
            if path:
                fid = QFontDatabase.addApplicationFont(path)
                fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
                fam = fams[0] if fams else ""
        except Exception:  # noqa: BLE001
            fam = ""
        _FONT_FAMILY.append(fam)
    return _FONT_FAMILY[0]


def annotation_qfont(px: int, bold: bool = False) -> QFont:
    fam = annotation_font_family()
    f = QFont(fam) if fam else QFont()
    f.setPixelSize(max(4, int(px)))
    f.setBold(bold)
    return f


@dataclass
class Roi:
    """A region/line of interest in image pixel coordinates."""

    kind: str
    points: list[tuple[float, float]]
    name: str = ""
    id: int = field(default_factory=lambda: next(_roi_ids))

    def __post_init__(self) -> None:
        self.points = [(float(x), float(y)) for x, y in self.points]
        if not self.name:
            self.name = f"{self.kind}-{self.id}"

    @property
    def is_area(self) -> bool:
        return ROI_KINDS.get(self.kind, False)

    def vertices(self) -> np.ndarray:
        """Polygon vertices (N×2) for area ROIs (rect/ellipse expanded), raw points otherwise."""
        from labstudio.core.image_ops import ellipse_vertices, rect_vertices

        p = np.asarray(self.points, float)
        if self.kind == "rect":
            return rect_vertices(p[0, 0], p[0, 1], p[1, 0], p[1, 1])
        if self.kind == "ellipse":
            return ellipse_vertices(p[0, 0], p[0, 1], p[1, 0], p[1, 1])
        return p

    def bounds(self) -> tuple[float, float, float, float]:
        v = self.vertices()
        return float(v[:, 0].min()), float(v[:, 1].min()), float(v[:, 0].max()), float(v[:, 1].max())


def _pen(color, width: float = 1.5, cosmetic: bool = True, style=Qt.SolidLine) -> QPen:
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    pen.setCosmetic(cosmetic)
    pen.setStyle(style)
    pen.setJoinStyle(Qt.RoundJoin)
    pen.setCapStyle(Qt.RoundCap)
    return pen


def roi_path(roi: Roi) -> QPainterPath:
    path = QPainterPath()
    pts = roi.points
    if not pts:
        return path
    if roi.kind == "rect" and len(pts) >= 2:
        path.addRect(QRectF(QPointF(*pts[0]), QPointF(*pts[1])).normalized())
    elif roi.kind == "ellipse" and len(pts) >= 2:
        path.addEllipse(QRectF(QPointF(*pts[0]), QPointF(*pts[1])).normalized())
    elif roi.kind == "point":
        for x, y in pts:
            path.moveTo(x - 4, y)
            path.lineTo(x + 4, y)
            path.moveTo(x, y - 4)
            path.lineTo(x, y + 4)
    else:
        path.moveTo(*pts[0])
        for p in pts[1:]:
            path.lineTo(*p)
        if roi.kind in ("polygon", "freehand") and len(pts) > 2:
            path.closeSubpath()
    return path


class RoiItem(QGraphicsPathItem):
    """Graphics item for a :class:`Roi` (yellow, cyan when selected)."""

    def __init__(self, roi: Roi):
        super().__init__(roi_path(roi))
        self.roi = roi
        self.setZValue(20)
        self.set_selected(False)
        if roi.kind == "point":
            self._labels = []
            for i, (x, y) in enumerate(roi.points, start=1):
                t = QGraphicsSimpleTextItem(str(i), self)
                t.setBrush(QColor("yellow"))
                t.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                t.setPos(x + 2, y + 2)
                self._labels.append(t)

    def set_selected(self, sel: bool) -> None:
        self.setPen(_pen("#00e5ff" if sel else "#ffd400", 2.0 if sel else 1.5))


class ImageViewer(QGraphicsView):
    """Zoomable/pannable image view with ROI tools and overlay layers.

    Signals
    -------
    mouseMoved(x, y): image pixel coordinates under the cursor (NaN when outside).
    roiCreated(tool, roi): a ROI finished with ``tool`` (tool may be 'scale'/'profile').
    clickedAt(tool, x, y): click with the 'text' tool.
    arrowCreated(x0, y0, x1, y1): arrow drawn with the 'arrow' tool.
    """

    mouseMoved = Signal(float, float)
    roiCreated = Signal(str, object)
    clickedAt = Signal(str, float, float)
    arrowCreated = Signal(float, float, float, float)
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self._pix = QGraphicsPixmapItem()
        self._pix.setTransformationMode(Qt.FastTransformation)  # crisp pixels when zoomed in
        self._pix.setZValue(0)
        self.scene().addItem(self._pix)
        self._detail = QGraphicsPixmapItem()      # full-resolution tile for big images
        self._detail.setTransformationMode(Qt.FastTransformation)
        self._detail.setZValue(1)
        self._detail.setVisible(False)
        self.scene().addItem(self._detail)
        self._overlay = QGraphicsPixmapItem()
        self._overlay.setZValue(5)
        self.scene().addItem(self._overlay)
        self._img_shape: tuple[int, int] = (0, 0)
        self._qimage_buf = None
        self._full: np.ndarray | None = None       # full-resolution 8-bit display array
        self.pyramid_step = 1                      # downsampling of the base pixmap
        self._detail_rect: tuple[int, int, int, int] | None = None
        self.horizontalScrollBar().valueChanged.connect(lambda _: self._update_detail())
        self.verticalScrollBar().valueChanged.connect(lambda _: self._update_detail())
        self._shape_items: list[QGraphicsItem] = []
        self._annotation_items: list[QGraphicsItem] = []
        self._scalebar_items: list[QGraphicsItem] = []
        self._roi_items: dict[int, RoiItem] = {}
        self._tool = "pan"
        self._drawing: list[tuple[float, float]] | None = None
        self._temp: QGraphicsPathItem | None = None
        self._panning: QPointF | None = None
        self.overlays_visible = True
        self.setTool("pan")

    # ------------------------------------------------------------------ image
    @staticmethod
    def array_to_qimage(arr: np.ndarray) -> tuple[QImage, np.ndarray]:
        """uint8 gray (H×W) or RGB(A) → QImage (returns the backing buffer too)."""
        a = np.ascontiguousarray(arr)
        if a.dtype != np.uint8:
            raise ValueError("display arrays must be uint8")
        h, w = a.shape[:2]
        if a.ndim == 2:
            q = QImage(a.data, w, h, a.strides[0], QImage.Format_Grayscale8)
        elif a.shape[2] == 3:
            q = QImage(a.data, w, h, a.strides[0], QImage.Format_RGB888)
        else:
            q = QImage(a.data, w, h, a.strides[0], QImage.Format_RGBA8888)
        return q, a

    def set_image(self, display: np.ndarray | None, fit: bool = False) -> None:
        """Show an 8-bit gray/RGB display array. ``fit`` re-fits the view."""
        if display is None:
            self._pix.setPixmap(QPixmap())
            self._detail.setVisible(False)
            self._full = None
            self._img_shape = (0, 0)
            self.scene().setSceneRect(QRectF(0, 0, 1, 1))
            return
        new_shape = display.shape[:2]
        self._full = np.ascontiguousarray(display)
        step = max(1, int(math.ceil(max(new_shape) / PYRAMID_LIMIT)))
        self.pyramid_step = step
        base = downsample_display(self._full, step) if step > 1 else self._full
        q, buf = self.array_to_qimage(base)
        self._qimage_buf = buf
        self._pix.setPixmap(QPixmap.fromImage(q))
        self._pix.setScale(float(step))
        self._detail_rect = None
        self._detail.setVisible(False)
        changed = new_shape != self._img_shape
        self._img_shape = new_shape
        h, w = new_shape
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        if fit or changed:
            self.fit()
        self._update_detail()

    @property
    def image_shape(self) -> tuple[int, int]:
        return self._img_shape

    def _update_detail(self) -> None:
        """Big images: when zoomed in past the base level, show the visible region at full res."""
        if self._full is None or self.pyramid_step <= 1:
            return
        if self.zoom_factor() * self.pyramid_step <= 1.05:
            self._detail.setVisible(False)
            return
        h, w = self._img_shape
        r = self.mapToScene(self.viewport().rect()).boundingRect()
        mx, my = r.width() * 0.25, r.height() * 0.25
        x0, y0 = max(0, int(r.left() - mx)), max(0, int(r.top() - my))
        x1, y1 = min(w, int(math.ceil(r.right() + mx))), min(h, int(math.ceil(r.bottom() + my)))
        if x1 <= x0 or y1 <= y0:
            self._detail.setVisible(False)
            return
        cur = self._detail_rect
        if cur is not None and self._detail.isVisible() and cur[0] <= r.left() and cur[1] <= r.top() \
                and cur[2] >= r.right() and cur[3] >= r.bottom():
            return
        tile = np.ascontiguousarray(self._full[y0:y1, x0:x1])
        q, buf = self.array_to_qimage(tile)
        self._detail_buf = buf
        self._detail.setPixmap(QPixmap.fromImage(q))
        self._detail.setPos(x0, y0)
        self._detail.setVisible(True)
        self._detail_rect = (x0, y0, x1, y1)

    def display_full(self) -> np.ndarray | None:
        """Full-resolution 8-bit display array currently shown."""
        return self._full

    def set_overlay(self, rgba: np.ndarray | None) -> None:
        """Semi-transparent RGBA overlay (same size as the image), e.g. threshold preview."""
        if rgba is None:
            self._overlay.setPixmap(QPixmap())
            self._overlay_buf = None
            return
        q, buf = self.array_to_qimage(rgba)
        self._overlay_buf = buf
        self._overlay.setPixmap(QPixmap.fromImage(q))
        h, w = self._img_shape
        self._overlay.setScale(float(w) / rgba.shape[1] if w and rgba.shape[1] else 1.0)
        self._overlay.setVisible(self.overlays_visible)

    # ------------------------------------------------------------------ zoom
    def zoom_factor(self) -> float:
        return self.transform().m11()

    def fit(self) -> None:
        h, w = self._img_shape
        if h and w:
            self.fitInView(QRectF(0, 0, w, h), Qt.KeepAspectRatio)
            self.zoomChanged.emit(self.zoom_factor())

    def zoom_100(self) -> None:
        self.setTransform(QTransform())
        self.zoomChanged.emit(1.0)
        self._update_detail()

    def zoom_by(self, f: float) -> None:
        z = self.zoom_factor() * f
        if 0.005 <= z <= 64:
            self.scale(f, f)
            self.zoomChanged.emit(self.zoom_factor())
            self._update_detail()

    def resizeEvent(self, ev) -> None:  # noqa: N802
        super().resizeEvent(ev)
        self._update_detail()

    def wheelEvent(self, ev) -> None:  # noqa: N802
        self.zoom_by(1.25 if ev.angleDelta().y() > 0 else 0.8)

    # ------------------------------------------------------------------ tools
    def tool(self) -> str:
        return self._tool

    def setTool(self, tool: str) -> None:  # noqa: N802
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool}")
        self._cancel_drawing()
        self._tool = tool
        self.setDragMode(QGraphicsView.ScrollHandDrag if tool == "pan" else QGraphicsView.NoDrag)
        self.viewport().setCursor(Qt.OpenHandCursor if tool == "pan" else Qt.CrossCursor)

    def _img_pos(self, ev) -> tuple[float, float]:
        p = self.mapToScene(ev.position().toPoint())
        return float(p.x()), float(p.y())

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        h, w = self._img_shape
        return float(np.clip(x, 0, w)), float(np.clip(y, 0, h))

    def _cancel_drawing(self) -> None:
        self._drawing = None
        if self._temp is not None:
            self.scene().removeItem(self._temp)
            self._temp = None

    def _update_temp(self, kind: str, pts) -> None:
        if self._temp is None:
            self._temp = QGraphicsPathItem()
            self._temp.setPen(_pen("#ff4081", 1.5, style=Qt.DashLine))
            self._temp.setZValue(30)
            self.scene().addItem(self._temp)
        self._temp.setPath(roi_path(Roi(kind if kind != "angle" else "polyline", list(pts), name="tmp",
                                        id=0)))

    def _finish(self, pts) -> None:
        tool = self._tool
        kind = TOOLS[tool]
        self._cancel_drawing()
        if tool == "arrow":
            if len(pts) >= 2:
                self.arrowCreated.emit(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1])
            return
        if kind is None:
            return
        if kind in ("line",) and len(pts) >= 2 and math.dist(pts[0], pts[-1]) < 1:
            return
        if kind in ("rect", "ellipse") and (abs(pts[-1][0] - pts[0][0]) < 1 or abs(pts[-1][1] - pts[0][1]) < 1):
            return
        if kind in ("polygon", "freehand") and len(pts) < 3:
            return
        if kind == "polyline" and len(pts) < 2:
            return
        if kind in ("line", "rect", "ellipse"):
            pts = [pts[0], pts[-1]]
        self.roiCreated.emit(tool, Roi(kind, list(pts)))

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.MiddleButton:
            self._panning = ev.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        if self._tool == "pan" or ev.button() != Qt.LeftButton or not self._img_shape[0]:
            if ev.button() == Qt.RightButton and self._drawing:
                self._finish(self._drawing)
                return
            super().mousePressEvent(ev)
            return
        x, y = self._clamp(*self._img_pos(ev))
        tool, kind = self._tool, TOOLS[self._tool]
        if tool == "text":
            self.clickedAt.emit("text", x, y)
            return
        if kind == "point":
            if self._drawing is None:
                self._drawing = []
            self._drawing.append((x, y))
            self._update_temp("point", self._drawing)
            return
        if kind in ("polygon", "polyline", "angle"):
            if self._drawing is None:
                self._drawing = [(x, y), (x, y)]
            else:
                self._drawing[-1] = (x, y)
                self._drawing.append((x, y))
            if kind == "angle" and len(self._drawing) > 3:
                self._finish(self._drawing[:3])
                return
            self._update_temp(kind, self._drawing)
            return
        # drag tools: line, rect, ellipse, freehand, arrow
        self._drawing = [(x, y), (x, y)]
        self._update_temp(kind or "line", self._drawing)

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._panning is not None:
            d = ev.position() - self._panning
            self._panning = ev.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        x, y = self._img_pos(ev)
        h, w = self._img_shape
        if 0 <= x < w and 0 <= y < h:
            self.mouseMoved.emit(x, y)
        else:
            self.mouseMoved.emit(float("nan"), float("nan"))
        if self._drawing is not None and self._tool != "pan":
            kind = TOOLS[self._tool] or "line"
            cx, cy = self._clamp(x, y)
            if kind == "freehand":
                if math.dist(self._drawing[-1], (cx, cy)) >= 1.0:
                    self._drawing.append((cx, cy))
            elif kind != "point":
                self._drawing[-1] = (cx, cy)
            self._update_temp(kind, self._drawing)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.MiddleButton and self._panning is not None:
            self._panning = None
            self.viewport().setCursor(Qt.OpenHandCursor if self._tool == "pan" else Qt.CrossCursor)
            return
        if self._drawing is not None and ev.button() == Qt.LeftButton:
            kind = TOOLS[self._tool]
            if self._tool == "arrow" or kind in ("line", "rect", "ellipse", "freehand"):
                self._finish(self._drawing)
                return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:  # noqa: N802
        if self._drawing is not None and TOOLS.get(self._tool) in ("polygon", "polyline", "point"):
            pts = self._drawing
            if TOOLS[self._tool] != "point":
                # the double click added a duplicate vertex — drop it
                pts = [p for i, p in enumerate(pts) if i == 0 or math.dist(p, pts[i - 1]) > 0.5]
            self._finish(pts)
            return
        super().mouseDoubleClickEvent(ev)

    def keyPressEvent(self, ev) -> None:  # noqa: N802
        if ev.key() == Qt.Key_Escape:
            self._cancel_drawing()
            return
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter) and self._drawing is not None:
            self._finish(self._drawing)
            return
        if ev.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_by(1.25)
            return
        if ev.key() == Qt.Key_Minus:
            self.zoom_by(0.8)
            return
        super().keyPressEvent(ev)

    def finish_current(self) -> None:
        """Finish an in-progress multi-click ROI (polygon/polyline/points)."""
        if self._drawing is not None:
            self._finish(self._drawing)

    # ------------------------------------------------------------------ ROIs
    def add_roi_item(self, roi: Roi) -> RoiItem:
        item = RoiItem(roi)
        item.setVisible(self.overlays_visible)
        self.scene().addItem(item)
        self._roi_items[roi.id] = item
        return item

    def remove_roi_item(self, roi_id: int) -> None:
        item = self._roi_items.pop(roi_id, None)
        if item is not None:
            self.scene().removeItem(item)

    def clear_rois(self) -> None:
        for rid in list(self._roi_items):
            self.remove_roi_item(rid)

    def select_roi(self, roi_id: int | None) -> None:
        for rid, it in self._roi_items.items():
            it.set_selected(rid == roi_id)

    # ------------------------------------------------------------------ vector overlays
    def _make_shape(self, s: dict) -> list[QGraphicsItem]:
        kind = s.get("kind")
        color = s.get("color", "#00e676")
        width = float(s.get("width", 1.5))
        items: list[QGraphicsItem] = []
        if kind in ("polyline", "polygon"):
            pts = np.asarray(s["pts"], float)
            if len(pts) < 2:
                return items
            path = QPainterPath(QPointF(*pts[0]))
            for p in pts[1:]:
                path.lineTo(QPointF(*p))
            if kind == "polygon" or s.get("closed"):
                path.closeSubpath()
            it = QGraphicsPathItem(path)
            it.setPen(_pen(color, width, cosmetic=s.get("cosmetic", True)))
            items.append(it)
        elif kind == "paths":  # many polylines in one item (fast for particle outlines)
            path = QPainterPath()
            for pts in s["paths"]:
                pts = np.asarray(pts, float)
                if len(pts) < 2:
                    continue
                path.addPolygon(QPolygonF([QPointF(*p) for p in pts]))
                if s.get("closed", True):
                    path.closeSubpath()
            it = QGraphicsPathItem(path)
            it.setPen(_pen(color, width))
            items.append(it)
        elif kind == "points":
            path = QPainterPath()
            r = float(s.get("size", 3.0))
            for x, y in np.asarray(s["pts"], float).reshape(-1, 2):
                if s.get("marker", "o") == "+":
                    path.moveTo(x - r, y)
                    path.lineTo(x + r, y)
                    path.moveTo(x, y - r)
                    path.lineTo(x, y + r)
                else:
                    path.addEllipse(QPointF(x, y), r, r)
            it = QGraphicsPathItem(path)
            it.setPen(_pen(color, width, cosmetic=False))
            if s.get("fill"):
                it.setBrush(QColor(s["fill"]))
            items.append(it)
        elif kind == "text":
            it = QGraphicsSimpleTextItem(str(s["text"]))
            it.setFont(annotation_qfont(int(s.get("size", 14)), bool(s.get("bold", False))))
            it.setBrush(QColor(color))
            it.setPos(*s["pos"])
            if s.get("outline", True):
                it.setPen(_pen("black" if QColor(color).lightness() > 128 else "white", 0.6,
                               cosmetic=False))
            items.append(it)
        elif kind == "arrow":
            (x0, y0), (x1, y1) = s["p0"], s["p1"]
            w = width
            L = math.hypot(x1 - x0, y1 - y0) or 1.0
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            head = float(s.get("head", max(6.0, 4 * w)))
            path = QPainterPath(QPointF(x0, y0))
            path.lineTo(QPointF(x1 - ux * head * 0.8, y1 - uy * head * 0.8))
            line = QGraphicsPathItem(path)
            line.setPen(_pen(color, w, cosmetic=False))
            tri = QPolygonF([QPointF(x1, y1),
                             QPointF(x1 - ux * head - uy * head * 0.45, y1 - uy * head + ux * head * 0.45),
                             QPointF(x1 - ux * head + uy * head * 0.45, y1 - uy * head - ux * head * 0.45)])
            hp = QPainterPath()
            hp.addPolygon(tri)
            hp.closeSubpath()
            hi = QGraphicsPathItem(hp)
            hi.setPen(_pen(color, 0.5, cosmetic=False))
            hi.setBrush(QColor(color))
            items += [line, hi]
        elif kind == "rect":
            x0, y0, x1, y1 = s["rect"]
            it = QGraphicsRectItem(QRectF(QPointF(x0, y0), QPointF(x1, y1)).normalized())
            it.setPen(_pen(color, width, cosmetic=s.get("cosmetic", True),
                           style=Qt.DashLine if s.get("dashed") else Qt.SolidLine))
            if s.get("fill"):
                c = QColor(s["fill"])
                if "alpha" in s:
                    c.setAlpha(int(s["alpha"]))
                it.setBrush(QBrush(c, Qt.BDiagPattern) if s.get("hatch") else QBrush(c))
            items.append(it)
        return items

    def _add_items(self, shapes, store: list, z: float) -> None:
        for s in shapes:
            for it in self._make_shape(s):
                it.setZValue(z)
                it.setVisible(self.overlays_visible)
                self.scene().addItem(it)
                store.append(it)

    def set_shapes(self, shapes: list[dict]) -> None:
        """Replace the analysis overlay shapes (test lines, outlines, labels...)."""
        for it in self._shape_items:
            self.scene().removeItem(it)
        self._shape_items = []
        self._add_items(shapes, self._shape_items, 10)

    def set_annotations(self, shapes: list[dict]) -> None:
        """Replace user annotations (text, arrows)."""
        for it in self._annotation_items:
            self.scene().removeItem(it)
        self._annotation_items = []
        self._add_items(shapes, self._annotation_items, 15)

    def set_scale_bar(self, spec: dict | None) -> None:
        """Draw a scale bar overlay. ``spec``: length_px, label, position, color, font_px,
        thickness, background (bool)."""
        for it in self._scalebar_items:
            self.scene().removeItem(it)
        self._scalebar_items = []
        h, w = self._img_shape
        if spec is None or not h:
            return
        L = float(spec["length_px"])
        th = float(spec.get("thickness") or max(2, round(h * 0.012)))
        fpx = int(spec.get("font_px") or max(10, round(h * 0.035)))
        margin = float(spec.get("margin") or max(4, round(min(h, w) * 0.025)))
        color = QColor(spec.get("color", "white"))
        text = QGraphicsSimpleTextItem(spec.get("label", ""))
        text.setFont(annotation_qfont(fpx, bold=True))
        text.setBrush(color)
        tb = text.boundingRect()
        gap = max(2.0, th / 2)
        bw = max(L, tb.width())
        bh = th + (gap + tb.height() if spec.get("label") else 0)
        pos = spec.get("position", "lower right")
        x0 = w - margin - bw if "right" in pos else margin
        y0 = h - margin - bh if "lower" in pos else margin
        items: list[QGraphicsItem] = []
        if spec.get("background", True):
            pad = max(3.0, th)
            bg = QGraphicsRectItem(QRectF(x0 - pad, y0 - pad, bw + 2 * pad, bh + 2 * pad))
            bc = spec.get("box_color")
            bg.setBrush(QColor(bc) if bc else (QColor("black") if color.lightness() > 127 else QColor("white")))
            bg.setPen(QPen(Qt.NoPen))
            items.append(bg)
        bar = QGraphicsRectItem(QRectF(x0 + (bw - L) / 2, y0, L, th))
        bar.setBrush(color)
        bar.setPen(QPen(Qt.NoPen))
        items.append(bar)
        if spec.get("label"):
            text.setPos(x0 + (bw - tb.width()) / 2, y0 + th + gap - 0.15 * tb.height())
            items.append(text)
        for i, it in enumerate(items):
            it.setZValue(40 + i)
            it.setVisible(self.overlays_visible)
            self.scene().addItem(it)
        self._scalebar_items = items

    def clear_overlays(self) -> None:
        self.set_shapes([])
        self.set_overlay(None)

    def set_overlays_visible(self, vis: bool) -> None:
        self.overlays_visible = vis
        for it in (self._shape_items + self._annotation_items + self._scalebar_items
                   + list(self._roi_items.values())):
            it.setVisible(vis)
        self._overlay.setVisible(vis)

    # ------------------------------------------------------------------ export
    def render_image(self, with_overlays: bool = True, include_rois: bool = False) -> QImage:
        """Render the image region at native resolution (optionally with overlays) to a QImage."""
        h, w = self._img_shape
        out = QImage(w, h, QImage.Format_RGB32)
        out.fill(Qt.black)
        hidden = []
        base_items = [self._pix, self._detail]
        base_vis = [it.isVisible() for it in base_items]
        if not with_overlays or not include_rois:
            groups = [] if with_overlays else (self._shape_items + self._annotation_items
                                               + self._scalebar_items + [self._overlay])
            if not include_rois:
                groups = groups + list(self._roi_items.values())
            for it in groups:
                if it.isVisible():
                    hidden.append(it)
                    it.setVisible(False)
        if self._temp is not None:
            self._temp.setVisible(False)
        p = QPainter(out)
        p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        if self._full is not None:  # native-resolution pixels, never the downsampled base level
            q, _buf = self.array_to_qimage(self._full)
            p.drawImage(0, 0, q)
            for it in base_items:
                it.setVisible(False)
        self.scene().render(p, QRectF(0, 0, w, h), QRectF(0, 0, w, h))
        p.end()
        for it, v in zip(base_items, base_vis):
            it.setVisible(v)
        for it in hidden:
            it.setVisible(True)
        if self._temp is not None:
            self._temp.setVisible(True)
        return out


def downsample_display(a: np.ndarray, step: int) -> np.ndarray:
    """Block-mean downsampling of an 8-bit display array by an integer ``step``."""
    if step <= 1:
        return a
    h, w = a.shape[:2]
    hh, ww = (h // step) * step, (w // step) * step
    core = a[:hh, :ww]
    shp = (hh // step, step, ww // step, step) + a.shape[2:]
    out = core.reshape(shp).mean(axis=(1, 3), dtype=np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def qimage_to_array(q: QImage) -> np.ndarray:
    """QImage → H×W×3 uint8 RGB array."""
    q = q.convertToFormat(QImage.Format_RGB888)
    w, h = q.width(), q.height()
    buf = np.frombuffer(q.constBits(), dtype=np.uint8, count=q.sizeInBytes())
    return buf.reshape(h, q.bytesPerLine())[:, : w * 3].reshape(h, w, 3).copy()
