"""The overhead bed view: pixmap + overlay items + monolith mouse semantics.

Scene coordinates == bed-view pixels (the monolith's overlay space), so every
value from bed_coords/overlay_model drops in unchanged. The view emits scene
coordinates; the workspace view converts to inches via BedTransform.

Mouse semantics (port of OverlayGcode onclick/onmousemove/onrelease):
left click-release without an intervening move -> placeRequested; right
press-drag -> rotateRequested per move and on release; hover -> hoverMoved.
"""

import numpy as np
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

ACCENT = "#b68235"
ACCENT_700 = "#7d5411"
ALERT = "#a3352a"
DRAWN = "#2e7d32"


def bgr_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """OpenCV BGR ndarray -> QPixmap. QImage does NOT own the numpy buffer:
    copy before the array can be freed or reused. Pass strides explicitly."""
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]
    qimg = QImage(frame.data, w, h, frame.strides[0], QImage.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class BedCanvas(QGraphicsView):
    placeRequested = Signal(float, float)    # scene px, left click (no drag)
    rotateRequested = Signal(float, float)   # scene px, right press/drag
    hoverMoved = Signal(float, float)        # scene px

    def __init__(self, scene=None):
        super().__init__()
        self._scene = scene if scene is not None else QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)

        self._pixmap_item = None
        self._path_items = []
        self._drawn_item = None
        self._ghost_item = None
        self._target_items = []
        self._plate_item = None
        self._router = None
        self._moved = False
        self._press_button = None

    # -- content ------------------------------------------------------------

    def set_bed_image(self, frame: np.ndarray):
        pix = bgr_to_qpixmap(frame)
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pix)
            self._pixmap_item.setZValue(0)
        else:
            self._pixmap_item.setPixmap(pix)
        self._scene.setSceneRect(0, 0, pix.width(), pix.height())
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    @staticmethod
    def _polyline_path(points, powers) -> QPainterPath:
        path = QPainterPath()
        pen_down = False
        for (x, y), power in zip(points, powers):
            if power <= 0.0 or not pen_down:
                path.moveTo(x, y)
                pen_down = True
            else:
                path.lineTo(x, y)
        return path

    def set_overlay(self, polylines, tool_width: int, selected: int = -1):
        """polylines: [{'points': [(px,py)...], 'powers': [...]}]; selected
        path gets the accent color (-1 = none highlighted / all normal)."""
        for item in self._path_items:
            self._scene.removeItem(item)
        self._path_items = []
        for i, poly in enumerate(polylines):
            color = QColor(ACCENT if i == selected else ACCENT_700)
            pen = QPen(color, max(1, tool_width))
            pen.setCapStyle(Qt.RoundCap)
            item = self._scene.addPath(self._polyline_path(poly["points"], poly["powers"]), pen)
            item.setZValue(1)
            self._path_items.append(item)

    def set_drawn(self, polylines):
        if self._drawn_item is not None:
            self._scene.removeItem(self._drawn_item)
            self._drawn_item = None
        if polylines and polylines[0]["points"]:
            poly = polylines[0]
            pen = QPen(QColor(DRAWN), 2)
            self._drawn_item = self._scene.addPath(
                self._polyline_path(poly["points"], [0.0] + [1.0] * (len(poly["points"]) - 1)), pen)
            self._drawn_item.setZValue(2)

    def set_ghost(self, points_px):
        """Preview of the next drawn segment/arc (dashed)."""
        if self._ghost_item is not None:
            self._scene.removeItem(self._ghost_item)
            self._ghost_item = None
        if points_px and len(points_px) >= 2:
            pen = QPen(QColor(DRAWN), 1, Qt.DashLine)
            path = QPainterPath(QPointF(*points_px[0]))
            for p in points_px[1:]:
                path.lineTo(*p)
            self._ghost_item = self._scene.addPath(path, pen)
            self._ghost_item.setZValue(6)

    def set_probe_targets(self, points_px, done: int = 0):
        for item in self._target_items:
            self._scene.removeItem(item)
        self._target_items = []
        for i, (x, y) in enumerate(points_px):
            color = QColor(ACCENT if i < done else ACCENT_700)
            pen = QPen(color, 2)
            group = self._scene.createItemGroup([
                self._scene.addLine(x - 6, y, x + 6, y, pen),
                self._scene.addLine(x, y - 6, x, y + 6, pen),
                self._scene.addEllipse(x - 4, y - 4, 8, 8, pen),
            ])
            group.setZValue(3)
            self._target_items.append(group)

    def set_touch_plate(self, corners_px):
        if self._plate_item is not None:
            self._scene.removeItem(self._plate_item)
            self._plate_item = None
        if corners_px:
            pen = QPen(QColor(ACCENT), 2, Qt.DashLine)
            path = QPainterPath(QPointF(*corners_px[0]))
            for p in list(corners_px[1:]) + [corners_px[0]]:
                path.lineTo(*p)
            self._plate_item = self._scene.addPath(path, pen)
            self._plate_item.setZValue(4)

    def set_router(self, pos_px):
        """pos_px = (x, y) scene pixels, or None to hide."""
        if pos_px is None:
            if self._router is not None:
                self._router.setVisible(False)
            return
        if self._router is None:
            self._router = self._build_marker()
        self._router.setVisible(True)
        self._router.setPos(QPointF(*pos_px))

    def _build_marker(self):
        pen = QPen(QColor(ACCENT_700), 3)
        group = self._scene.createItemGroup([])
        group.addToGroup(self._scene.addEllipse(-9, -9, 18, 18, QPen(QColor(ACCENT), 2)))
        group.addToGroup(self._scene.addLine(0, -18, 0, -6, pen))
        group.addToGroup(self._scene.addLine(0, 6, 0, 18, pen))
        group.addToGroup(self._scene.addLine(-18, 0, -6, 0, pen))
        group.addToGroup(self._scene.addLine(6, 0, 18, 0, pen))
        group.addToGroup(self._scene.addEllipse(-2, -2, 4, 4, pen, QBrush(QColor(ACCENT_700))))
        group.setZValue(5)
        return group

    # -- mouse (monolith semantics) -----------------------------------------

    def _scene_pos(self, event):
        p = self.mapToScene(event.position().toPoint())
        return p.x(), p.y()

    def mousePressEvent(self, event):
        self._moved = False
        self._press_button = event.button()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        x, y = self._scene_pos(event)
        self.hoverMoved.emit(x, y)
        if self._press_button is not None:
            self._moved = True
            if self._press_button == Qt.RightButton:
                self.rotateRequested.emit(x, y)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        x, y = self._scene_pos(event)
        if event.button() == Qt.LeftButton and not self._moved:
            self.placeRequested.emit(x, y)
        elif event.button() == Qt.RightButton:
            self.rotateRequested.emit(x, y)
        self._press_button = None
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        if self._scene.sceneRect().width() > 0:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)
