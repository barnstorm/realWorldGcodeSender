"""Reusable widget helpers for the Classical Qt UI skeleton.

These factories keep the views terse and push all styling into classical.qss via
object roles / properties. Nothing here talks to hardware — it is pure UI scaffold.
"""
from pathlib import Path

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPixmap, QPen, QColor, QBrush, QPainter
from PySide6.QtWidgets import (
    QLabel, QPushButton, QFrame, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QGraphicsView, QGraphicsScene,
)

ACCENT = "#b68235"
ACCENT_700 = "#7d5411"


# ---- small factories ---------------------------------------------------------
def label(text, role=None, align=None):
    lab = QLabel(text)
    if role:
        lab.setProperty("role", role)
    if align:
        lab.setAlignment(align)
    return lab


def heading(text):
    return label(text, "h6")


def kicker(text):
    return label(text, "kicker")


def muted(text):
    return label(text, "muted")


def button(text, variant=None):
    btn = QPushButton(text)
    if variant:
        btn.setProperty("variant", variant)
    btn.setCursor(Qt.PointingHandCursor)
    return btn


def hline():
    line = QFrame()
    line.setProperty("hline", "true")
    line.setFixedHeight(1)
    return line


def toolbar_separator():
    sep = QFrame()
    sep.setFrameShape(QFrame.VLine)
    sep.setFixedWidth(1)
    sep.setStyleSheet("color:#d7d3d3;")
    return sep


class Panel(QFrame):
    """Bordered, unfilled surface."""
    def __init__(self):
        super().__init__()
        self.setObjectName("panel")


class Dock(QFrame):
    """A side dock column."""
    def __init__(self, width=300):
        super().__init__()
        self.setObjectName("dock")
        self.setFixedWidth(width)
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(18, 18, 18, 18)
        self.v.setSpacing(18)


class Field(QWidget):
    """Labelled text input (mirrors the .field pattern)."""
    def __init__(self, label_text, value=""):
        super().__init__()
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        col.addWidget(muted(label_text))
        self.edit = QLineEdit(value)
        col.addWidget(self.edit)


class Dro(Panel):
    """Digital readout: axis rows with large tabular figures."""
    def __init__(self, axes=("X", "Y", "Z"), values=("0.000", "0.000", "0.000")):
        super().__init__()
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        self.values = {}
        for i, (ax, val) in enumerate(zip(axes, values)):
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(12, 8, 12, 8)
            axl = muted(ax)
            axl.setFixedWidth(18)
            val_lab = label(val, "dro", Qt.AlignRight | Qt.AlignVCenter)
            h.addWidget(axl)
            h.addWidget(val_lab, 1)
            col.addWidget(row)
            if i < len(axes) - 1:
                col.addWidget(hline())
            self.values[ax] = val_lab

    def set_value(self, axis, text):
        if axis in self.values:
            self.values[axis].setText(text)


class JogPad(QWidget):
    """3x3 XY jog grid + a Z column."""
    _XY = [["\u2196", "\u2191", "\u2197"],
           ["\u2190", "XY0", "\u2192"],
           ["\u2199", "\u2193", "\u2198"]]

    def __init__(self):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setSpacing(8)
        for r in range(3):
            for c in range(3):
                btn = button(self._XY[r][c])
                btn.setFixedSize(58, 58)
                grid.addWidget(btn, r, c)
        row.addWidget(grid_host)

        zcol = QVBoxLayout()
        zcol.setSpacing(8)
        for t in ("Z+", "Z-"):
            b = button(t)
            b.setFixedWidth(62)
            b.setMinimumHeight(58)
            zcol.addWidget(b)
        row.addLayout(zcol)
        row.addStretch(1)


class BedView(QGraphicsView):
    """Overhead bed view with a live router-position marker.

    In the real app: set the pixmap from warp_to_overhead(); draw the G-code path
    and probe targets as QGraphicsItems; call set_router_pos() from live grbl WPos.
    """
    def __init__(self, image_path=None):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        rect = None
        if image_path and Path(image_path).exists():
            pix = QPixmap(str(image_path))
            self._scene.addPixmap(pix)
            rect = pix.rect()
        if rect is None:
            self._scene.addRect(0, 0, 640, 640, QPen(QColor("#c9b48a")),
                                QBrush(QColor("#ded8cb")))
            rect = self._scene.itemsBoundingRect()

        w, h = rect.width(), rect.height()
        self._marker = self._build_marker()
        self.set_router_pos(w * 0.52, h * 0.34)

    def _build_marker(self):
        pen = QPen(QColor(ACCENT_700), 3)
        group = self._scene.createItemGroup([])
        group.addToGroup(self._scene.addEllipse(-9, -9, 18, 18, QPen(QColor(ACCENT), 2)))
        group.addToGroup(self._scene.addLine(0, -18, 0, -6, pen))
        group.addToGroup(self._scene.addLine(0, 6, 0, 18, pen))
        group.addToGroup(self._scene.addLine(-18, 0, -6, 0, pen))
        group.addToGroup(self._scene.addLine(6, 0, 18, 0, pen))
        group.addToGroup(self._scene.addEllipse(-2, -2, 4, 4, pen, QBrush(QColor(ACCENT_700))))
        return group

    def set_router_pos(self, x, y):
        self._marker.setPos(QPointF(x, y))

    def resizeEvent(self, event):
        self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)
