"""Reusable widgets for the Classical Qt UI.

Adapted from design_handoff_qt_ui/pyside6_skeleton/widgets.py: factories keep
the views terse and push styling into classical.qss via roles/properties.
Additions over the skeleton: Tag (fidelity chips), StatePill, Console.
Nothing here talks to hardware.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QLabel, QPushButton, QFrame, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QPlainTextEdit,
)

ACCENT = "#b68235"
ACCENT_700 = "#7d5411"
ALERT = "#a3352a"


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


def section(title, *widgets):
    box = QVBoxLayout()
    box.setSpacing(10)
    box.addWidget(heading(title))
    box.addWidget(hline())
    for w in widgets:
        box.addWidget(w)
    host = QWidget()
    host.setLayout(box)
    return host


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
    def __init__(self, label_text, value="", read_only=False):
        super().__init__()
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        col.addWidget(muted(label_text))
        self.edit = QLineEdit(value)
        self.edit.setReadOnly(read_only)
        col.addWidget(self.edit)


class Tag(QLabel):
    """Fidelity chip. variant: 'accent' (probe), 'outline' (manual),
    'neutral' (vision) -- the handoff's source->tag mapping."""
    def __init__(self, text="", variant="neutral"):
        super().__init__(text)
        self.setProperty("role", "tag")
        self.set_variant(variant)
        self.setAlignment(Qt.AlignCenter)

    def set_variant(self, variant):
        self.setProperty("variant", variant)
        # re-polish so QSS picks up the changed property
        self.style().unpolish(self)
        self.style().polish(self)

    def set_tag(self, text, variant):
        self.setText(text)
        self.set_variant(variant)


class StatePill(QLabel):
    """Right-aligned status word (IDLE / RUN / HOLD / ALARM / OFFLINE)."""
    def __init__(self, text="OFFLINE"):
        super().__init__(text)
        self.setProperty("role", "state")

    def set_state(self, text, alert=False):
        self.setText(text)
        self.setStyleSheet("color:%s;" % (ALERT if alert else ""))


class Dro(Panel):
    """Digital readout: axis rows with large tabular figures."""
    def __init__(self, axes=("X", "Y", "Z"), values=("—", "—", "—")):
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

    def set_xyz(self, x, y, z, fmt="%.3f"):
        self.set_value("X", fmt % x)
        self.set_value("Y", fmt % y)
        self.set_value("Z", fmt % z)

    def clear(self):
        for ax in self.values:
            self.set_value(ax, "—")


class JogPad(QWidget):
    """3x3 XY jog grid + a Z column. Buttons exposed for wiring; M1 keeps
    them disabled (no sender)."""
    _XY = [["↖", "↑", "↗"],
           ["←", "XY0", "→"],
           ["↙", "↓", "↘"]]
    _DIRS = {(0, 0): (-1, 1), (0, 1): (0, 1), (0, 2): (1, 1),
             (1, 0): (-1, 0), (1, 1): None,   (1, 2): (1, 0),
             (2, 0): (-1, -1), (2, 1): (0, -1), (2, 2): (1, -1)}

    def __init__(self):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        self.xy_buttons = {}   # (dx, dy) -> button; None key = center XY0
        self.z_buttons = {}    # +1 / -1 -> button

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setSpacing(8)
        for r in range(3):
            for c in range(3):
                btn = button(self._XY[r][c])
                btn.setFixedSize(58, 58)
                grid.addWidget(btn, r, c)
                self.xy_buttons[self._DIRS[(r, c)]] = btn
        row.addWidget(grid_host)

        zcol = QVBoxLayout()
        zcol.setSpacing(8)
        for t, dz in (("Z+", 1), ("Z-", -1)):
            b = button(t)
            b.setFixedWidth(62)
            b.setMinimumHeight(58)
            zcol.addWidget(b)
            self.z_buttons[dz] = b
        row.addLayout(zcol)
        row.addStretch(1)

    def set_enabled(self, enabled):
        for b in list(self.xy_buttons.values()) + list(self.z_buttons.values()):
            b.setEnabled(enabled)


class Console(QPlainTextEdit):
    """Monospace log strip -- the one functional exception to the serif
    system. Batched appends only (ConsoleFeed drains a deque into here)."""
    def __init__(self, max_blocks=2000):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_blocks)
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)
        self.setObjectName("console")
        self.setFixedHeight(88)

    def append_lines(self, lines):
        if lines:
            self.appendPlainText("\n".join(lines))
