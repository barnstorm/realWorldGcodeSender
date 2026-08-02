from PySide6.QtWidgets import QFrame, QHBoxLayout

from qt_ui.widgets import toolbar_separator


def toolbar(*items):
    bar = QFrame()
    bar.setObjectName("viewToolbar")
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(18, 9, 18, 9)
    layout.setSpacing(8)
    for item in items:
        layout.addWidget(toolbar_separator() if item == "|" else item)
    layout.addStretch(1)
    return bar

