"""The four workspace views. Each is a self-contained QWidget: a top toolbar row
plus a body of docks/panels. Content is placeholder scaffold — wire the buttons
and readouts to GCodeSender / the probing strategies / app_config in the real app.
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QScrollArea, QFrame,
    QProgressBar, QCheckBox, QComboBox,
)

from widgets import (
    Dock, Panel, Field, Dro, JogPad, BedView,
    label, heading, kicker, muted, button, hline, toolbar_separator,
)

REF = Path(__file__).resolve().parent.parent / "screens" / "ref"


def _toolbar(*items):
    bar = QFrame()
    bar.setStyleSheet("background:#f8f4f4; border-bottom:1px solid #d7d3d3;")
    h = QHBoxLayout(bar)
    h.setContentsMargins(18, 10, 18, 10)
    h.setSpacing(8)
    for it in items:
        if it == "|":
            h.addWidget(toolbar_separator())
        else:
            h.addWidget(it)
    h.addStretch(1)
    return bar


def _section(title, *widgets):
    box = QVBoxLayout()
    box.setSpacing(10)
    box.addWidget(heading(title))
    box.addWidget(hline())
    for w in widgets:
        box.addWidget(w)
    host = QWidget()
    host.setLayout(box)
    return host


# ---------------------------------------------------------------- Workspace ----
class WorkspaceView(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(_toolbar(
            button("Open"), button("Import SVG"), "|",
            button("Recapture"), "|",
            button("Track head", "primary"), button("Touch-off Z"),
            button("Z-Mesh"), button("Edge refine"), "|",
            button("Send job", "primary"), button("Hold"), button("Home"),
        ))

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # left dock: job + cut params
        left = Dock(290)
        params = QGridLayout()
        params.setSpacing(10)
        for i, (lbl, val) in enumerate([
            ("Thickness", "0.500 in"), ("Cutter d", "0.125 in"),
            ("Feed rate", "60 ipm"), ("Depth / pass", "0.100 in"),
            ("Safe height", "0.250 in"), ("Tab height", "0.080 in"),
        ]):
            params.addWidget(Field(lbl, val), i // 2, i % 2)
        phost = QWidget()
        phost.setLayout(params)
        left.v.addWidget(_section("Job & Paths", Field("Source file", "puzzles2.svg")))
        left.v.addWidget(_section("Cut parameters", phost))
        left.v.addStretch(1)

        # center: bed view
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(20, 12, 20, 12)
        cv.addWidget(kicker("STILL CAPTURE - overhead-warped"))
        self.bed = BedView(REF / "cutPath.png")
        cv.addWidget(self.bed, 1)

        # right dock: position + frame + probe + jog
        right = Dock(322)
        right.v.addWidget(_section("Position (live - WCS)",
                                   Dro(values=("4.812", "7.335", "0.000"))))
        probe = QGridLayout()
        for i, t in enumerate(["Touch-off Z", "Z-Mesh", "Edge refine", "XYZ plate"]):
            probe.addWidget(button(t), i // 2, i % 2)
        phost2 = QWidget()
        phost2.setLayout(probe)
        right.v.addWidget(_section("Probe", phost2))
        right.v.addWidget(_section("Jog", JogPad()))
        right.v.addStretch(1)

        body.addWidget(left)
        body.addWidget(center, 1)
        body.addWidget(right)
        body_host = QWidget()
        body_host.setLayout(body)
        root.addWidget(body_host, 1)

        # live job-progress panel
        root.addWidget(self._job_panel())

    def _job_panel(self):
        panel = QFrame()
        panel.setStyleSheet("background:#f8f4f4; border-top:1px solid #d7d3d3;")
        h = QHBoxLayout(panel)
        h.setContentsMargins(20, 12, 20, 12)
        h.setSpacing(24)
        h.addWidget(label("RUNNING", "state"))
        h.addWidget(label("38%", "dro"))
        bar = QProgressBar()
        bar.setValue(38)
        bar.setTextVisible(False)
        colb = QVBoxLayout()
        colb.addWidget(muted("Line 458 / 1,204     Path 7 / 18"))
        colb.addWidget(bar)
        colhost = QWidget()
        colhost.setLayout(colb)
        h.addWidget(colhost, 1)
        h.addWidget(muted("Elapsed 04:12    Remaining 06:48    58 ipm - 18k"))
        h.addWidget(button("Hold"))
        h.addWidget(button("Stop", "alert"))
        return panel


# -------------------------------------------------------------- Calibration ----
class CalibrationView(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(_toolbar(
            button("Camera: /dev/video0"), button("Recapture"), "|",
            button("Detect markers", "primary"), button("Camera calibration..."),
            "|", button("Reset"),
        ))
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(20, 12, 20, 12)
        cv.addWidget(kicker("RAW CAMERA - angled view"))
        cv.addWidget(BedView(REF / "cnc13.jpg"), 1)

        right = Dock(340)
        det = QGridLayout()
        for i, (k, v) in enumerate([("Left rail", "4 / 4"), ("Right rail", "4 / 4"),
                                    ("Touch-plate id 66", "found"),
                                    ("Homography residual", "0.6 px")]):
            det.addWidget(muted(k), i, 0)
            det.addWidget(label(v), i, 1, alignment=Qt.AlignRight)
        dhost = QWidget()
        dhost.setLayout(det)
        right.v.addWidget(_section("Detection", dhost))

        phys = QGridLayout()
        for i, (lbl, val) in enumerate([("Bed X", "35.0"), ("Bed Y", "35.0"),
                                        ("Bed Z", "3.5"), ("Box width", "0.706")]):
            phys.addWidget(Field(lbl, val), i // 2, i % 2)
        phost = QWidget()
        phost.setLayout(phys)
        right.v.addWidget(_section("Physical setup", phost))
        right.v.addWidget(button("Accept & open workspace", "primary"))
        right.v.addStretch(1)

        body.addWidget(center, 1)
        body.addWidget(right)
        host = QWidget()
        host.setLayout(body)
        root.addWidget(host, 1)


# ------------------------------------------------------------------ Machine ----
class MachineView(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(_toolbar(
            button("Home ($H)"), button("Unlock ($X)"), "|",
            button("Hold"), button("Resume"), button("Kill alarm", "alert"),
        ))
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        left = Dock(296)
        left.v.addWidget(_section("Coordinates - WCS",
                                  Dro(values=("4.812", "7.335", "0.000"))))
        goto = QGridLayout()
        for i, t in enumerate(["XY zero", "Z safe", "Work zero", "Machine 0"]):
            goto.addWidget(button(t), i // 2, i % 2)
        ghost = QWidget()
        ghost.setLayout(goto)
        left.v.addWidget(_section("Go to", ghost))
        left.v.addStretch(1)

        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(20, 12, 20, 12)
        cv.addWidget(kicker("Watch the head while you jog"))
        cv.addWidget(BedView(REF / "cutPath.png"), 1)

        right = Dock(360)
        right.v.addWidget(_section("Jog", JogPad()))
        ov = QGridLayout()
        for i, t in enumerate(["Feed 100%", "Rapid 100%", "Spindle 100%"]):
            ov.addWidget(button(t), i, 0)
        ovhost = QWidget()
        ovhost.setLayout(ov)
        right.v.addWidget(_section("Overrides", ovhost))
        right.v.addWidget(_section("Recovery",
                                   button("Feed hold"), button("Resume"),
                                   button("Kill alarm ($X)", "alert")))
        right.v.addStretch(1)

        body.addWidget(left)
        body.addWidget(center, 1)
        body.addWidget(right)
        host = QWidget()
        host.setLayout(body)
        root.addWidget(host, 1)


# ----------------------------------------------------------------- Settings ----
class SettingsView(QWidget):
    SECTIONS = {
        "Physical setup": [
            "ChArUco box width", "Bed size X", "Bed size Y", "Bed size Z",
            "Right ref box X", "Right ref box Y", "Right ref Z offset", "Right far height",
            "Left ref box X", "Left ref box Y", "Left ref Z offset", "Left far height",
        ],
        "Cutting parameters": [
            "Material thickness", "Cutter diameter", "Cut feed rate", "Depth per pass",
            "Depth below material", "Safe height", "Tab height",
        ],
        "Vision": ["Bed view size (px)", "Camera device index", "Camera width", "Camera height"],
        "Communication": ["COM port", "Baud rate"],
        "Probing": ["Touch-plate height", "Touch-plate width", "Distance to notch",
                    "Probe feed fast", "Probe feed slow"],
    }

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        nav = Dock(224)
        for name in self.SECTIONS:
            b = button(name)
            b.setStyleSheet("text-align:left;")
            nav.v.addWidget(b)
        nav.v.addStretch(1)

        form_host = QWidget()
        form = QVBoxLayout(form_host)
        form.setContentsMargins(34, 26, 34, 26)
        form.setSpacing(28)
        for title, fields in self.SECTIONS.items():
            grid = QGridLayout()
            grid.setSpacing(12)
            for i, f in enumerate(fields):
                grid.addWidget(Field(f), i // 3, i % 3)
            if title == "Communication":
                grid.addWidget(QCheckBox("Auto-detect port"), 1, 0)
            ghost = QWidget()
            ghost.setLayout(grid)
            form.addWidget(_section(title, ghost))
        form.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_host)
        scroll.setFrameShape(QFrame.NoFrame)

        body.addWidget(nav)
        body.addWidget(scroll, 1)
        host = QWidget()
        host.setLayout(body)
        root.addWidget(host, 1)

        # action bar
        actions = QFrame()
        actions.setStyleSheet("background:#f8f4f4; border-top:1px solid #d7d3d3;")
        ah = QHBoxLayout(actions)
        ah.setContentsMargins(22, 10, 22, 10)
        ah.addWidget(button("Load default"))
        ah.addWidget(button("Load from file..."))
        ah.addStretch(1)
        ah.addWidget(muted("Unsaved changes"))
        ah.addWidget(button("Apply"))
        ah.addWidget(button("Save as..."))
        ah.addWidget(button("Save", "primary"))
        root.addWidget(actions)
