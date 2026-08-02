"""Version-adaptive machine and jog controls."""

from PySide6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from qt_ui.bed_scene import BedCanvas
from qt_ui.views.common import toolbar
from qt_ui.widgets import Dock, Dro, JogPad, StatePill, button, kicker, muted, section


class MachineView(QWidget):
    def __init__(self, context, controller, shared_scene):
        super().__init__()
        self.ctx = context
        self.controller = controller
        self.feed = 300.0
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        home = button("Home ($H)")
        unlock = button("Unlock ($X)")
        hold = button("Hold")
        resume = button("Resume")
        kill = button("Kill alarm", "alert")
        root.addWidget(toolbar(home, unlock, "|", hold, resume, kill))
        home.clicked.connect(controller.home)
        unlock.clicked.connect(controller.unlock)
        hold.clicked.connect(controller.feed_hold)
        resume.clicked.connect(controller.resume)
        kill.clicked.connect(controller.kill_alarm)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._left())
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(20, 12, 20, 12)
        center_layout.addWidget(kicker("WATCH THE HEAD WHILE YOU JOG"))
        self.bed = BedCanvas(shared_scene)
        center_layout.addWidget(self.bed, 1)
        body.addWidget(center, 1)
        body.addWidget(self._right())
        host = QWidget()
        host.setLayout(body)
        root.addWidget(host, 1)

    def _left(self):
        dock = Dock(296)
        self.dro = Dro()
        dock.v.addWidget(section("Coordinates - WCS", self.dro))
        zeros = QGridLayout()
        for column, axis in enumerate(("X", "Y", "Z")):
            control = button(axis + "0")
            control.clicked.connect(lambda _checked=False, a=axis: self.controller.zero_axis(a))
            zeros.addWidget(control, 0, column)
        host = QWidget()
        host.setLayout(zeros)
        dock.v.addWidget(section("Set work zero", host))
        self.mcs = QLabel("MCS  --  --  --")
        dock.v.addWidget(section("Machine coordinates", self.mcs))
        dock.v.addStretch(1)
        return dock

    def _right(self):
        dock = Dock(360)
        header = QHBoxLayout()
        self.state = StatePill("OFFLINE")
        self.version = muted("waiting for controller")
        header.addWidget(self.state)
        header.addStretch(1)
        header.addWidget(self.version)
        header_host = QWidget()
        header_host.setLayout(header)
        self.step = QComboBox()
        for value in ("0.001", "0.01", "0.1", "1.0"):
            self.step.addItem(value, float(value))
        self.step.setCurrentIndex(2)
        self.jog = JogPad()
        self.jog.set_enabled(False)
        for direction, control in self.jog.xy_buttons.items():
            if direction is None:
                control.clicked.connect(lambda: self.controller.jog_to(0.0, 0.0))
            else:
                control.clicked.connect(lambda _checked=False, d=direction: self._jog(d[0], d[1], 0))
        for dz, control in self.jog.z_buttons.items():
            control.clicked.connect(lambda _checked=False, z=dz: self._jog(0, 0, z))
        dock.v.addWidget(section("Jog", header_host, self.step, self.jog))
        overrides = QGridLayout()
        self.override_controls = {}
        for row, kind in enumerate(("feed", "rapid", "spindle")):
            minus, plus = button("-"), button("+")
            value = QLabel("100%")
            minus.clicked.connect(lambda _checked=False, k=kind: self.controller.adjust_override(k, -1))
            plus.clicked.connect(lambda _checked=False, k=kind: self.controller.adjust_override(k, 1))
            overrides.addWidget(QLabel(kind.title()), row, 0)
            overrides.addWidget(minus, row, 1)
            overrides.addWidget(value, row, 2)
            overrides.addWidget(plus, row, 3)
            self.override_controls[kind] = (minus, plus)
        override_host = QWidget()
        override_host.setLayout(overrides)
        dock.v.addWidget(section("Overrides", override_host))
        dock.v.addStretch(1)
        return dock

    def _jog(self, dx, dy, dz):
        step = float(self.step.currentData())
        self.controller.jog_relative(dx * step, dy * step, dz * step, self.feed)

    def set_capabilities(self, caps):
        self.ctx.grbl_caps = caps
        self.version.setText("GRBL " + caps.version)
        self.jog.set_enabled(True)
        for kind in ("rapid", "spindle"):
            for control in self.override_controls[kind]:
                control.setEnabled(caps.realtime_overrides)
                control.setToolTip("" if caps.realtime_overrides else "Requires GRBL 1.1+")

    def update_state(self, mode, mpos, wpos):
        self.state.set_state(mode.upper(), mode.lower() == "alarm")
        self.dro.set_xyz(*wpos)
        self.mcs.setText("MCS  %.3f  %.3f  %.3f" % mpos)

    def disconnected(self):
        self.state.set_state("OFFLINE", True)
        self.dro.clear()
        self.jog.set_enabled(False)

