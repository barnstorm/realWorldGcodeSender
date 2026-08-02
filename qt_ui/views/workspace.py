"""Primary toolpath placement and probing workspace."""

import math
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout, QHBoxLayout, QHeaderView, QLabel, QProgressBar, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from qt_ui.bed_scene import BedCanvas
from qt_ui.views.common import toolbar
from qt_ui.widgets import Dock, Dro, Field, JogPad, Tag, button, kicker, muted, section


class WorkspaceView(QWidget):
    request_open = Signal()
    request_recapture = Signal()

    def __init__(self, context, controller):
        super().__init__()
        self.ctx = context
        self.controller = controller
        self.last_in = None
        self._rotation_anchor = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.open_btn = button("Open")
        self.recapture_btn = button("Recapture")
        self.touch_btn = button("Touch-off Z")
        self.mesh_btn = button("Z-Mesh")
        self.send_btn = button("Send job", "primary")
        self.hold_btn = button("Hold")
        self.resume_btn = button("Resume")
        self.kill_btn = button("Kill alarm", "alert")
        self.home_btn = button("Home")
        root.addWidget(toolbar(
            self.open_btn, self.recapture_btn, "|", self.touch_btn, self.mesh_btn,
            "|", self.send_btn, self.hold_btn, self.resume_btn, self.kill_btn,
            self.home_btn,
        ))

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._left_dock())
        body.addWidget(self._center(), 1)
        body.addWidget(self._right_dock())
        host = QWidget()
        host.setLayout(body)
        root.addWidget(host, 1)
        root.addWidget(self._job_bar())

        self.open_btn.clicked.connect(self.request_open)
        self.recapture_btn.clicked.connect(self.request_recapture)
        self.touch_btn.clicked.connect(controller.touch_off)
        self.mesh_btn.clicked.connect(self._mesh)
        self.send_btn.clicked.connect(self._send)
        self.hold_btn.clicked.connect(controller.feed_hold)
        self.resume_btn.clicked.connect(controller.resume)
        self.kill_btn.clicked.connect(controller.kill_alarm)
        self.home_btn.clicked.connect(controller.home)
        self.bed.placeRequested.connect(self._place)
        self.bed.rotateRequested.connect(self._rotate)
        self.bed.hoverMoved.connect(self._hover)

    def _left_dock(self):
        dock = Dock(290)
        self.source = Field("Source file", "", True)
        dock.v.addWidget(section("Job & Paths", self.source))
        params = QGridLayout()
        config = self.ctx.config.cutting_parameters
        values = [
            ("Thickness", config.material_thickness), ("Cutter dia.", config.cutter_diameter),
            ("Feed", config.cut_feed_rate), ("Depth / pass", config.depth_per_pass),
            ("Safe height", config.safe_height), ("Tab height", config.tab_height),
        ]
        for index, (name, value) in enumerate(values):
            params.addWidget(Field(name, "%g" % value, True), index // 2, index % 2)
        params_host = QWidget()
        params_host.setLayout(params)
        dock.v.addWidget(section("Cut parameters", params_host))
        self.paths = QTableWidget(0, 2)
        self.paths.setHorizontalHeaderLabels(["Path", "State"])
        self.paths.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.paths.verticalHeader().hide()
        self.paths.setMaximumHeight(250)
        self.paths.cellClicked.connect(self._select_row)
        dock.v.addWidget(section("Paths", self.paths,
                                 muted("Click bed to place; right-drag to rotate; n/p selects")))
        dock.v.addStretch(1)
        return dock

    def _center(self):
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(20, 12, 20, 12)
        top = QHBoxLayout()
        top.addWidget(kicker("STILL CAPTURE - OVERHEAD-WARPED"))
        top.addStretch(1)
        self.hover_label = muted("Cursor: --")
        top.addWidget(self.hover_label)
        layout.addLayout(top)
        self.bed = BedCanvas()
        layout.addWidget(self.bed, 1)
        return host

    def _right_dock(self):
        dock = Dock(322)
        self.dro = Dro()
        dock.v.addWidget(section("Position - live WCS - in", self.dro))
        frame_grid = QGridLayout()
        self.frame_values = {}
        for row, key in enumerate(("X origin", "Y origin", "Angle", "Z surface")):
            value = QLabel("--")
            tag = Tag("DEFAULT", "neutral")
            frame_grid.addWidget(muted(key), row, 0)
            frame_grid.addWidget(value, row, 1)
            frame_grid.addWidget(tag, row, 2)
            self.frame_values[key] = (value, tag)
        frame_host = QWidget()
        frame_host.setLayout(frame_grid)
        dock.v.addWidget(section("Workpiece frame", frame_host,
                                 muted("Vision proposes, probe disposes")))
        probes = QGridLayout()
        for index, (text, slot) in enumerate((("Touch-off Z", self.controller.touch_off),
                                               ("Z-Mesh", self._mesh))):
            control = button(text)
            control.clicked.connect(slot)
            probes.addWidget(control, index // 2, index % 2)
        probe_host = QWidget()
        probe_host.setLayout(probes)
        dock.v.addWidget(section("Probe", probe_host))
        self.jog = JogPad()
        self.jog.set_enabled(False)
        dock.v.addWidget(section("Jog", self.jog))
        dock.v.addStretch(1)
        return dock

    def _job_bar(self):
        host = QWidget()
        host.setObjectName("jobBar")
        layout = QHBoxLayout(host)
        layout.setContentsMargins(20, 10, 20, 10)
        self.job_state = QLabel("READY")
        self.job_state.setProperty("role", "state")
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setValue(0)
        layout.addWidget(self.job_state)
        layout.addWidget(self.progress, 1)
        return host

    def load_context(self):
        path = self.ctx.svg_file or self.ctx.gcode_file or "No toolpath loaded"
        self.source.edit.setText(Path(path).name if path else "")
        count = len(self.ctx.svg_paths) if self.ctx.svg is not None else (1 if self.ctx.gcode_points else 0)
        self.paths.setRowCount(count + 1 if count else 0)
        if count:
            self.paths.setItem(0, 0, QTableWidgetItem("All paths"))
            self.paths.setItem(0, 1, QTableWidgetItem("active"))
            for index in range(count):
                self.paths.setItem(index + 1, 0, QTableWidgetItem("Path %d" % (index + 1)))
                self.paths.setItem(index + 1, 1, QTableWidgetItem("cut"))
            self.paths.selectRow(max(0, self.ctx.path_index + 1))
        self.refresh_overlay()

    def refresh_overlay(self):
        if self.ctx.bed_image is not None:
            self.bed.set_bed_image(self.ctx.bed_image)
        self.bed.set_overlay(self.ctx.overlay_scene_paths(), self.ctx.tool_width_px(),
                             self.ctx.path_index)
        self.bed.set_drawn(self.ctx.drawn_scene_path())
        self.bed.set_touch_plate(self.ctx.touch_plate_px)
        targets = []
        for target in self.ctx.probe_targets:
            xy = target[:2] if isinstance(target, (tuple, list)) else (target.x, target.y)
            targets.append(self.ctx.transform.phy_to_pixels(*xy))
        self.bed.set_probe_targets(targets)
        self.refresh_frame()

    def refresh_frame(self):
        frame = self.ctx.workpiece_frame
        if frame is None:
            return
        rows = {
            "X origin": (frame.x, "%.4f"), "Y origin": (frame.y, "%.4f"),
            "Angle": (frame.angle, "%.2f"), "Z surface": (frame.z.nominal, "%.4f"),
        }
        for key, (measured, fmt) in rows.items():
            value, tag = self.frame_values[key]
            display = measured.value * 180.0 / math.pi if key == "Angle" else measured.value
            value.setText(fmt % display)
            source = measured.source.name
            variant = "accent" if source in ("PROBE", "SCAN") else ("outline" if source == "MANUAL" else "neutral")
            if key == "Z surface" and frame.z.is_mesh:
                source += " - mesh (%d)" % len(frame.z.samples)
            tag.set_tag(source, variant)

    def update_state(self, mode, mpos, wpos):
        self.dro.set_xyz(*wpos)
        self.bed.set_router(self.ctx.transform.inches_to_pixels(mpos[0], mpos[1]))
        self.job_state.setText(mode.upper())

    def set_progress(self, value):
        self.progress.setValue(value)

    def set_busy(self, busy):
        self.job_state.setText("RUNNING" if busy else "READY")
        if not busy:
            self.refresh_frame()

    def _hover(self, px, py):
        self.last_in = self.ctx.transform.pixels_to_inches(px, py)
        self.hover_label.setText("Cursor: %.3f, %.3f in" % self.last_in)

    def _place(self, px, py):
        x, y = self.ctx.transform.pixels_to_inches(px, py)
        self.last_in = (x, y)
        self.ctx.x_offset, self.ctx.y_offset = x, y
        if self.ctx.svg is not None:
            if self.ctx.path_index == -1:
                self.ctx.path_offsets = [[x, y] for _ in self.ctx.path_offsets]
            else:
                points = self.ctx.svg_paths[self.ctx.path_index]
                min_x = min(point[0] for point in points)
                min_y = min(point[1] for point in points)
                self.ctx.path_offsets[self.ctx.path_index] = [x - min_x, y - min_y]
        self.refresh_overlay()

    def _rotate(self, px, py):
        x, y = self.ctx.transform.pixels_to_inches(px, py)
        self.ctx.rotation = math.degrees(math.atan2(y - self.ctx.y_offset,
                                                   x - self.ctx.x_offset) - math.pi / 2.0)
        self.refresh_overlay()

    def _select_row(self, row, _column):
        self.ctx.path_index = row - 1
        self.refresh_overlay()

    def next_path(self):
        maximum = len(self.ctx.svg_paths) - 1
        self.ctx.path_index = min(self.ctx.path_index + 1, maximum)
        self.load_context()

    def prev_path(self):
        self.ctx.path_index = max(self.ctx.path_index - 1, -1)
        self.load_context()

    def draw_point(self):
        if self.last_in is not None:
            self.ctx.drawn_points.append(self.last_in)
            self.refresh_overlay()

    def draw_arc(self):
        self.draw_point()

    def erase_point(self):
        if self.ctx.drawn_points:
            self.ctx.drawn_points.pop()
            self.refresh_overlay()

    def erase_all(self):
        self.ctx.drawn_points = []
        self.refresh_overlay()

    def jog_to_cursor(self):
        if self.last_in is not None:
            self.controller.jog_to(*self.last_in)

    def _mesh(self):
        self.controller.propose_mesh_targets()
        self.refresh_overlay()
        self.controller.z_mesh()

    def _send(self):
        if self.ctx.svg is not None:
            self.controller.send_svg()
        elif self.ctx.gcode_file:
            self.controller.send_gcode_file()
        else:
            self.controller.send_drawn()

