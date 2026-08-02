"""Camera calibration view, including the explicit detection error state."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

import camera_intrinsics
from qt_ui.bed_scene import BedCanvas
from qt_ui.camera_enum import available_cameras
from qt_ui.lens_dialog import LensCalibrationDialog
from qt_ui.views.common import toolbar
from qt_ui.widgets import Dock, Field, Tag, button, kicker, muted, section


class CalibrationView(QWidget):
    request_recapture = Signal()
    accepted = Signal()
    camera_changed = Signal(object)   # device index (int) or None for test image
    rotation_changed = Signal(int)    # degrees clockwise

    def __init__(self, context):
        super().__init__()
        self.ctx = context
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.camera = QComboBox()
        self.camera.addItem("Test image (no camera)", None)
        for device_index, name in available_cameras():
            self.camera.addItem(name, device_index)
        self.lens_btn = button("Calibrate lens")
        self.rotate_ccw = button("Rotate CCW")
        self.rotate_cw = button("Rotate CW")
        self.rotation_label = muted("0\N{DEGREE SIGN}")
        self.recapture = button("Recapture")
        self.detect = button("Detect markers", "primary")
        self.reset = button("Reset")
        root.addWidget(toolbar(muted("Camera"), self.camera, self.lens_btn, "|",
                               self.rotate_ccw, self.rotation_label, self.rotate_cw,
                               "|", self.recapture, self.detect, "|", self.reset))
        self.alert = QLabel("")
        self.alert.setObjectName("alertBanner")
        self.alert.hide()
        root.addWidget(self.alert)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(20, 12, 20, 12)
        center_layout.addWidget(kicker("RAW CAMERA - ANGLED VIEW"))
        self.raw = BedCanvas()
        center_layout.addWidget(self.raw, 1)
        body.addWidget(center, 1)
        body.addWidget(self._dock())
        host = QWidget()
        host.setLayout(body)
        root.addWidget(host, 1)

        self.recapture.clicked.connect(self.request_recapture)
        self.detect.clicked.connect(self.request_recapture)
        self.accept_btn.clicked.connect(self.accepted)
        self.camera.currentIndexChanged.connect(
            lambda _index: self.camera_changed.emit(self.camera.currentData()))
        self.camera.currentIndexChanged.connect(lambda _index: self._update_lens_state())
        self.lens_btn.clicked.connect(self._open_lens_dialog)
        self.rotate_ccw.clicked.connect(lambda: self._rotate(-90))
        self.rotate_cw.clicked.connect(lambda: self._rotate(90))
        self._update_lens_state()

    def _rotate(self, delta):
        vision = self.ctx.config.vision_settings
        vision.camera_rotation = (vision.camera_rotation + delta) % 360
        self.rotation_label.setText("%d\N{DEGREE SIGN}" % vision.camera_rotation)
        self.rotation_changed.emit(vision.camera_rotation)

    def select_camera(self, device_index):
        """Sync the combo to the session's capture source without emitting."""
        at = self.camera.findData(device_index) if device_index is not None else 0
        self.camera.blockSignals(True)
        self.camera.setCurrentIndex(max(at, 0))
        self.camera.blockSignals(False)
        self._update_lens_state()

    def _update_lens_state(self):
        device = self.camera.currentData()
        self.lens_btn.setEnabled(device is not None)
        self.rotate_ccw.setEnabled(device is not None)
        self.rotate_cw.setEnabled(device is not None)
        self.rotation_label.setText(
            "%d\N{DEGREE SIGN}" % (self.ctx.config.vision_settings.camera_rotation % 360))
        if device is None:
            self.lens_tag.set_tag("n/a", "neutral")
            return
        profile = camera_intrinsics.load_profile(device)
        if profile is None:
            self.lens_tag.set_tag("not calibrated", "alert")
        else:
            self.lens_tag.set_tag("rms %.2f px" % profile["rms"], "accent")

    def _open_lens_dialog(self):
        device = self.camera.currentData()
        if device is None:
            return
        dialog = LensCalibrationDialog(device, self.camera.currentText(),
                                       self.ctx.config, self)
        dialog.saved.connect(lambda _index: self._update_lens_state())
        dialog.exec()

    def _dock(self):
        dock = Dock(340)
        grid = QGridLayout()
        self.detection = {}
        for row, key in enumerate(("Left rail", "Right rail", "Touch plate", "Residual")):
            value = Tag("--", "neutral")
            grid.addWidget(muted(key), row, 0)
            grid.addWidget(value, row, 1)
            self.detection[key] = value
        self.lens_tag = Tag("--", "neutral")
        grid.addWidget(muted("Lens"), len(self.detection), 0)
        grid.addWidget(self.lens_tag, len(self.detection), 1)
        host = QWidget()
        host.setLayout(grid)
        dock.v.addWidget(section("Detection", host))
        p = self.ctx.config.physical_setup
        fields = QGridLayout()
        for index, (name, value) in enumerate((("Bed X", p.bed_size_x), ("Bed Y", p.bed_size_y),
                                               ("Bed Z", p.bed_size_z), ("Box width", p.box_width))):
            fields.addWidget(Field(name, "%g" % value, True), index // 2, index % 2)
        field_host = QWidget()
        field_host.setLayout(fields)
        dock.v.addWidget(section("Physical setup", field_host))
        self.accept_btn = button("Accept & open workspace", "primary")
        dock.v.addWidget(self.accept_btn)
        dock.v.addStretch(1)
        return dock

    def show_result(self, valid, message="", info=None):
        if self.ctx.raw_frame is not None:
            self.raw.set_bed_image(self.ctx.raw_frame)
        if valid:
            info = info or {}
            counts = info.get("counts", {})
            for side in ("Left", "Right"):
                rail = counts.get("%s rail" % side, 0)
                bed = counts.get("%s bed" % side, 0)
                text = "%d found" % rail + (" +%d bed" % bed if bed else "")
                self.detection["%s rail" % side].set_tag(
                    text, "accent" if rail or bed else "alert")
            rms = info.get("rms")
            self.detection["Residual"].set_tag(
                "%.2f px" % rms if rms is not None else "valid", "neutral")
            plate_found = info.get("plate_found", False)
            self.detection["Touch plate"].set_tag(
                "found" if plate_found else "not found",
                "accent" if plate_found else "alert")
            self.alert.hide()
            self.accept_btn.setEnabled(True)
        else:
            self.alert.setText("Calibration invalid: " + (message or "rail markers not detected"))
            self.alert.show()
            for tag in self.detection.values():
                tag.set_tag("not found", "alert")
            self.accept_btn.setEnabled(False)

