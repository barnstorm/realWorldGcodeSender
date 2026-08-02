"""Lens calibration dialog: capture calibration-target views, compute intrinsics.

Point the selected camera at the LightBurn AprilTag sheet (CalibrationTags.pdf
-- partial views are fine) or the printed checkerboard (page 4 of
markers/marker_sheets.pdf, must be fully visible) and collect 8+ captures at
varied angles and positions.  Compute runs cv2.calibrateCamera in a worker
thread; Save writes the profile that capture_bed_image() then applies to
every live frame.
"""

import threading

import cv2
from PySide6.QtCore import Qt, QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QHBoxLayout, QLabel, QVBoxLayout,
)

import camera_intrinsics
from qt_ui.bed_scene import bgr_to_qpixmap
from qt_ui.widgets import button, muted

PREVIEW_W, PREVIEW_H = 640, 360


class _CalibrateSignals(QObject):
    finished = Signal(float, object, object)   # rms, camera_matrix, dist_coeffs
    failed = Signal(str)


class LensCalibrationDialog(QDialog):
    saved = Signal(int)   # device index

    def __init__(self, device_index, camera_name, config, parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self.camera_name = camera_name
        self.config = config
        self.views = []
        self.frame_size = None
        self.result = None
        self._last_target = None
        self._auto_countdown = 0

        self.setWindowTitle("Lens calibration - %s" % camera_name)
        root = QVBoxLayout(self)
        self.preview = QLabel("Opening camera...")
        self.preview.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background: #222; color: #aaa;")
        root.addWidget(self.preview)
        self.status = muted("Show the AprilTag sheet or checkerboard to the camera")
        root.addWidget(self.status)

        bar = QHBoxLayout()
        self.auto = QCheckBox("Auto-capture every 2 s")
        self.auto.setChecked(True)
        self.capture_btn = button("Capture")
        self.compute_btn = button("Compute")
        self.save_btn = button("Save profile", "primary")
        self.capture_btn.setEnabled(False)
        self.compute_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        close_btn = button("Close")
        close_btn.clicked.connect(self.reject)
        bar.addWidget(self.auto)
        bar.addStretch(1)
        bar.addWidget(self.capture_btn)
        bar.addWidget(self.compute_btn)
        bar.addWidget(self.save_btn)
        bar.addWidget(close_btn)
        root.addLayout(bar)

        self.capture_btn.clicked.connect(self._capture)
        self.compute_btn.clicked.connect(self._compute)
        self.save_btn.clicked.connect(self._save)

        self.cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.vision_settings.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.vision_settings.camera_height)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(400)

    # -- capture loop -------------------------------------------------------

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status.setText("Camera did not return a frame")
            return
        self.frame_size = (frame.shape[1], frame.shape[0])
        target = camera_intrinsics.find_target(frame, fast=True)
        self._last_target = target
        display = frame.copy()
        if target is not None:
            if target["kind"] == "apriltag":
                for outline in target["outlines"]:
                    cv2.polylines(display, [outline.astype(int)], True,
                                  (60, 220, 60), 2)
            else:
                cv2.drawChessboardCorners(
                    display, camera_intrinsics.CHECKERBOARD_SIZE,
                    target["corners"], True)
        self.capture_btn.setEnabled(target is not None)
        self._set_status(target)
        if target is not None and self.auto.isChecked() and self.result is None:
            self._auto_countdown += 1
            if self._auto_countdown >= 5:   # 5 ticks x 400 ms = 2 s
                self._capture()
        else:
            self._auto_countdown = 0
        self._show(display)

    def _set_status(self, target):
        if self.result is not None:
            return
        if target is None:
            seen = "No target visible"
        elif target["kind"] == "apriltag":
            seen = "AprilTag sheet: %d tags" % target["count"]
        else:
            seen = "Checkerboard found"
        self.status.setText(
            "%s - %d capture%s (need %d+), vary angle and position"
            % (seen, len(self.views), "" if len(self.views) == 1 else "s",
               camera_intrinsics.MIN_CAPTURES))

    def _show(self, frame):
        self.preview.setPixmap(
            bgr_to_qpixmap(cv2.resize(frame, (PREVIEW_W, PREVIEW_H))))

    def _capture(self):
        if self._last_target is None:
            return
        self.views.append(self._last_target)
        self._auto_countdown = 0
        self.compute_btn.setEnabled(
            len(self.views) >= camera_intrinsics.MIN_CAPTURES)
        self._set_status(self._last_target)

    # -- calibration --------------------------------------------------------

    def _compute(self):
        self.compute_btn.setEnabled(False)
        self.status.setText("Computing intrinsics from %d captures..."
                            % len(self.views))
        signals = _CalibrateSignals(self)
        signals.finished.connect(self._computed)
        signals.failed.connect(self._compute_failed)
        views, size = list(self.views), self.frame_size

        def worker():
            try:
                rms, matrix, coeffs = camera_intrinsics.calibrate(views, size)
                signals.finished.emit(rms, matrix, coeffs)
            except Exception as exc:
                signals.failed.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _computed(self, rms, matrix, coeffs):
        self.result = (rms, matrix, coeffs)
        self.save_btn.setEnabled(True)
        self.compute_btn.setEnabled(True)
        self.status.setText(
            "Reprojection error %.3f px over %d views - Save to apply "
            "(under ~1 px is good)" % (rms, len(self.views)))

    def _compute_failed(self, message):
        self.compute_btn.setEnabled(True)
        self.status.setText("Calibration failed: " + message)

    def _save(self):
        rms, matrix, coeffs = self.result
        camera_intrinsics.save_profile(self.device_index, matrix, coeffs,
                                       self.frame_size, rms, self.camera_name)
        self.saved.emit(self.device_index)
        self.accept()

    def done(self, code):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        super().done(code)
