"""Application composition and startup pipeline for the PySide6 UI."""

import sys
import threading
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)

from app_config import get_config
from qt_ui.actions import RecoveryFilter, build_recovery_actions, build_workspace_actions
from qt_ui.bed_coords import BedTransform
from qt_ui.bridge import GrblBridge
from qt_ui.context import AppContext
from qt_ui.machine_controller import MachineController
from qt_ui.toolpath_loader import load_gcode_file, load_svg, svg_paths_as_tuples
from qt_ui.views import CalibrationView, MachineView, SettingsView, WorkspaceView
from qt_ui.widgets import Console, StatePill
from workpiece_frame import WorkpieceFrame


def capture_and_calibrate(use_camera=False):
    from realWorldGcodeSender import (
        calibrate_bed, capture_bed_image, locate_touch_plate, warp_to_overhead,
    )
    cap, frame = capture_bed_image(use_camera)
    try:
        if frame is None:
            raise RuntimeError("Camera did not return an image")
        try:
            annotated, _bed_to_orig, orig_to_bed, boxes, ids, info = calibrate_bed(frame)
        except Exception as exc:
            # Detection failed: keep the frame so the user can see what the
            # camera saw and re-aim / pick another device.
            return {"valid": False, "error": str(exc), "raw": frame}
        overhead = warp_to_overhead(annotated, orig_to_bed)
        plate = locate_touch_plate(annotated, boxes, ids, orig_to_bed)
        return {"valid": True, "raw": annotated, "bed": overhead,
                "plate": plate, "info": info}
    finally:
        if cap is not None:
            cap.release()


def apply_calibration(context, result):
    context.raw_frame = result.get("raw")
    context.bed_image = result.get("bed")
    context.calibration_info = result.get("info")
    context.touch_plate_px = [tuple(map(float, point)) for point in result.get("plate", [])]
    if context.touch_plate_px:
        context.ref_points = context.transform.set_ref_from_pixels(context.touch_plate_px)
        x, y = context.transform.ref_plate_measured
        context.workpiece_frame = WorkpieceFrame.eyeballed(x, y)


def load_toolpath(context, path, force_svg=False, force_gcode=False):
    path = str(Path(path).resolve())
    svg_mode = force_svg or (not force_gcode and Path(path).suffix.lower() == ".svg")
    if svg_mode:
        cnc = load_svg(path, context.config.cutting_parameters.cutter_diameter)
        context.svg = cnc
        context.svg_file = path
        context.svg_paths, context.svg_colors = svg_paths_as_tuples(cnc)
        context.path_offsets = [[0.0, 0.0] for _ in context.svg_paths]
        context.gcode_file = None
        context.gcode_points = []
        context.gcode_powers = []
    else:
        context.gcode_points, context.gcode_powers = load_gcode_file(path)
        context.gcode_file = path
        context.svg = None
        context.svg_file = None
        context.svg_paths = []
        context.svg_colors = []
        context.path_offsets = []
    context.path_index = -1


class PipelineSignals(QObject):
    completed = Signal(object)
    failed = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self, context, use_camera=False):
        super().__init__()
        self.ctx = context
        self.use_camera = use_camera
        self.controller = MachineController(context)
        self.bridge = GrblBridge(context.sender, self)
        self._pipeline_signals = None
        self._pipeline_thread = None
        self.setWindowTitle("realWorldGcodeSender")
        self.setMinimumSize(1000, 680)

        self.workspace = WorkspaceView(context, self.controller)
        self.calibration = CalibrationView(context)
        self.machine = MachineView(context, self.controller, self.workspace.bed.scene())
        self.settings = SettingsView(context)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.calibration, "Calibration")
        self.tabs.addTab(self.workspace, "Workspace")
        self.tabs.addTab(self.machine, "Machine")
        self.tabs.addTab(self.settings, "Settings")
        self.tabs.setCurrentIndex(1)

        self.console = Console()
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.console)
        self.setCentralWidget(central)
        self._build_menu()
        self._build_status()
        self._wire()
        self._refresh_views()

    def _build_menu(self):
        menus = {name: self.menuBar().addMenu(name) for name in
                 ("File", "Edit", "View", "Machine", "Probe", "Help")}
        workspace_actions = build_workspace_actions(self.workspace, self.controller)
        recovery = build_recovery_actions(self, self.controller)
        menus["File"].addAction(workspace_actions["send_gcode"])
        menus["File"].addAction(workspace_actions["send_svg"])
        for key in ("home", "jog_to_cursor"):
            menus["Machine"].addAction(workspace_actions[key])
        for action in recovery.values():
            menus["Machine"].addAction(action)
        for key in ("touch_off", "z_mesh"):
            menus["Probe"].addAction(workspace_actions[key])
        self._recovery_filter = RecoveryFilter(self.controller, QApplication.instance())
        QApplication.instance().installEventFilter(self._recovery_filter)

    def _build_status(self):
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.connection = QLabel("Offline")
        self.status_message = QLabel("Ready")
        self.state = StatePill("OFFLINE")
        bar.addWidget(self.connection)
        bar.addWidget(QLabel("inches"))
        bar.addWidget(self.status_message, 1)
        bar.addPermanentWidget(self.state)

    def _wire(self):
        self.workspace.request_open.connect(self.open_toolpath)
        self.workspace.request_recapture.connect(self.recapture)
        self.calibration.request_recapture.connect(self.recapture)
        self.calibration.accepted.connect(lambda: self.tabs.setCurrentIndex(1))
        self.calibration.select_camera(
            self.ctx.config.vision_settings.camera_device_index
            if self.use_camera else None)
        self.calibration.camera_changed.connect(self._camera_selected)
        self.calibration.rotation_changed.connect(
            lambda degrees: self.status_message.setText(
                "Camera rotation %d\N{DEGREE SIGN} CW -- Recapture to apply" % degrees))
        self.settings.applied.connect(self._refresh_views)
        self.controller.message.connect(self._message)
        self.bridge.console_lines.connect(self.console.append_lines)
        self.bridge.state_changed.connect(self._state_changed)
        self.bridge.progress.connect(self.workspace.set_progress)
        self.bridge.busy_changed.connect(self.workspace.set_busy)
        self.bridge.capabilities_changed.connect(self._capabilities)
        self.bridge.booted.connect(lambda version: self._connected("GRBL " + version))
        self.bridge.disconnected.connect(self._disconnected)
        self.bridge.disconnected.connect(self.machine.disconnected)
        self.bridge.alarm.connect(lambda message: self._alarm(message))
        self.bridge.grbl_error.connect(lambda message: self._alarm(message))
        self.bridge.probe_hit.connect(self._probe_hit)

    def _refresh_views(self):
        self.workspace.load_context()
        self.calibration.show_result(self.ctx.bed_image is not None,
                                     "No valid calibration",
                                     self.ctx.calibration_info)

    def show_calibration_result(self, valid, message="", info=None):
        self.ctx.calibration_info = info
        self.calibration.show_result(valid, message, info)

    def open_toolpath(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open toolpath", "", "Toolpaths (*.svg *.gcode *.nc);;All files (*)")
        if not path:
            return
        try:
            load_toolpath(self.ctx, path)
            self.workspace.load_context()
        except Exception as exc:
            QMessageBox.critical(self, "Could not load toolpath", str(exc))

    def recapture(self):
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            return
        self.status_message.setText("Capturing and calibrating...")
        self._pipeline_signals = PipelineSignals(self)
        self._pipeline_signals.completed.connect(self._pipeline_done)
        self._pipeline_signals.failed.connect(self._pipeline_failed)

        def worker():
            try:
                self._pipeline_signals.completed.emit(capture_and_calibrate(self.use_camera))
            except Exception as exc:
                self._pipeline_signals.failed.emit(str(exc))

        self._pipeline_thread = threading.Thread(target=worker, daemon=True)
        self._pipeline_thread.start()

    def _camera_selected(self, device_index):
        if device_index is None:
            self.use_camera = False
            self.status_message.setText("Using saved test image -- Recapture to reload")
        else:
            self.use_camera = True
            self.ctx.config.vision_settings.camera_device_index = int(device_index)
            self.status_message.setText(
                "Camera set to %s -- Recapture to grab a frame"
                % self.calibration.camera.currentText())

    def _pipeline_done(self, result):
        if not result.get("valid"):
            if result.get("raw") is not None:
                self.ctx.raw_frame = result["raw"]
            self._pipeline_failed(result.get("error", "detection failed"))
            return
        apply_calibration(self.ctx, result)
        self._refresh_views()
        self.status_message.setText("Calibration accepted")

    def _pipeline_failed(self, message):
        self.calibration.show_result(False, message)
        self.tabs.setCurrentIndex(0)
        self.status_message.setText("Calibration invalid")
        self.state.set_state("DETECTION FAILED", True)

    def _message(self, message):
        self.status_message.setText(message)
        self.console.append_lines(["> " + message])

    def _state_changed(self, mode, mpos, wpos):
        self.workspace.update_state(mode, mpos, wpos)
        self.machine.update_state(mode, mpos, wpos)
        self.state.set_state(mode.upper(), mode.lower() == "alarm")

    def _capabilities(self, caps):
        self.ctx.grbl_caps = caps
        self.machine.set_capabilities(caps)

    def _connected(self, label):
        port = self.ctx.config.communication_settings.com_port
        self.connection.setText("Connected - %s - %s" % (port, label))

    def _disconnected(self):
        self.connection.setText("Offline")
        self.state.set_state("OFFLINE", True)

    def _alarm(self, message):
        self.status_message.setText(message)
        self.state.set_state("ALARM", True)
        self.console.append_lines(["! " + message])

    def _probe_hit(self, _position):
        done = getattr(self, "_probe_done", 0) + 1
        self._probe_done = done
        points = []
        for target in self.ctx.probe_targets:
            xy = target[:2] if isinstance(target, (tuple, list)) else (target.x, target.y)
            points.append(self.ctx.transform.phy_to_pixels(*xy))
        self.workspace.bed.set_probe_targets(points, done)

    def closeEvent(self, event):
        self.bridge.close()
        if self.ctx.sender is not None:
            self.ctx.sender.gerbil.disconnect()
        super().closeEvent(event)


def build_application(path, force_svg=False, force_gcode=False, use_camera=False,
                      enable_sender=False):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("realWorldGcodeSender")
    app.setFont(QFont("Lora", 10))
    qss = Path(__file__).resolve().parent / "resources" / "classical.qss"
    app.setStyleSheet(qss.read_text(encoding="utf-8"))

    config = get_config()
    context = AppContext(config, BedTransform.from_config(config))
    calibration_error = None
    calibration_result = None
    try:
        calibration_result = capture_and_calibrate(use_camera)
    except Exception as exc:
        calibration_error = str(exc)
    if calibration_result and calibration_result.get("valid"):
        apply_calibration(context, calibration_result)
    else:
        if calibration_result is not None:
            calibration_error = calibration_result.get("error", "detection failed")
            context.raw_frame = calibration_result.get("raw")
        if context.raw_frame is None:
            fallback = cv2.imread(str(Path(__file__).resolve().parent.parent / "cnc13.jpg"))
            if fallback is not None:
                context.raw_frame = fallback
                context.bed_image = cv2.resize(fallback, (int(context.transform.bed_view_pixels),) * 2)

    if path:
        load_toolpath(context, path, force_svg, force_gcode)
    if enable_sender:
        try:
            from realWorldGcodeSender import GCodeSender
            context.sender = GCodeSender()
        except Exception as exc:
            calibration_error = ((calibration_error + "\n") if calibration_error else "") + \
                                "Machine connection failed: " + str(exc)

    window = MainWindow(context, use_camera)
    if calibration_error:
        window.show_calibration_result(False, calibration_error)
    return app, window

