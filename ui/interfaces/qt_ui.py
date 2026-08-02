"""UIInterface adapter for the native PySide6 application."""

from pathlib import Path

from ui.interfaces.base import UIInterface


class QtUI(UIInterface):
    def __init__(self, config=None):
        super().__init__(config or {})
        self._app = None
        self._window = None

    def initialize(self):
        from qt_ui.app import build_application
        self._app, self._window = build_application(
            self.config.get("file", "puzzles2.svg"),
            force_svg=self.config.get("svg", False),
            force_gcode=self.config.get("gcode", False),
            use_camera=self.config.get("live", False),
            enable_sender=self.config.get("sender", False),
        )
        self._window.resize(1300, 824)
        self.is_running = True
        return True

    def run(self):
        if not self.is_running and not self.initialize():
            return
        self._window.show()
        self._app.exec()

    def stop(self):
        self.is_running = False
        if self._window is not None:
            self._window.close()

    def display_image(self, image, window_name="main"):
        if self._window is None:
            return
        self._window.ctx.bed_image = image.copy()
        self._window.workspace.refresh_overlay()

    def display_path(self, path_points, window_name="main"):
        if self._window is None:
            return
        self._window.ctx.drawn_points = [tuple(point[:2]) for point in path_points]
        self._window.workspace.refresh_overlay()

    def show_message(self, message, message_type="info"):
        if self._window is not None:
            self._window._message(str(message))

    def show_dialog(self, title, message, dialog_type="info"):
        from PySide6.QtWidgets import QMessageBox
        if dialog_type == "question":
            result = QMessageBox.question(self._window, title, message)
            return "yes" if result == QMessageBox.Yes else "no"
        method = QMessageBox.warning if dialog_type == "warning" else QMessageBox.information
        method(self._window, title, message)
        return "ok"

    def get_file_path(self, title="Select File", file_types=None, save=False):
        from PySide6.QtWidgets import QFileDialog
        filters = ";;".join("%s (*.%s)" % (name, ext.lstrip("*."))
                            for name, ext in (file_types or []))
        method = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
        filename, _ = method(self._window, title, "", filters)
        return Path(filename) if filename else None

    def update_status(self, status):
        if self._window is None:
            return
        mode = status.get("mode", "Idle")
        mpos = tuple(status.get("mpos", (0.0, 0.0, 0.0)))
        wpos = tuple(status.get("wpos", (0.0, 0.0, 0.0)))
        self._window._state_changed(mode, mpos, wpos)
        if "progress" in status:
            self._window.workspace.set_progress(int(status["progress"]))

    def create_settings_panel(self, settings):
        return self._window.settings if self._window is not None else None

