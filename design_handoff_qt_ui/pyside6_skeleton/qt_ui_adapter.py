"""Adapter that plugs the Qt UI into the existing monolith.

Drop this into the repo as `ui/interfaces/qt_ui.py`. It implements the same
`UIInterface` contract as `ui/interfaces/matplotlib_ui.py`, so the app can select
Qt without touching the capture/calibrate/warp/overlay pipeline in
`realWorldGcodeSender.py`.

This file is a STUB showing the seams — the method bodies are where you connect
the skeleton widgets to the backend. It falls back to a local base class so it can
be read/linted outside the repo, but in the repo you should
`from ui.interfaces.base import UIInterface`.
"""
from __future__ import annotations

import sys
from typing import Optional, List, Tuple

try:  # in-repo
    from ui.interfaces.base import UIInterface  # type: ignore
except Exception:  # standalone reading
    class UIInterface:  # minimal shim
        def __init__(self, config: dict):
            self.config = config
            self.is_running = False


class QtUI(UIInterface):
    """PySide6 implementation of UIInterface."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._app = None
        self._win = None

    # -- lifecycle ------------------------------------------------------------
    def initialize(self) -> bool:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QFont
        from pathlib import Path
        from main_window import MainWindow  # skeleton module

        self._app = QApplication.instance() or QApplication(sys.argv)
        qss = Path(__file__).with_name("classical.qss")
        if qss.exists():
            self._app.setStyleSheet(qss.read_text(encoding="utf-8"))
        self._app.setFont(QFont("Lora", 10))
        self._win = MainWindow()
        self._win.resize(1300, 824)
        self.is_running = True
        return True

    def run(self) -> None:
        if not self.is_running and not self.initialize():
            return
        self._win.show()
        self._app.exec()

    def stop(self) -> None:
        self.is_running = False
        if self._win:
            self._win.close()

    # -- display --------------------------------------------------------------
    def display_image(self, image, window_name: str = "main") -> None:
        """Convert the cv2/np overhead frame to a QPixmap and set it on the
        active BedView. e.g.
            h, w = image.shape[:2]
            qimg = QImage(image.data, w, h, 3 * w, QImage.Format_BGR888)
            bed_view.set_pixmap(QPixmap.fromImage(qimg))
        """
        raise NotImplementedError

    def display_path(self, path_points: List[Tuple[float, float]],
                     window_name: str = "main") -> None:
        """Draw the G-code overlay as a QGraphicsPathItem on the BedView scene."""
        raise NotImplementedError

    # -- live telemetry (poll grbl ~5-10 Hz) ----------------------------------
    def update_status(self, status: dict) -> None:
        """Push WPos/MPos, state, overrides, job progress into the DRO, router
        marker (bed_view.set_router_pos(x, y)), progress panel and status bar."""
        raise NotImplementedError

    # -- dialogs --------------------------------------------------------------
    def show_message(self, message: str, message_type: str = "info") -> None:
        raise NotImplementedError

    def show_dialog(self, title: str, message: str,
                    dialog_type: str = "info") -> Optional[str]:
        from PySide6.QtWidgets import QMessageBox
        if dialog_type == "question":
            r = QMessageBox.question(self._win, title, message)
            return "yes" if r == QMessageBox.Yes else "no"
        QMessageBox.information(self._win, title, message)
        return "ok"

    def get_file_path(self, title: str = "Select File", file_types=None,
                      save: bool = False):
        from PySide6.QtWidgets import QFileDialog
        from pathlib import Path
        if save:
            name, _ = QFileDialog.getSaveFileName(self._win, title)
        else:
            name, _ = QFileDialog.getOpenFileName(self._win, title)
        return Path(name) if name else None

    def create_settings_panel(self, settings: dict):
        """The SettingsView already renders app_config fields; bind them here."""
        return None


# Wiring notes
# ------------
# * Buttons/keys -> GCodeSender: home_machine, absolute_move, work_offset_move,
#   set_inches/set_mm, gerbil.hold/resume/killalarm; send via send_svf/send_file/
#   send_drawnPoints (warp with toolpath_warp.warp_gcode_lines when frame.z.is_mesh).
# * Probe buttons -> probing.get_strategy("z_touch_off"|"z_mesh"|"edge_refine")
#   with propose_z_targets / propose_edge_targets and GCodeSenderMachine.
# * Long ops -> run on a QThread worker with signals (mirror GCodeSender.run_async)
#   so the UI never blocks; keep space/r/x (feed-hold/resume/kill) always live.
# * Settings -> app_config.get_config()/save_config(); mark a dirty flag for the
#   "Unsaved changes" indicator.
