"""Main window: menu bar, workspace tabs, status bar, and the four views."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QHBoxLayout, QLabel, QStatusBar,
)

from views import WorkspaceView, CalibrationView, MachineView, SettingsView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("realWorldGcodeSender")
        self._build_menu()

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        # order matches the design tab strip
        tabs.addTab(CalibrationView(), "Calibration")
        tabs.addTab(WorkspaceView(), "Workspace")
        tabs.addTab(MachineView(), "Machine")
        tabs.addTab(SettingsView(), "Settings")
        tabs.setCurrentIndex(1)  # Workspace is the default
        self.setCentralWidget(tabs)

        self._build_status_bar()

    def _build_menu(self):
        for name in ("File", "Edit", "View", "Machine", "Probe", "Help"):
            self.menuBar().addMenu(name)

    def _build_status_bar(self):
        bar = QStatusBar()
        self.setStatusBar(bar)
        left = QWidget()
        h = QHBoxLayout(left)
        h.setContentsMargins(8, 0, 8, 0)
        h.setSpacing(18)
        h.addWidget(QLabel("\u25cf Connected - COM3"))
        h.addWidget(QLabel("inches"))
        h.addWidget(QLabel("Job 458 / 1,204"))
        bar.addWidget(left)
        state = QLabel("RUN")
        state.setProperty("role", "state")
        bar.addPermanentWidget(state)
