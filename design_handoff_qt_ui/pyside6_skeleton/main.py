"""Entry point for the Classical Qt UI skeleton.

    pip install -r requirements.txt
    python main.py

Shows the full app shell (menu, workspace tabs, docks/panels, status bar) styled
with classical.qss. Panels are placeholder scaffold — see qt_ui_adapter.py and the
handoff README for how to wire them to the real codebase.
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from main_window import MainWindow


def load_stylesheet() -> str:
    qss = Path(__file__).resolve().parent / "classical.qss"
    return qss.read_text(encoding="utf-8") if qss.exists() else ""


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("realWorldGcodeSender")
    app.setStyleSheet(load_stylesheet())
    app.setFont(QFont("Lora", 10))

    win = MainWindow()
    win.resize(1300, 824)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
