"""Native Qt entry point; the matplotlib application remains unchanged."""

import argparse
import os
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Qt UI for realWorldGcodeSender")
    parser.add_argument("file", nargs="?", default="puzzles2.svg")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--svg", action="store_true", help="Force SVG input mode")
    mode.add_argument("--gcode", action="store_true", help="Force G-code input mode")
    parser.add_argument("--live", action="store_true", help="Capture from the configured camera")
    parser.add_argument("--sender", action="store_true", help="Connect to the configured GRBL controller")
    parser.add_argument("--smoke", action="store_true", help="Render every tab and exit")
    parser.add_argument("--smoke-dir", default="qt_smoke")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import matplotlib
    matplotlib.use("Agg")
    from PySide6.QtCore import QTimer
    from qt_ui.app import build_application

    app, window = build_application(args.file, args.svg, args.gcode,
                                    args.live, args.sender)
    window.resize(1300, 824)
    window.show()
    if args.smoke:
        output = Path(args.smoke_dir)
        output.mkdir(parents=True, exist_ok=True)

        def capture_tabs():
            for index, name in enumerate(("calibration", "workspace", "machine", "settings")):
                window.tabs.setCurrentIndex(index)
                app.processEvents()
                window.grab().save(str(output / (name + ".png")))
            window.close()
            app.quit()

        QTimer.singleShot(250, capture_tabs)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

