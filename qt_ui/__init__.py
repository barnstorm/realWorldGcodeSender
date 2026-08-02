"""PySide6 UI for realWorldGcodeSender.

Backed by the monolith (GCodeSender + the capture/calibrate/warp pipeline in
realWorldGcodeSender.py), per PROBING_DESIGN.md's one-codebase decision. The
matplotlib UI remains alongside; this package must never import OverlayGcode
or create matplotlib figures. Design source of truth: design_handoff_qt_ui/.
"""
