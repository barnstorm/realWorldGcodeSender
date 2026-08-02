# PySide6 skeleton — realWorldGcodeSender Qt UI

A runnable scaffold of the Qt UI: the app shell (menu, workspace tabs, docks,
status bar), the Classical theme as a Qt Style Sheet, placeholder panels for every
screen, and an adapter stub that plugs into the existing monolith. **Structure and
styling only — no hardware wiring.**

## Run
```bash
pip install -r requirements.txt
python main.py
```
Opens the 1300×824 shell with the four tabs (Calibration · Workspace · Machine ·
Settings), styled by `classical.qss`. For the intended type, install the
**Cormorant Garamond** and **Lora** fonts (otherwise Qt falls back to a default
serif). The bed views load `../screens/ref/*.jpg|png` if present.

## Files
| File | Role |
|---|---|
| `main.py` | entry point — QApplication, loads QSS, shows the window |
| `classical.qss` | Classical tokens → Qt Style Sheet (colors, type, tabs, buttons, inputs) |
| `main_window.py` | `QMainWindow` — menu bar, `QTabWidget` of views, status bar |
| `views.py` | `WorkspaceView`, `CalibrationView`, `MachineView`, `SettingsView` |
| `widgets.py` | helpers — `Dock`, `Panel`, `Field`, `Dro`, `JogPad`, `BedView`, buttons |
| `qt_ui_adapter.py` | `QtUI(UIInterface)` stub — how it plugs into the repo pipeline |

## What to build next (wiring)
The skeleton deliberately stops at the UI. To make it live:
1. Copy `qt_ui_adapter.py` into the repo as `ui/interfaces/qt_ui.py`; import the
   real `UIInterface`.
2. Feed `warp_to_overhead()` output to `BedView` as a `QPixmap`; draw the G-code
   path + probe targets as `QGraphicsItem`s.
3. Poll grbl status (~5–10 Hz) and push into the DRO, router marker
   (`BedView.set_router_pos`), job-progress panel and status bar.
4. Connect toolbar/keys to `GCodeSender` and the probing strategies; run long ops
   on a `QThread` worker (mirror `GCodeSender.run_async`) and keep the recovery
   keys (`space`/`r`/`x`) live.
5. Bind the Settings form to `app_config` (`get_config`/`save_config`), with a
   dirty flag for the "Unsaved changes" indicator.

See the parent `../README.md` for the full screen specs, tokens, and behavior.

## Note
This scaffold favors clarity over completeness: side panels are laid out with
fixed-width `Dock` frames rather than `QDockWidget`s (simpler for a tabbed shell).
Swap to `QDockWidget` if you want detachable/rearrangeable docks.
