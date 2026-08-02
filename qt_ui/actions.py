"""Keyboard shortcuts and the always-live recovery keys.

Monolith letters become QActions with WidgetWithChildrenShortcut context:
typing in a QLineEdit naturally suppresses them (Qt's ShortcutOverride), and
clicking the bed restores them. Recovery is un-suppressible two ways: F6/F7/F8
as ApplicationShortcut, and a QApplication event filter that maps space/r/x to
hold/resume/kill whenever focus is NOT a text-entry widget. Recovery actions
are never disabled by the busy state.
"""

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QLineEdit, QPlainTextEdit, QTextEdit


class RecoveryFilter(QObject):
    """App-level filter: space/r/x -> hold/resume/kill unless typing."""

    KEYMAP = {Qt.Key_Space: "feed_hold", Qt.Key_R: "resume", Qt.Key_X: "kill_alarm"}

    def __init__(self, controller, app):
        super().__init__(app)
        self.controller = controller
        self.app = app

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and not event.isAutoRepeat():
            name = self.KEYMAP.get(event.key())
            if name and event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier):
                focus = self.app.focusWidget()
                if not isinstance(focus, (QLineEdit, QPlainTextEdit, QTextEdit)):
                    getattr(self.controller, name)()
                    return True
        return super().eventFilter(obj, event)


def make_action(parent, text, shortcut, slot, context=Qt.WidgetWithChildrenShortcut,
                enabled=True, tooltip=None):
    act = QAction(text, parent)
    if shortcut:
        act.setShortcut(QKeySequence(shortcut))
        act.setShortcutContext(context)
    act.triggered.connect(slot)
    act.setEnabled(enabled)
    if tooltip:
        act.setToolTip(tooltip)
    parent.addAction(act)
    return act


def build_workspace_actions(workspace, controller):
    """The monolith key map, parented to the workspace view. Returns
    {name: QAction}. Machine actions are enabled/disabled by the caller
    based on sender presence; recovery stays enabled always."""
    online = controller.ctx.sender is not None
    off_tip = None if online else "Machine offline — run with --sender"

    acts = {
        "send_gcode": make_action(workspace, "Send G-code file", "G",
                                  controller.send_gcode_file, enabled=online, tooltip=off_tip),
        "send_svg": make_action(workspace, "Send SVG job", "S",
                                controller.send_svg, enabled=online, tooltip=off_tip),
        "send_drawn": make_action(workspace, "Send drawn path", "Shift+C",
                                  controller.send_drawn, enabled=online, tooltip=off_tip),
        "home": make_action(workspace, "Home machine ($H)", "H",
                            controller.home, enabled=online, tooltip=off_tip),
        "touch_off": make_action(workspace, "Touch-off Z", "Z",
                                 controller.touch_off, enabled=online, tooltip=off_tip),
        "z_mesh": make_action(workspace, "Probe Z mesh", "Shift+Z",
                              controller.z_mesh, enabled=online, tooltip=off_tip),
        "jog_to_cursor": make_action(workspace, "Move to cursor", "M",
                                     workspace.jog_to_cursor, enabled=online, tooltip=off_tip),
        "next_path": make_action(workspace, "Select next path", "N", workspace.next_path),
        "prev_path": make_action(workspace, "Select previous path", "P", workspace.prev_path),
        "draw_point": make_action(workspace, "Draw point", "D", workspace.draw_point),
        "draw_arc": make_action(workspace, "Draw arc", "C", workspace.draw_arc),
        "erase_point": make_action(workspace, "Erase last drawn point", "E", workspace.erase_point),
        "erase_all": make_action(workspace, "Erase all drawn points", "Shift+E", workspace.erase_all),
    }
    return acts


def build_recovery_actions(window, controller):
    """Un-suppressible recovery: ApplicationShortcut F-keys (plus the letter
    keys via RecoveryFilter). Never disabled."""
    ctx = Qt.ApplicationShortcut
    return {
        "hold": make_action(window, "Feed hold (!)", "F6", controller.feed_hold, context=ctx),
        "resume": make_action(window, "Resume (~)", "F7", controller.resume, context=ctx),
        "kill": make_action(window, "Kill alarm ($X)", "F8", controller.kill_alarm, context=ctx),
    }
