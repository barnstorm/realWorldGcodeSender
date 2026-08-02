"""Thread-safe, observer-only bridge from GCodeSender events into Qt."""

from collections import deque

from PySide6.QtCore import QObject, QTimer, Signal

from qt_ui.grbl_caps import parse_boot_banner


class GrblBridge(QObject):
    state_changed = Signal(str, object, object)
    progress = Signal(int)
    alarm = Signal(str)
    grbl_error = Signal(str)
    probe_hit = Signal(object)
    booted = Signal(str)
    disconnected = Signal()
    capabilities_changed = Signal(object)
    console_lines = Signal(object)
    busy_changed = Signal(bool)

    _NOISY = {"on_write", "on_line_sent", "on_bufsize_change", "on_read"}

    def __init__(self, sender=None, parent=None):
        super().__init__(parent)
        self.sender = sender
        self.capabilities = None
        self._console = deque(maxlen=5000)
        self._busy = False
        self._report_scale = 1.0

        self._console_timer = QTimer(self)
        self._console_timer.setInterval(100)
        self._console_timer.timeout.connect(self._drain_console)
        self._console_timer.start()

        self._job_timer = QTimer(self)
        self._job_timer.setInterval(200)
        self._job_timer.timeout.connect(self._poll_job)
        self._job_timer.start()

        if sender is not None:
            sender.add_event_listener(self.handle_event)

    def close(self):
        if self.sender is not None:
            self.sender.remove_event_listener(self.handle_event)

    def _queue_console(self, event, data):
        if event == "on_read" and data:
            self._console.append("< " + str(data[0]))
        elif event == "on_write" and data:
            self._console.append("> " + str(data[0]).rstrip())
        elif event == "on_line_sent" and len(data) >= 2:
            self._console.append("> " + str(data[1]).rstrip())

    def handle_event(self, event, *data):
        """May be called on Gerbil's serial thread; emit copied values only."""
        self._queue_console(event, data)
        if event == "on_read" and data:
            caps = parse_boot_banner(str(data[0]))
            if caps is not None:
                self.capabilities = caps
                self.capabilities_changed.emit(caps)
                self.booted.emit(caps.version)
        elif event == "on_stateupdate" and len(data) >= 3:
            mode = str(data[0])
            mpos = tuple(float(v) * self._report_scale for v in tuple(data[1]))
            wpos = tuple(float(v) * self._report_scale for v in tuple(data[2]))
            self.state_changed.emit(mode, mpos, wpos)
        elif event == "on_progress_percent" and data:
            self.progress.emit(int(data[0]))
        elif event == "on_alarm" and data:
            self.alarm.emit(str(data[0]))
        elif event == "on_error" and data:
            self.grbl_error.emit(str(data[0]))
        elif event == "on_probe" and data:
            self.probe_hit.emit(tuple(data[0]))
        elif event == "on_disconnected":
            self.disconnected.emit()
        elif event == "on_settings_downloaded" and data and isinstance(data[0], dict):
            setting = data[0].get(13, data[0].get("13"))
            if isinstance(setting, dict):
                setting = setting.get("val")
            if setting is not None:
                self._report_scale = 1.0 if int(float(setting)) else 1.0 / 25.4

    def _drain_console(self):
        if not self._console:
            return
        lines = []
        while self._console and len(lines) < 250:
            lines.append(self._console.popleft())
        self.console_lines.emit(lines)

    def _poll_job(self):
        busy = bool(self.sender and self.sender.is_busy())
        if busy != self._busy:
            self._busy = busy
            self.busy_changed.emit(busy)

