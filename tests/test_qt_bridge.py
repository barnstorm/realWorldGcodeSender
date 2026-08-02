import threading

from qt_ui.bridge import GrblBridge


class FakeSender:
    def __init__(self):
        self.listener = None
        self.busy = False

    def add_event_listener(self, listener):
        self.listener = listener

    def remove_event_listener(self, listener):
        if self.listener == listener:
            self.listener = None

    def is_busy(self):
        return self.busy

    def fire(self, event, *data):
        self.listener(event, *data)


def test_bridge_queues_serial_thread_state_on_qt_thread(qtbot):
    sender = FakeSender()
    bridge = GrblBridge(sender)
    with qtbot.waitSignal(bridge.state_changed, timeout=1000) as signal:
        thread = threading.Thread(
            target=sender.fire,
            args=("on_stateupdate", "Run", (1, 2, 3), (0.5, 1, 2)),
        )
        thread.start()
        thread.join()
    assert signal.args == ["Run", (1.0, 2.0, 3.0), (0.5, 1.0, 2.0)]
    bridge.close()


def test_bridge_detects_version_and_batches_console(qtbot):
    sender = FakeSender()
    bridge = GrblBridge(sender)
    with qtbot.waitSignal(bridge.capabilities_changed, timeout=1000) as signal:
        sender.fire("on_read", "Grbl 1.1h ['$' for help]")
    assert signal.args[0].jog_commands

    with qtbot.waitSignal(bridge.console_lines, timeout=1000) as console:
        sender.fire("on_write", "$J=G91 G20 X0.1 F300")
    assert any("$J=" in line for line in console.args[0])
    bridge.close()


def test_bridge_reports_busy_edges(qtbot):
    sender = FakeSender()
    bridge = GrblBridge(sender)
    sender.busy = True
    with qtbot.waitSignal(bridge.busy_changed, timeout=1000) as signal:
        pass
    assert signal.args == [True]
    bridge.close()

