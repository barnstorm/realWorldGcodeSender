"""The single object allowed to talk to the machine.

Ports every monolith key handler with identical guard semantics: operations
that inject motion are busy-guarded via sender.is_busy(); recovery (hold /
resume / kill alarm) is NEVER guarded. With no sender (offline mode) every
machine call no-ops with a status message -- vision/placement still works.

Deviation from the monolith (deliberate, noted in the plan): the 's' handler
mutated cncPaths in place before sending; here a deep copy is transformed and
sent, so re-sending can never double-apply offsets.

The Qt UI never reads sender.respQueue or calls get_absolute_pos() -- live
telemetry arrives via GrblBridge (M2) observing gerbil events.
"""

import math
from copy import deepcopy

from PySide6.QtCore import QObject, Signal


class MachineController(QObject):
    message = Signal(str)          # one-line operator feedback (status bar + console)

    def __init__(self, context):
        super().__init__()
        self.ctx = context

    # -- guards -------------------------------------------------------------

    def _sender(self):
        if self.ctx.sender is None:
            self.message.emit("Machine offline — restart with --sender to connect")
            return None
        return self.ctx.sender

    def _idle_sender(self, action):
        sender = self._sender()
        if sender is None:
            return None
        if sender.is_busy():
            self.message.emit("CNC busy; ignoring %s" % action)
            return None
        return sender

    # -- recovery (never guarded) -------------------------------------------

    def feed_hold(self):
        sender = self._sender()
        if sender:
            self.message.emit("FEED HOLD (!)")
            sender.gerbil.hold()

    def resume(self):
        sender = self._sender()
        if sender:
            self.message.emit("Resume (~)")
            sender.gerbil.resume()

    def kill_alarm(self):
        sender = self._sender()
        if sender:
            self.message.emit("Kill alarm ($X)")
            sender.gerbil.killalarm()

    # -- motion / jobs (busy-guarded, port of onkeypress) --------------------

    def home(self):
        sender = self._idle_sender("home")
        if sender:
            self.message.emit("Homing ($H)")
            sender.home_machine()

    def unlock(self):
        sender = self._sender()
        if sender:
            sender.gerbil.killalarm()

    def zero_axis(self, axis):
        sender = self._idle_sender("zero %s" % axis)
        if sender:
            sender.set_cur_pos_as(**{axis.lower(): 0.0})

    def jog_relative(self, dx=0.0, dy=0.0, dz=0.0, feed=300.0):
        sender = self._idle_sender("jog")
        if not sender:
            return
        caps = self.ctx.grbl_caps
        if caps is None:
            self.message.emit("Waiting for controller version before jogging")
            return
        words = "".join(" %s%.5f" % (axis, value) for axis, value in
                        (("X", dx), ("Y", dy), ("Z", dz)) if value)
        if not words:
            return
        if caps.jog_commands:
            sender.gerbil.send_immediately("$J=G91 G20%s F%.1f" % (words, feed))
        else:
            # GRBL 0.9 has no jog mode.  Keep each move bounded and restore G90
            # immediately so an interrupted jog cannot leak incremental mode.
            sender.gerbil.send_immediately("G91 G20")
            sender.gerbil.send_immediately("G1%s F%.1f" % (words, feed))
            sender.gerbil.send_immediately("G90")

    def cancel_jog(self):
        sender = self._sender()
        if sender and self.ctx.grbl_caps and self.ctx.grbl_caps.jog_commands:
            sender.gerbil.send_realtime(0x85)

    def adjust_override(self, kind, direction):
        sender = self._sender()
        caps = self.ctx.grbl_caps
        if not sender or caps is None:
            return
        if not caps.realtime_overrides:
            if kind == "feed":
                sender.gerbil.set_feed_override(True)
                current = getattr(sender.gerbil.preprocessor, "request_feed", 100.0)
                sender.gerbil.request_feed(max(1.0, float(current) + direction * 10.0))
            else:
                self.message.emit("%s override requires GRBL 1.1+" % kind.title())
            return
        commands = {"feed": (0x92, 0x91), "spindle": (0x9B, 0x9A)}
        if kind == "rapid":
            sender.gerbil.send_realtime(0x95 if direction > 0 else 0x97)
        elif kind in commands:
            sender.gerbil.send_realtime(commands[kind][1 if direction > 0 else 0])

    def jog_to(self, x_in, y_in):
        """'m' key: absolute move to a bed position."""
        sender = self._idle_sender("move")
        if sender:
            self.message.emit("Move to %.3f, %.3f" % (x_in, y_in))
            sender.absolute_move(x_in, y_in, feed=300)

    def send_gcode_file(self):
        """'g' key."""
        ctx = self.ctx
        sender = self._idle_sender("send gcode")
        if not sender or not ctx.gcode_file:
            return
        self.message.emit("Sending %s" % ctx.gcode_file)
        sender.run_async("send_file", sender.send_file, ctx.gcode_file,
                         ctx.x_offset, ctx.y_offset,
                         ctx.rotation * math.pi / 180.0)

    def send_svg(self):
        """'s' key -- transforms a COPY of the pristine paths (monolith
        mutated in place; same numbers, no double-apply hazard)."""
        ctx = self.ctx
        sender = self._idle_sender("send svg")
        if not sender or ctx.svg is None:
            return
        rotation = ctx.rotation * math.pi / 180.0
        origin = list(ctx.path_offsets[-1]) if ctx.path_offsets else [0.0, 0.0]
        cnc = deepcopy(ctx.svg)
        for path, offset in zip(cnc.cncPaths, ctx.path_offsets):
            for p in path.points3D:
                p.X, p.Y = p.X + offset[0], p.Y + offset[1]
            for p in path.points3D:
                px = origin[0] + math.cos(rotation) * (p.X - origin[0]) - math.sin(rotation) * (p.Y - origin[1])
                py = origin[1] + math.sin(rotation) * (p.X - origin[0]) + math.cos(rotation) * (p.Y - origin[1])
                p.X, p.Y = px, py
        self.message.emit("Sending SVG job")
        sender.run_async("send_svf", sender.send_svf, cnc, ctx.workpiece_frame)

    def send_drawn(self):
        """'C' key."""
        ctx = self.ctx
        sender = self._idle_sender("send drawn points")
        if not sender or not ctx.drawn_points:
            return
        from svgToGCode import Point3D
        rx, ry = ctx.transform.ref_plate_measured
        offset = Point3D(-rx, -ry)
        points = [Point3D(x, y, 0) for x, y in ctx.drawn_points]
        self.message.emit("Sending drawn path")
        sender.run_async("send_drawnPoints", sender.send_drawnPoints,
                         offset, points, ctx.workpiece_frame)

    # -- probing (port of 'z' / 'Z') ----------------------------------------

    def touch_off(self):
        ctx = self.ctx
        sender = self._idle_sender("touch-off")
        if not sender:
            return
        if not ctx.ref_points:
            self.message.emit("No touch plate located in the image")
            return
        from workpiece_frame import Measured, Source, WorkpieceFrame, ZSurface
        ref_points = [list(p) for p in ctx.ref_points]
        self.message.emit("Probing touch plate (Z)")

        def zero_and_frame():
            sender.zero_on_refPlate(ref_points, True)
            ctx.workpiece_frame = WorkpieceFrame(
                x=Measured(0.0, Source.PROBE, tolerance=0.002),
                y=Measured(0.0, Source.PROBE, tolerance=0.002),
                angle=Measured(0.0, Source.PROBE),
                z=ZSurface(nominal=Measured(0.0, Source.PROBE, tolerance=0.001)))
        sender.run_async("zero_on_refPlate", zero_and_frame)

    def propose_mesh_targets(self):
        """Preview step: propose and remember z-mesh targets (drawn on the
        bed before the user commits to probing)."""
        from probing.targets import propose_z_targets
        paths = self.ctx.work_coord_paths()
        if not paths:
            self.message.emit("No cut paths loaded or drawn; nothing to mesh-probe")
            return []
        self.ctx.probe_targets = propose_z_targets(paths)
        return self.ctx.probe_targets

    def z_mesh(self):
        ctx = self.ctx
        sender = self._idle_sender("Z mesh")
        if not sender:
            return
        targets = ctx.probe_targets or self.propose_mesh_targets()
        if not targets:
            return
        from workpiece_frame import WorkpieceFrame
        from probing.base import get_strategy
        from realWorldGcodeSender import GCodeSenderMachine
        import probing.strategies  # noqa: F401  (registers strategies)
        if ctx.workpiece_frame is None:
            ctx.workpiece_frame = WorkpieceFrame.eyeballed(0.0, 0.0)
        self.message.emit("Probing %d-point Z mesh" % len(targets))

        def run_mesh():
            sender.flushGcodeRespQue()
            machine = GCodeSenderMachine(sender)
            ctx.workpiece_frame = get_strategy("z_mesh").refine(
                ctx.workpiece_frame, machine, targets)
        sender.run_async("z_mesh", run_mesh)
