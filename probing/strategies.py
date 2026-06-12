"""Concrete probe strategies (refiners).

Each consumes vision-seeded targets and returns a higher-fidelity frame. They
share the `frame -> better frame` contract and depend only on the `Machine`
protocol, so a fake `Machine` (a callable returning canned contacts) simulates
them with no hardware.
"""

import math
from dataclasses import replace
from typing import List

from workpiece_frame import Measured, Source, WorkpieceFrame, ZSurface
from probing.base import Machine, ProbeStrategy, ProbeTarget, register


@register("z_touch_off")
class ZTouchOff(ProbeStrategy):
    """Single-point Z touch-off. Owns z.nominal only. Assumes flat, level stock.

    This is the everyday 'set Z0 on the real surface' case -- the refined version
    of typing in the material thickness.
    """

    def __init__(self, max_drop: float = -2.75, feed: float = 5.9):
        self.max_drop = max_drop
        self.feed = feed

    def refine(self, frame: WorkpieceFrame, machine: Machine,
               targets: List[ProbeTarget]) -> WorkpieceFrame:
        # Probe straight down at the current XY (or the first target, if given).
        if targets:
            t = targets[0]
            mx, my = frame.to_machine_xy(t.x, t.y)
            machine.move(x=mx, y=my, feed=180)
        contact = machine.probe(z=self.max_drop, feed=self.feed)
        z = contact[2]
        return frame.with_z_nominal(Measured(z, Source.PROBE, tolerance=0.001))


@register("z_mesh")
class ZMesh(ProbeStrategy):
    """Multi-point Z surface map. Owns z.samples. Probes each vision-proposed
    target and records (x, y, z) in workpiece coordinates; the toolpath layer
    later warps cut Z to follow ZSurface.z_at()."""

    def __init__(self, clearance: float = 0.1, max_drop: float = -2.75,
                 feed: float = 5.9):
        self.clearance = clearance
        self.max_drop = max_drop
        self.feed = feed

    def refine(self, frame: WorkpieceFrame, machine: Machine,
               targets: List[ProbeTarget]) -> WorkpieceFrame:
        samples = []
        for t in targets:
            if t.kind != "z":
                continue
            mx, my = frame.to_machine_xy(t.x, t.y)
            machine.move(x=mx, y=my, feed=180)
            contact = machine.probe(z=self.max_drop, feed=self.feed)
            samples.append((t.x, t.y, contact[2]))
            machine.move(z=contact[2] + self.clearance, feed=180)
        if not samples:
            return frame
        return replace(frame, z=replace(frame.z, samples=samples))


@register("edge_refine")
class EdgeRefine(ProbeStrategy):
    """Refine vision's XY origin / angle by touching the real workpiece edge.

    Owns frame.x / frame.y / frame.angle. Vision seeds *where* the edge is to
    within a camera pixel; this drops in beside it and refines to machine
    resolution along each target's `approach` normal, correcting for cutter
    radius at the tool tip.

    STUB: the geometry (mapping touched edges back onto origin + angle, handling
    one- vs two-edge corner cases) is the open algorithmic piece -- see
    PROBING_DESIGN.md. Left unimplemented rather than shipped half-correct,
    because this strategy moves the tool toward the part.
    """

    def refine(self, frame: WorkpieceFrame, machine: Machine,
               targets: List[ProbeTarget]) -> WorkpieceFrame:
        raise NotImplementedError(
            "edge_refine geometry not implemented yet; see PROBING_DESIGN.md")
