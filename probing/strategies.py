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


def _wrap_half_turn(angle: float, ref: float) -> float:
    """Map an edge-direction angle (ambiguous mod pi) onto the half-turn
    nearest `ref`."""
    return angle + round((ref - angle) / math.pi) * math.pi


def _edge_direction(pts) -> float:
    """Direction of the line through the two most-separated touch points."""
    (ax, ay), (bx, by) = max(
        ((p, q) for i, p in enumerate(pts) for q in pts[i + 1:]),
        key=lambda pq: (pq[1][0] - pq[0][0]) ** 2 + (pq[1][1] - pq[0][1]) ** 2)
    return math.atan2(by - ay, bx - ax)


@register("edge_refine")
class EdgeRefine(ProbeStrategy):
    """Refine vision's XY origin / angle by touching the real workpiece edges.

    Owns frame.x / frame.y / frame.angle. Vision seeds where each edge is to
    within a camera pixel; each target drops the tool in *beside* the expected
    edge and probes along the target's inward `approach` normal. Supported
    edges are the two through the workpiece origin: approach ~ 0 touches the
    x=0 edge, approach ~ pi/2 touches the y=0 edge (approach angles are in the
    workpiece frame). Other approaches are skipped: the far edges' positions
    depend on stock size, which the frame does not know.

    Geometry:
    - The tool is a cylinder, so at contact its *center* sits cutter_radius
      short of the edge. Contacts are pushed out by that radius only after the
      refined angle is known, so the radius correction does not inherit
      vision's angle error.
    - Two or more touches on one edge give that edge's direction, hence the
      angle. Estimates from both edges are averaged (each wrapped to the
      half-turn nearest the seeded angle first).
    - Each probed edge pins the origin along that edge's normal. An unprobed
      normal keeps its vision value, and the wider tolerance recorded for the
      one-edge case says so.
    """

    def __init__(self, cutter_radius: float = 0.0625, probe_depth: float = 0.2,
                 standoff: float = 0.15, overtravel: float = 0.3,
                 clearance: float = 0.25, feed: float = 5.9,
                 travel_feed: float = 100.0):
        self.cutter_radius = cutter_radius
        self.probe_depth = probe_depth    # how far below the surface to touch
        self.standoff = standoff          # start this far outside the expected edge
        self.overtravel = overtravel      # probe this far past the expected edge
        self.clearance = clearance        # Z clearance between targets
        self.feed = feed
        self.travel_feed = travel_feed

    def refine(self, frame: WorkpieceFrame, machine: Machine,
               targets: List[ProbeTarget]) -> WorkpieceFrame:
        theta = frame.angle.value
        contacts = {"x": [], "y": []}  # raw tool-center contact points

        def near(a: float, b: float) -> bool:
            return abs((a - b + math.pi) % (2 * math.pi) - math.pi) <= math.pi / 4

        for t in targets:
            if t.kind not in ("edge", "corner") or t.approach is None:
                continue
            if near(t.approach, 0.0):
                edge = "x"
            elif near(t.approach, math.pi / 2.0):
                edge = "y"
            else:
                continue

            a = theta + t.approach  # approach normal in machine coords
            nx, ny = math.cos(a), math.sin(a)
            ex, ey = frame.to_machine_xy(t.x, t.y)  # expected edge point
            back = self.standoff + self.cutter_radius
            sx, sy = ex - nx * back, ey - ny * back
            surface = frame.surface_z_at(t.x, t.y)

            machine.move(z=surface + self.clearance, feed=self.travel_feed)
            machine.move(x=sx, y=sy, feed=self.travel_feed)
            machine.move(z=surface - self.probe_depth, feed=self.travel_feed)
            c = machine.probe(x=ex + nx * self.overtravel,
                              y=ey + ny * self.overtravel, feed=self.feed)
            machine.move(x=sx, y=sy, feed=self.travel_feed)
            machine.move(z=surface + self.clearance, feed=self.travel_feed)
            contacts[edge].append((c[0], c[1]))

        if not contacts["x"] and not contacts["y"]:
            return frame

        # -- angle from edge directions (needs 2+ touches on one edge) ------
        estimates = []
        if len(contacts["x"]) >= 2:
            # the x=0 edge runs along the workpiece Y axis (theta + pi/2)
            estimates.append(_wrap_half_turn(
                _edge_direction(contacts["x"]) - math.pi / 2.0, theta))
        if len(contacts["y"]) >= 2:
            # the y=0 edge runs along the workpiece X axis
            estimates.append(_wrap_half_turn(_edge_direction(contacts["y"]), theta))
        new_theta = sum(estimates) / len(estimates) if estimates else theta

        # -- push contacts out to the true edge along the *refined* normals -
        ca, sa = math.cos(new_theta), math.sin(new_theta)
        r = self.cutter_radius
        x_edge_pts = [(px + ca * r, py + sa * r) for px, py in contacts["x"]]
        y_edge_pts = [(px - sa * r, py + ca * r) for px, py in contacts["y"]]

        # -- origin: each probed edge pins the coordinate along its normal --
        old_u = frame.x.value * ca + frame.y.value * sa    # origin . Xw
        old_v = -frame.x.value * sa + frame.y.value * ca   # origin . Yw
        u = (sum(px * ca + py * sa for px, py in x_edge_pts) / len(x_edge_pts)
             if x_edge_pts else old_u)
        v = (sum(-px * sa + py * ca for px, py in y_edge_pts) / len(y_edge_pts)
             if y_edge_pts else old_v)
        ox = u * ca - v * sa
        oy = u * sa + v * ca

        both = bool(x_edge_pts) and bool(y_edge_pts)
        xy_tol = 0.002 if both else 0.01
        new_angle = (Measured(new_theta, Source.PROBE, tolerance=0.002)
                     if estimates else frame.angle)
        return replace(frame,
                       x=Measured(ox, Source.PROBE, tolerance=xy_tol),
                       y=Measured(oy, Source.PROBE, tolerance=xy_tol),
                       angle=new_angle)
