"""Unit tests for the probing seam: toolpath warp, target selection, and the
probe strategies driven by fake machines (no hardware, no cv2, no serial).

Run with:  python tests/test_probing.py
"""

import math
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workpiece_frame import Measured, Source, WorkpieceFrame, ZSurface
from probing.base import ProbeTarget
from probing.strategies import EdgeRefine, ZMesh
from probing.targets import propose_edge_targets, propose_z_targets
from toolpath_warp import warp_gcode_lines, warp_points


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeEdgeMachine:
    """Simulates rectangular stock for edge probing: knows the true origin,
    angle, and cutter radius; probe() stops where the tool circle first meets
    one of the two origin edge planes."""

    def __init__(self, origin, angle, cutter_radius):
        self.ox, self.oy = origin
        self.xw = (math.cos(angle), math.sin(angle))    # x=0 edge normal
        self.yw = (-math.sin(angle), math.cos(angle))   # y=0 edge normal
        self.r = cutter_radius
        self.pos = [0.0, 0.0, 1.0]

    def move(self, x=None, y=None, z=None, feed=100):
        for i, v in enumerate((x, y, z)):
            if v is not None:
                self.pos[i] = v

    def position(self):
        return tuple(self.pos)

    def probe(self, x=None, y=None, z=None, feed=5.9):
        tx = self.pos[0] if x is None else x
        ty = self.pos[1] if y is None else y
        sx, sy = self.pos[0], self.pos[1]
        length = math.hypot(tx - sx, ty - sy)
        assert length > 0, "probe with no XY motion"
        ux, uy = (tx - sx) / length, (ty - sy) / length
        best_t = None
        for nx, ny in (self.xw, self.yw):
            denom = ux * nx + uy * ny
            if denom <= 1e-12:
                continue  # moving away from / parallel to this edge
            s0 = (sx - self.ox) * nx + (sy - self.oy) * ny
            t = (-self.r - s0) / denom  # tool center stops r outside the plane
            if t >= 0 and (best_t is None or t < best_t):
                best_t = t
        assert best_t is not None and best_t <= length, "probe missed the stock"
        self.pos[0] = sx + ux * best_t
        self.pos[1] = sy + uy * best_t
        return [self.pos[0], self.pos[1], self.pos[2]]


class FakeSurfaceMachine:
    """Z-only probing against a known surface z = f(x, y)."""

    def __init__(self, surface):
        self.surface = surface
        self.pos = [0.0, 0.0, 1.0]

    def move(self, x=None, y=None, z=None, feed=100):
        for i, v in enumerate((x, y, z)):
            if v is not None:
                self.pos[i] = v

    def position(self):
        return tuple(self.pos)

    def probe(self, x=None, y=None, z=None, feed=5.9):
        assert z is not None, "expected a Z probe"
        self.pos[2] = self.surface(self.pos[0], self.pos[1])
        return [self.pos[0], self.pos[1], self.pos[2]]


# ---------------------------------------------------------------------------
# toolpath_warp
# ---------------------------------------------------------------------------

class TestWarpPoints(unittest.TestCase):
    def test_flat_surface_is_noop(self):
        pts = [(0.0, 0.0, -0.1), (1.0, 0.0, -0.1)]
        out = warp_points(pts, lambda x, y: 0.0, nominal=0.0, max_segment=10.0)
        self.assertEqual(out, pts)

    def test_long_move_is_split_and_follows_surface(self):
        surface = lambda x, y: 0.05 * x
        out = warp_points([(0.0, 0.0, -0.1), (2.0, 0.0, -0.1)], surface,
                          max_segment=0.5)
        self.assertEqual(len(out), 5)  # start + 4 sub-segments
        for x, y, z in out:
            self.assertAlmostEqual(z, -0.1 + 0.05 * x, places=9)
        self.assertAlmostEqual(out[-1][0], 2.0)

    def test_nominal_offsets_shift(self):
        out = warp_points([(0.0, 0.0, -0.1)], lambda x, y: 0.02, nominal=0.05)
        self.assertAlmostEqual(out[0][2], -0.1 + 0.02 - 0.05)


class TestWarpGcodeLines(unittest.TestCase):
    def _xyz(self, line):
        words = dict(re.findall(r"([XYZ])([-+]?[0-9.]+)", line))
        return {k: float(v) for k, v in words.items()}

    def test_linear_moves_warped_and_split(self):
        surface = lambda x, y: 0.05 * x
        lines = ["G20",
                 "G1 X0.0000 Y0.0000 Z-0.1000 F30.0",
                 "G1 X2.0000 Y0.0000"]
        out = warp_gcode_lines(lines, surface, nominal=0.0, max_segment=0.5)
        self.assertEqual(out[0], "G20")
        self.assertEqual(len(out), 6)  # G20 + first move + 4 sub-segments
        for line in out[1:]:
            c = self._xyz(line)
            self.assertAlmostEqual(c["Z"], -0.1 + 0.05 * c["X"], places=4)
        # feed carried on the first line of the original move only
        self.assertIn("F30", out[1])
        self.assertNotIn("F", out[2])
        self.assertAlmostEqual(self._xyz(out[-1])["X"], 2.0)

    def test_passthrough_of_non_linear_lines(self):
        lines = ["G21", "M3 S1000", "G2 X1.0 Y1.0 I0.5 J0.0"]
        out = warp_gcode_lines(lines, lambda x, y: 0.5)
        self.assertEqual(out, lines)

    def test_incremental_mode_disables_warping(self):
        lines = ["G91", "G1 X1.0 Y0.0 Z-0.1"]
        out = warp_gcode_lines(lines, lambda x, y: 0.5)
        self.assertEqual(out, lines)

    def test_rapid_shifted_but_not_split(self):
        lines = ["G0 X2.0000 Y0.0000 Z0.2500"]
        out = warp_gcode_lines(lines, lambda x, y: 0.05 * x)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(self._xyz(out[0])["Z"], 0.25 + 0.1, places=4)


# ---------------------------------------------------------------------------
# probing.targets
# ---------------------------------------------------------------------------

class TestZTargets(unittest.TestCase):
    def test_targets_clear_of_path(self):
        square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]
        targets = propose_z_targets([square], margin=0.25, grid=3)
        self.assertGreaterEqual(len(targets), 3)
        for t in targets:
            self.assertEqual(t.kind, "z")
            d = min(self._seg_dist(t.x, t.y, a, b)
                    for a, b in zip(square, square[1:]))
            self.assertGreaterEqual(d, 0.25 - 1e-9)

    def test_fallback_when_nothing_is_clear(self):
        # dense hatch: no candidate is `margin` clear, farthest ones returned
        hatch = [[(0.0, y), (2.0, y)] for y in [0.0, 0.5, 1.0, 1.5, 2.0]]
        targets = propose_z_targets(hatch, margin=0.5, grid=3,
                                    bounds=(0.0, 0.0, 2.0, 2.0))
        self.assertEqual(len(targets), 3)

    def test_empty_paths(self):
        self.assertEqual(propose_z_targets([]), [])

    @staticmethod
    def _seg_dist(px, py, a, b):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) /
                         (dx * dx + dy * dy)))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class TestEdgeTargets(unittest.TestCase):
    def test_rect_targets(self):
        targets = propose_edge_targets(6.0, 4.0, inset=0.5)
        self.assertEqual(len(targets), 4)
        self.assertTrue(all(t.kind == "edge" for t in targets))
        approaches = sorted(t.approach for t in targets)
        self.assertAlmostEqual(approaches[0], 0.0)
        self.assertAlmostEqual(approaches[-1], math.pi / 2)


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------

class TestZMeshStrategy(unittest.TestCase):
    def test_mesh_records_surface(self):
        surface = lambda x, y: 0.02 * x + 0.01 * y
        machine = FakeSurfaceMachine(surface)
        frame = WorkpieceFrame.eyeballed(0.0, 0.0)
        targets = [ProbeTarget(x, y) for x, y in
                   [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)]]
        refined = ZMesh().refine(frame, machine, targets)
        self.assertTrue(refined.z.is_mesh)
        self.assertEqual(len(refined.z.samples), 4)
        for t in targets:
            self.assertAlmostEqual(refined.surface_z_at(t.x, t.y),
                                   surface(t.x, t.y), places=9)

    def test_mesh_feeds_gcode_warp(self):
        surface = lambda x, y: 0.05 * x
        machine = FakeSurfaceMachine(surface)
        frame = WorkpieceFrame.eyeballed(0.0, 0.0)
        targets = [ProbeTarget(x, y) for x, y in
                   [(0.0, 0.0), (2.0, 0.0), (1.0, 1.0)]]
        refined = ZMesh().refine(frame, machine, targets)
        out = warp_gcode_lines(["G1 X0.0 Y0.0 Z-0.1 F30", "G1 X2.0 Y0.0"],
                               refined.z.z_at,
                               nominal=refined.z.nominal.value)
        z_end = float(re.search(r"Z([-+0-9.]+)", out[-1]).group(1))
        self.assertAlmostEqual(z_end, -0.1 + surface(2.0, 0.0), places=3)


class TestEdgeRefineStrategy(unittest.TestCase):
    TRUE_ORIGIN = (1.0, 2.0)
    TRUE_ANGLE = math.radians(2.0)
    R = 0.0625

    def test_recovers_origin_and_angle(self):
        machine = FakeEdgeMachine(self.TRUE_ORIGIN, self.TRUE_ANGLE, self.R)
        # vision seed: a little off in origin and angle
        frame = WorkpieceFrame.eyeballed(1.03, 1.97, math.radians(1.5))
        targets = propose_edge_targets(6.0, 4.0, inset=0.5)
        refined = EdgeRefine(cutter_radius=self.R).refine(frame, machine, targets)

        self.assertAlmostEqual(refined.angle.value, self.TRUE_ANGLE, places=6)
        self.assertAlmostEqual(refined.x.value, self.TRUE_ORIGIN[0], places=6)
        self.assertAlmostEqual(refined.y.value, self.TRUE_ORIGIN[1], places=6)
        self.assertEqual(refined.x.source, Source.PROBE)
        self.assertEqual(refined.angle.source, Source.PROBE)
        # z untouched: strategy owns x/y/angle only
        self.assertEqual(refined.z, frame.z)

    def test_single_edge_pins_one_normal_only(self):
        machine = FakeEdgeMachine(self.TRUE_ORIGIN, self.TRUE_ANGLE, self.R)
        frame = WorkpieceFrame.eyeballed(1.03, 1.97, self.TRUE_ANGLE)
        targets = [t for t in propose_edge_targets(6.0, 4.0, inset=0.5)
                   if abs(t.approach) < 1e-9]  # x=0 edge only
        refined = EdgeRefine(cutter_radius=self.R).refine(frame, machine, targets)
        ca, sa = math.cos(self.TRUE_ANGLE), math.sin(self.TRUE_ANGLE)
        # pinned along the x-edge normal...
        self.assertAlmostEqual(refined.x.value * ca + refined.y.value * sa,
                               self.TRUE_ORIGIN[0] * ca + self.TRUE_ORIGIN[1] * sa,
                               places=6)
        # ...while the unprobed normal keeps the vision value
        self.assertAlmostEqual(-refined.x.value * sa + refined.y.value * ca,
                               -1.03 * sa + 1.97 * ca, places=6)
        self.assertGreater(refined.x.tolerance, 0.002)

    def test_idempotent_reprobe(self):
        machine = FakeEdgeMachine(self.TRUE_ORIGIN, self.TRUE_ANGLE, self.R)
        frame = WorkpieceFrame.eyeballed(1.03, 1.97, math.radians(1.5))
        targets = propose_edge_targets(6.0, 4.0, inset=0.5)
        strategy = EdgeRefine(cutter_radius=self.R)
        once = strategy.refine(frame, machine, targets)
        twice = strategy.refine(once, machine, targets)
        self.assertAlmostEqual(twice.x.value, once.x.value, places=6)
        self.assertAlmostEqual(twice.angle.value, once.angle.value, places=6)

    def test_no_edge_targets_returns_frame_unchanged(self):
        machine = FakeEdgeMachine(self.TRUE_ORIGIN, self.TRUE_ANGLE, self.R)
        frame = WorkpieceFrame.eyeballed(1.03, 1.97)
        refined = EdgeRefine().refine(frame, machine, [ProbeTarget(1.0, 1.0)])
        self.assertEqual(refined, frame)


if __name__ == "__main__":
    unittest.main(verbosity=2)
