"""Pure toolpath -> overlay-polyline builder (no Qt, no cv2).

Port of the transform chain in OverlayGcode.overlaySvgOrGcode: copy the
loaded points, apply per-path offsets, rotate about the placement origin,
then map to bed-view pixels. The SAME math the send path applies -- what you
see is what gets cut. Operates on plain (x, y) tuples so it is trivially
unit-testable.

Each returned polyline is {"points": [(px, py), ...], "powers": [0..1, ...]}
where a power of 0 marks a rapid/jump segment (drawn as a move, not a line),
mirroring the monolith's laserPowers convention.
"""

import math
from typing import Dict, List, Sequence, Tuple

XY = Tuple[float, float]
Polyline = Dict[str, list]


def rotate_point(origin: XY, point: XY, angle: float) -> XY:
    """Counterclockwise rotation about origin, radians (monolith rotate())."""
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def svg_overlay(paths: Sequence[Sequence[XY]], colors: Sequence,
                offsets: Sequence[XY], rotation_deg: float) -> List[Polyline]:
    """SVG mode: per-path offset, then rotate everything about the LAST
    path's offset (the monolith's rotation origin), powers from path color
    (color[1] == 0 => first segment is a jump, rest full power)."""
    rotation = rotation_deg * math.pi / 180.0
    origin = tuple(offsets[-1]) if offsets else (0.0, 0.0)
    out = []
    for pts, color, offset in zip(paths, colors, offsets):
        moved = [(x + offset[0], y + offset[1]) for x, y in pts]
        moved = [rotate_point(origin, p, rotation) for p in moved]
        if color[1] == 0:
            powers = [0.0] + [1.0] * (len(moved) - 1)
        else:
            powers = [color[1] / 255.0] * len(moved)
        out.append({"points": moved, "powers": powers})
    return out


def gcode_overlay(points: Sequence[XY], powers: Sequence[float],
                  x_off: float, y_off: float, rotation_deg: float) -> List[Polyline]:
    """G-code-file mode: single polyline, offset then rotated about the
    offset point (monolith gCodeFile branch)."""
    rotation = rotation_deg * math.pi / 180.0
    moved = [(x + x_off, y + y_off) for x, y in points]
    moved = [rotate_point((x_off, y_off), p, rotation) for p in moved]
    return [{"points": moved, "powers": list(powers)}]


def drawn_overlay(points: Sequence[XY]) -> List[Polyline]:
    pts = list(points)
    return [{"points": pts, "powers": [1.0] * len(pts)}] if pts else []


def to_scene(polylines: Sequence[Polyline], transform) -> List[Polyline]:
    """Map polylines from inches to bed-view pixels via
    BedTransform.phy_to_pixels (overlay parity: no plate correction)."""
    return [{"points": [transform.phy_to_pixels(x, y) for x, y in p["points"]],
             "powers": p["powers"]} for p in polylines]
