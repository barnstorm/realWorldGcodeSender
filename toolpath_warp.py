"""Warp toolpath Z to follow the real (probed) surface.

The z_mesh strategy records where the top of the stock actually is
(ZSurface.samples); this module bends programmed moves so the *depth of cut*
stays constant across a warped / tilted workpiece.

Convention: work zero was set at the nominal top of stock, so a move's
programmed Z is relative to where the surface was assumed to be. If the real
surface at (x, y) sits at z_at(x, y) instead of `nominal`, the move's Z is
shifted by (z_at(x, y) - nominal) -- and any long feed move is first split into
sub-segments so Z follows the surface *between* mesh samples instead of
chording straight across them.

Pure geometry: no gerbil, no matplotlib, no numpy. This is the path layer's
piece of the probing design (PROBING_DESIGN.md) -- it consumes a z_at callable
and knows nothing about how the samples were captured, so it is portable
unchanged to a future modular tree.
"""

import math
import re
from typing import Callable, List, Sequence, Tuple

Point = Tuple[float, float, float]

_WORD = re.compile(r"([A-Za-z])\s*([-+]?[0-9]*\.?[0-9]+)")

# G words that reinterpret or move coordinates in ways this tracker does not
# model; seeing one resets the modal position to unknown.
_POSITION_UNSAFE_G = (10.0, 28.0, 28.1, 30.0, 92.0, 92.1)


def warp_points(points: Sequence[Point], z_at: Callable[[float, float], float],
                nominal: float = 0.0, max_segment: float = 0.5) -> List[Point]:
    """Return `points` with Z shifted onto the real surface, splitting any XY
    span longer than `max_segment` so the path follows the surface between
    samples instead of chording across them."""
    if not points:
        return []
    out = []
    px, py, pz = points[0]
    out.append((px, py, pz + z_at(px, py) - nominal))
    for x, y, z in points[1:]:
        dist = math.hypot(x - px, y - py)
        steps = max(1, math.ceil(dist / max_segment)) if max_segment > 0 else 1
        for i in range(1, steps + 1):
            t = i / steps
            ix = px + (x - px) * t
            iy = py + (y - py) * t
            iz = pz + (z - pz) * t
            out.append((ix, iy, iz + z_at(ix, iy) - nominal))
        px, py, pz = x, y, z
    return out


def warp_gcode_lines(lines: Sequence[str], z_at: Callable[[float, float], float],
                     nominal: float = 0.0, max_segment: float = 0.5,
                     decimals: int = 4) -> List[str]:
    """Apply warp_points() to a stream of already-generated G-code lines.

    Understands the linear, absolute subset the generators emit (G0/G1 with
    modal X/Y/Z/F words, inches). Everything else passes through untouched:
    arcs (G2/G3) are emitted as-is (their endpoint still updates the modal
    position), G92/G28-style lines and incremental mode (G91) mark the position
    unknown / stop warping, and a move can only be warped once enough modal
    state is known to evaluate the surface at its target.

    Feed moves (G1) are split per `max_segment`; rapids (G0) get their endpoint
    shifted but are not split -- they are travel, not cutting.
    """
    out = []
    cur = [None, None, None]  # modal position in *unwarped* programmed coords
    motion = None             # modal motion group (0/1/2/3)
    incremental = False       # G91 seen: warping unsafe until G90
    fmt = "%%.%df" % decimals

    for line in lines:
        body = line.split("(")[0].split(";")[0]
        pairs = [(L.upper(), float(v)) for L, v in _WORD.findall(body)]
        gvals = [v for L, v in pairs if L == "G"]
        coords = {L: v for L, v in pairs if L in "XYZ"}

        for g in gvals:
            if g in (0.0, 1.0, 2.0, 3.0):
                motion = int(g)

        if any(g == 91.0 for g in gvals):
            incremental = True
        if incremental:
            if any(g == 90.0 for g in gvals):
                incremental = False
                cur = [None, None, None]
            out.append(line)
            continue
        if any(g in _POSITION_UNSAFE_G for g in gvals):
            cur = [None, None, None]
            out.append(line)
            continue

        if not coords:
            out.append(line)
            continue

        tx = coords.get("X", cur[0])
        ty = coords.get("Y", cur[1])
        tz = coords.get("Z", cur[2])

        can_warp = (motion in (0, 1)
                    and all(g in (0.0, 1.0) for g in gvals)
                    and tx is not None and ty is not None and tz is not None)
        if not can_warp:
            out.append(line)
            cur = [tx, ty, tz]
            continue

        feed = next((v for L, v in pairs if L == "F"), None)
        extras = [(L, v) for L, v in pairs if L not in "GXYZF"]

        if motion == 1 and None not in cur:
            pts = warp_points([tuple(cur), (tx, ty, tz)], z_at,
                              nominal=nominal, max_segment=max_segment)[1:]
        else:
            # rapid, or start unknown: shift the endpoint only, no split
            pts = [(tx, ty, tz + z_at(tx, ty) - nominal)]

        for i, (wx, wy, wz) in enumerate(pts):
            words = ["G%d" % motion,
                     "X" + fmt % wx, "Y" + fmt % wy, "Z" + fmt % wz]
            if i == 0:
                words += ["%s%g" % (L, v) for L, v in extras]
                if feed is not None:
                    words.append("F%g" % feed)
            out.append(" ".join(words))
        cur = [tx, ty, tz]
    return out
