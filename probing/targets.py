"""Vision-driven probe target selection.

'Vision proposes, probe disposes': the overlay already knows -- via the ChArUco
homography and the user's placement -- where the cut will land. This module
turns that placed toolpath into concrete ProbeTargets: a coarse lattice over
the cut region for the z_mesh strategy, spaced off the cut lines themselves,
plus edge targets for edge_refine on rectangular stock.

Pure geometry: no cv2, no hardware, no matplotlib.
"""

import math
from typing import List, Optional, Sequence, Tuple

from probing.base import ProbeTarget

XY = Tuple[float, float]


def _point_segment_dist(px: float, py: float, ax: float, ay: float,
                        bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _dist_to_paths(x: float, y: float, paths: Sequence[Sequence[XY]]) -> float:
    best = float("inf")
    for poly in paths:
        if not poly:
            continue
        if len(poly) == 1:
            best = min(best, math.hypot(x - poly[0][0], y - poly[0][1]))
            continue
        for (ax, ay), (bx, by) in zip(poly, poly[1:]):
            best = min(best, _point_segment_dist(x, y, ax, ay, bx, by))
    return best


def propose_z_targets(paths: Sequence[Sequence[XY]], margin: float = 0.25,
                      grid: int = 3,
                      bounds: Optional[Tuple[float, float, float, float]] = None
                      ) -> List[ProbeTarget]:
    """Propose z_mesh probe points covering the cut region.

    paths  -- polylines [(x, y), ...] in the same coordinates the cut will be
              sent in (work coordinates)
    margin -- minimum distance from any cut segment, and how far the sample
              region extends past the cut's bounding box
    grid   -- candidates come from a grid x grid lattice over the region
    bounds -- (minx, miny, maxx, maxy) stock-region override; default is the
              cut bounding box grown by `margin`

    Candidates closer than `margin` to a cut line are dropped. If that leaves
    fewer than three points (no mesh), the farthest-from-the-cut candidates
    are used instead: probing happens *before* cutting, so landing on a future
    cut line is physically safe -- that surface is just about to be removed,
    making the sample less useful, not dangerous.
    """
    pts = [p for poly in paths for p in poly]
    if not pts:
        return []
    if bounds is None:
        minx = min(p[0] for p in pts) - margin
        maxx = max(p[0] for p in pts) + margin
        miny = min(p[1] for p in pts) - margin
        maxy = max(p[1] for p in pts) + margin
    else:
        minx, miny, maxx, maxy = bounds

    if grid <= 1:
        xs = [(minx + maxx) / 2.0]
        ys = [(miny + maxy) / 2.0]
    else:
        xs = [minx + (maxx - minx) * i / (grid - 1) for i in range(grid)]
        ys = [miny + (maxy - miny) * i / (grid - 1) for i in range(grid)]

    scored = sorted(((_dist_to_paths(cx, cy, paths), (cx, cy))
                     for cx in xs for cy in ys), key=lambda s: -s[0])
    keep = [p for d, p in scored if d >= margin]
    if len(keep) < 3:
        keep = [p for d, p in scored[:3]]
    return [ProbeTarget(x, y, kind="z") for (x, y) in keep]


def propose_edge_targets(width: float, height: float,
                         inset: float = 0.5) -> List[ProbeTarget]:
    """Targets for edge_refine on rectangular stock whose workpiece origin is
    the corner where the x=0 and y=0 edges meet: two touches on the x=0 edge
    (approach along +X) and two on the y=0 edge (approach along +Y), `inset`
    in from the corners. Coordinates are workpiece XY."""
    return [
        ProbeTarget(0.0, inset, kind="edge", approach=0.0),
        ProbeTarget(0.0, height - inset, kind="edge", approach=0.0),
        ProbeTarget(inset, 0.0, kind="edge", approach=math.pi / 2.0),
        ProbeTarget(width - inset, 0.0, kind="edge", approach=math.pi / 2.0),
    ]
