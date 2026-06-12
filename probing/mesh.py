"""Z-surface interpolation over probe samples.

This is the one piece with real algorithmic meat. The intended end state is a
Delaunay triangulation of the (x, y) samples with barycentric interpolation of Z
inside each triangle, so the toolpath rides a piecewise-planar real surface.

For now we ship inverse-distance weighting: dependency-free, good enough to make
ZSurface.z_at() functional and testable. Swap the body for triangulation later
without changing the signature.
"""

from typing import List, Tuple


def interpolate_z(samples: List[Tuple[float, float, float]],
                  x: float, y: float, power: float = 2.0) -> float:
    """Estimate surface Z at (x, y) from (x, y, z) samples via inverse-distance
    weighting. Returns the exact sample Z when (x, y) lands on one."""
    if not samples:
        raise ValueError("interpolate_z requires at least one sample")

    num = 0.0
    den = 0.0
    for sx, sy, sz in samples:
        d2 = (x - sx) ** 2 + (y - sy) ** 2
        if d2 == 0.0:
            return sz
        w = 1.0 / (d2 ** (power / 2.0))
        num += w * sz
        den += w
    return num / den
