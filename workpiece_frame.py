"""Workpiece coordinate frame: the single object that toolpaths and Z-surface
compensation ride in.

Every parameter -- XY origin, in-plane angle, Z surface -- is sourced
independently: from vision (coarse), manual entry (anytime), or a probe (fine).
Each value carries where it came from and how much to trust it. The frame is
always complete and cuttable from the cheapest sources (eyeballed vision XY +
assumed thickness); probing only raises fidelity on whatever it touches, and is
never on the critical path to a cut.

This module is deliberately pure data: no cv2, no serial, no matplotlib. That
makes it trivially unit-testable and portable -- it can live in the current
monolith today and be lifted unchanged into a future machine/ package later.
"""

import math
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import List, Optional, Tuple


class Source(IntEnum):
    """Where a frame value came from. Ordered worst -> best *typical* fidelity so
    diagnostics/UI can compare; it is NOT an automatic override policy (a human
    may deliberately override a bad probe -- the caller decides)."""
    DEFAULT = 0   # assumed / config default, e.g. nominal material thickness
    MANUAL  = 1   # a human set it: jog-to-touch, or typed a value
    VISION  = 2   # camera homography -- coarse, ~one camera pixel
    PROBE   = 3   # touched the real workpiece -- fine, ~machine resolution
    SCAN    = 4   # dense 3D capture (e.g. structured light) -- future source;
                  # populates ZSurface.samples directly, consumed unchanged


@dataclass
class Measured:
    """A scalar plus its provenance and 1-sigma uncertainty (inches, None=unknown)."""
    value: float
    source: Source = Source.DEFAULT
    tolerance: Optional[float] = None

    def refined_by(self, value: float, source: Source,
                   tolerance: Optional[float] = None) -> "Measured":
        """Return a copy updated by another source. Refinement policy lives with
        the caller; this just records the new value and its provenance."""
        return Measured(value, source, tolerance)


@dataclass
class ZSurface:
    """Top-of-stock height. Either a single plane (one Z, flat/level assumption)
    or a sampled mesh the toolpath is warped to follow."""
    # Fallback plane height, used when there are no samples (manual / assumed).
    nominal: Measured
    # (x, y, z) probe samples in workpiece XY. Fewer than 3 => treat as flat plane.
    samples: List[Tuple[float, float, float]] = field(default_factory=list)

    @property
    def is_mesh(self) -> bool:
        return len(self.samples) >= 3

    def z_at(self, x: float, y: float) -> float:
        """Real surface height at workpiece (x, y). Flat -> nominal; mesh ->
        interpolated. The interpolator is imported lazily to keep this module
        dependency-free at load time."""
        if not self.is_mesh:
            return self.nominal.value
        from probing.mesh import interpolate_z
        return interpolate_z(self.samples, x, y)


@dataclass
class WorkpieceFrame:
    """Origin + orientation + surface that all toolpaths are expressed in.

    x, y      -- workpiece origin in machine coordinates (inches)
    angle     -- in-plane rotation of the workpiece (radians)
    z         -- top-of-stock surface
    """
    x: Measured
    y: Measured
    angle: Measured
    z: ZSurface

    @classmethod
    def eyeballed(cls, x: float, y: float, angle: float = 0.0,
                  thickness: float = 0.0) -> "WorkpieceFrame":
        """The zero-probe baseline: vision/eyeballed XY+angle and an assumed Z.
        This already cuts; probe strategies only refine it."""
        return cls(
            x=Measured(x, Source.VISION),
            y=Measured(y, Source.VISION),
            angle=Measured(angle, Source.VISION),
            z=ZSurface(nominal=Measured(thickness, Source.DEFAULT)),
        )

    # -- transforms: everything rides in this frame -------------------------

    def to_machine_xy(self, wx: float, wy: float) -> Tuple[float, float]:
        """Map a point from workpiece XY to machine XY (rotate by angle, then
        translate to the origin)."""
        a = self.angle.value
        ca, sa = math.cos(a), math.sin(a)
        return (self.x.value + wx * ca - wy * sa,
                self.y.value + wx * sa + wy * ca)

    def surface_z_at(self, wx: float, wy: float) -> float:
        """Top-of-stock height (machine Z) at a workpiece XY point."""
        return self.z.z_at(wx, wy)

    def with_z_nominal(self, m: Measured) -> "WorkpieceFrame":
        return replace(self, z=replace(self.z, nominal=m))
