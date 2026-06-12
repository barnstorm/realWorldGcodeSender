"""Probing strategy seam.

A probe strategy is a *refiner*: it takes the current WorkpieceFrame plus
vision-proposed targets, touches the real workpiece, and returns a frame with
higher fidelity on the parameters it owns. Strategies are always optional and
re-runnable; the core stays cuttable with zero probing.

Strategies depend only on the tiny `Machine` protocol below -- NOT on gerbil,
GCodeSender, or the modular machine.controllers stack. The existing GCodeSender
already implements these operations (probe / work_offset_move / get_absolute_pos),
so it can be adapted to this protocol in a few lines without touching this code.
A future grbl.py would satisfy it too -- which is exactly what makes these
strategies portable across both codebases.

'Plugins' here means registration, not dynamic code loading: a strategy is a
class registered into a dict via @register. That is plenty extensible and far
safer/debuggable for a machine that moves a spinning cutter.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, Type

from workpiece_frame import WorkpieceFrame


class Machine(Protocol):
    """The minimal machine surface a probe strategy needs. Coordinates are in
    the active work coordinate system, inches."""

    def probe(self, x: Optional[float] = None, y: Optional[float] = None,
              z: Optional[float] = None, feed: float = 5.9) -> List[float]:
        """Single G38.2 probe toward (x, y, z); returns the contact [x, y, z]."""
        ...

    def move(self, x: Optional[float] = None, y: Optional[float] = None,
             z: Optional[float] = None, feed: float = 100) -> None:
        """Feed move within the current work coordinate system."""
        ...

    def position(self) -> Tuple[float, float, float]:
        """Current machine position (x, y, z)."""
        ...


@dataclass
class ProbeTarget:
    """A vision-proposed place to probe, in workpiece XY (inches).

    kind     -- "z" (touch the top), "edge" (refine one axis), "corner"
    approach -- for edge/corner: inward normal to probe along (radians)
    """
    x: float
    y: float
    kind: str = "z"
    approach: Optional[float] = None


class ProbeStrategy(ABC):
    """frame -> better frame. A strategy owns a subset of frame parameters and
    must leave the rest untouched."""

    #: set by @register
    name: str = "unnamed"

    @abstractmethod
    def refine(self, frame: WorkpieceFrame, machine: Machine,
               targets: List[ProbeTarget]) -> WorkpieceFrame:
        """Touch the real workpiece at/around `targets` and return a frame with
        the owned parameters raised to PROBE fidelity. Must be safe to call on an
        already-refined frame (idempotent-ish: re-probing just re-measures)."""
        raise NotImplementedError


# -- registry ("plugins" = registered strategies) --------------------------

_STRATEGIES: Dict[str, Type[ProbeStrategy]] = {}


def register(name: str):
    """Class decorator: register a ProbeStrategy under `name`."""
    def deco(cls: Type[ProbeStrategy]) -> Type[ProbeStrategy]:
        cls.name = name
        _STRATEGIES[name] = cls
        return cls
    return deco


def get_strategy(name: str) -> ProbeStrategy:
    return _STRATEGIES[name]()


def available() -> List[str]:
    return sorted(_STRATEGIES)
