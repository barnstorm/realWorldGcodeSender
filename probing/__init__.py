"""Probing: vision-seeded refiners that raise the fidelity of a WorkpieceFrame.

See PROBING_DESIGN.md for the design. Importing this package registers the
built-in strategies so they show up in `available()`.
"""

from probing.base import (
    Machine,
    ProbeStrategy,
    ProbeTarget,
    available,
    get_strategy,
    register,
)

# Import for side effect: registers the built-in strategies.
from probing import strategies  # noqa: F401

__all__ = [
    "Machine",
    "ProbeStrategy",
    "ProbeTarget",
    "available",
    "get_strategy",
    "register",
]
