"""GRBL banner parsing and version-dependent feature flags."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GrblCapabilities:
    version: str
    major: int
    minor: int
    patch: str = ""

    @property
    def modern(self) -> bool:
        return (self.major, self.minor) >= (1, 1)

    @property
    def jog_commands(self) -> bool:
        return self.modern

    @property
    def realtime_overrides(self) -> bool:
        return self.modern

    @property
    def report_format(self) -> str:
        return "pipe" if self.modern else "comma"


_BANNER = re.compile(r"\bGrbl\s+(\d+)\.(\d+)([a-zA-Z]?)\b")


def parse_boot_banner(line: str) -> Optional[GrblCapabilities]:
    match = _BANNER.search(str(line))
    if not match:
        return None
    major, minor, patch = match.groups()
    version = "%s.%s%s" % (major, minor, patch)
    return GrblCapabilities(version, int(major), int(minor), patch.lower())

