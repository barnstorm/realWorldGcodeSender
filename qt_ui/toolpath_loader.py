"""Load toolpaths for the Qt UI (no Qt imports; no matplotlib).

- SVG: via the in-repository svgToGCode module, with the exact parameters the
  monolith uses.
- G-code file: a faithful port of the pygcode parsing block in
  OverlayGcode.__init__ (units handling, arc expansion, laser powers),
  reusing the monolith's arcToPoints for parity.
"""

import re
from typing import List, Sequence, Tuple

XY = Tuple[float, float]

_NUM = "[+-]?([0-9]*[.])?[0-9]+"


def load_svg(svg_file: str, cutter_diameter: float):
    """Return the cncPathsClass object kept whole for preview and sending."""
    from svgToGCode import cncPathsClass
    cnc = cncPathsClass(inputSvgFile=svg_file,
                        pointsPerCurve=30,
                        distPerTab=7.87,
                        tabWidth=0.25,
                        cutterDiameter=cutter_diameter,
                        convertSvfToIn=True)
    cnc.orderCncHolePathsFirst()
    cnc.orderPartialCutsFirst()
    return cnc


def svg_paths_as_tuples(cnc) -> Tuple[List[List[XY]], List]:
    """Plain (x, y) polylines + colors from a cncPathsClass, for the pure
    overlay model (the cnc object itself stays pristine for sending)."""
    paths = [[(p.X, p.Y) for p in path.points3D] for path in cnc.cncPaths]
    colors = [path.color for path in cnc.cncPaths]
    return paths, colors


def _word(line: str, letter: str):
    m = re.search(letter + _NUM, line)
    return float(m.group()[1:]) if m else None


def load_gcode_file(gcode_file: str) -> Tuple[List[XY], List[float]]:
    """Port of the monolith's G-code preview parser: simulate the file with
    pygcode's Machine, convert to inches, expand arcs, track laser power.
    Returns (points, powers) with powers[i] in 0..1 (0 = rapid/jump)."""
    import pygcode
    from pygcode import Machine
    from pygcode.gcodes import MODAL_GROUP_MAP
    from realWorldGcodeSender import arcToPoints  # side-effect-free import

    machine = Machine()
    points: List[XY] = []
    powers: List[float] = []

    with open(gcode_file, "r") as fh:
        for line_text in fh.readlines():
            line = pygcode.Line(line_text)
            prev = machine.pos
            machine.process_block(line.block)

            motion = str(machine.mode.modal_groups[MODAL_GROUP_MAP["motion"]])
            s_code = str(machine.mode.modal_groups[MODAL_GROUP_MAP["spindle_speed"]])
            power = float(s_code.split("S")[1]) / 100.0
            powers.append(0.0 if motion in ("G00", "G0") else power)

            unit = str(machine.mode.modal_groups[MODAL_GROUP_MAP["units"]])
            scale = 1.0 if unit == "G20" else 1.0 / 25.4
            arc = motion in ("G02", "G2", "G03", "G3")
            before_comment = line_text.split("(")[0]

            if arc:
                x = _word(before_comment, "X")
                y = _word(before_comment, "Y")
                i = _word(before_comment, "I")
                j = _word(before_comment, "J")
                if x is not None and y is not None and i is not None and j is not None:
                    arc_pts = arcToPoints(prev.X * scale, prev.Y * scale,
                                          machine.pos.X * scale, machine.pos.Y * scale,
                                          i * scale, j * scale,
                                          "G02" in motion or "G2" == motion,
                                          machine.pos.Z * scale)
                    points.extend((p.X, p.Y) for p in arc_pts)
                    powers.extend([powers[-1]] * (len(points) - len(powers)))
                    continue
            points.append((machine.pos.X * scale, machine.pos.Y * scale))
    return points, powers
