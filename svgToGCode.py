"""Small, self-contained SVG path loader and GRBL G-code generator.

This module preserves the API historically imported from an untracked sibling
repository.  Coordinates are inches by default and SVG axes are mirrored into
the machine's negative X/Y work area, matching the existing generated jobs.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

from svgpathtools import svg2paths2


@dataclass
class Point3D:
    X: float
    Y: float
    Z: float = 0.0


def signedArea(points):
    if len(points) < 3:
        return 0.0
    return sum(
        points[index].X * points[(index + 1) % len(points)].Y
        - points[(index + 1) % len(points)].X * points[index].Y
        for index in range(len(points))
    ) / 2.0


def _distance(a, b):
    return math.hypot(b.X - a.X, b.Y - a.Y)


def _closed(points, tolerance=1e-6):
    return len(points) >= 3 and _distance(points[0], points[-1]) <= tolerance


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    return value


def _parse_length(value):
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([A-Za-z]*)\s*", value or "")
    if not match:
        return None
    number, unit = match.groups()
    factors = {"in": 1.0, "mm": 1.0 / 25.4, "cm": 1.0 / 2.54,
               "px": 1.0 / 96.0, "pt": 1.0 / 72.0, "": 1.0 / 25.4}
    if unit.lower() not in factors:
        raise ValueError("unsupported SVG length unit: %s" % unit)
    return float(number) * factors[unit.lower()]


def _stroke_color(attributes):
    style = {}
    for entry in attributes.get("style", "").split(";"):
        if ":" in entry:
            key, value = entry.split(":", 1)
            style[key.strip()] = value.strip()
    value = attributes.get("stroke", style.get("stroke", "#000000")).strip().lower()
    if value == "none":
        return None
    if re.fullmatch(r"#[0-9a-f]{3}", value):
        value = "#" + "".join(character * 2 for character in value[1:])
    if re.fullmatch(r"#[0-9a-f]{6}", value):
        return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))
    rgb = re.fullmatch(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", value)
    if rgb:
        return tuple(max(0, min(255, int(component))) for component in rgb.groups())
    return (0, 0, 0)


@dataclass
class CncPath:
    points3D: list
    color: tuple = (0, 0, 0)

    @property
    def is_closed(self):
        return _closed(self.points3D)

    @property
    def cut_fraction(self):
        # Historical files use black for full-depth cuts and green intensity
        # for partial-depth paths.
        green = self.color[1]
        return 1.0 if green == 0 else max(0.0, min(1.0, green / 255.0))


class cncPathsClass:
    def __init__(self, inputSvgFile=None, points3D=None, pointsPerCurve=30,
                 distPerTab=8.0, tabWidth=0.25, cutterDiameter=0.125,
                 convertSvfToIn=False):
        if inputSvgFile is None and points3D is None:
            raise ValueError("inputSvgFile or points3D is required")
        self.pointsPerCurve = max(2, int(pointsPerCurve))
        self.distPerTab = max(0.0, _finite(distPerTab, "distPerTab"))
        self.tabWidth = max(0.0, _finite(tabWidth, "tabWidth"))
        self.cutterDiameter = max(0.0, _finite(cutterDiameter, "cutterDiameter"))
        self.cncPaths = []
        if inputSvgFile is not None:
            self._load_svg(inputSvgFile, bool(convertSvfToIn))
        else:
            copied = [Point3D(_finite(point.X, "X"), _finite(point.Y, "Y"),
                              _finite(point.Z or 0.0, "Z")) for point in points3D]
            if len(copied) < 2:
                raise ValueError("a CNC path requires at least two points")
            self.cncPaths.append(CncPath(copied))

    def _load_svg(self, filename, convert_to_inches):
        paths, attributes, svg_attributes = svg2paths2(str(filename))
        view_box = [float(value) for value in
                    svg_attributes.get("viewBox", "0 0 1 1").replace(",", " ").split()]
        if len(view_box) != 4 or view_box[2] == 0 or view_box[3] == 0:
            raise ValueError("SVG viewBox must contain four non-zero values")
        if convert_to_inches:
            width = _parse_length(svg_attributes.get("width", ""))
            height = _parse_length(svg_attributes.get("height", ""))
            scale_x = width / view_box[2] if width is not None else 1.0 / 25.4
            scale_y = height / view_box[3] if height is not None else scale_x
        else:
            scale_x = scale_y = 1.0

        for path, attrs in zip(paths, attributes):
            color = _stroke_color(attrs)
            if color is None:
                continue
            points = []
            for segment in path:
                sample_count = 2 if segment.__class__.__name__ == "Line" else self.pointsPerCurve
                for index in range(sample_count):
                    if points and index == 0:
                        continue
                    value = segment.point(index / float(sample_count - 1))
                    x = -(value.real - view_box[0]) * scale_x
                    y = -(value.imag - view_box[1]) * scale_y
                    point = Point3D(_finite(x, "SVG X"), _finite(y, "SVG Y"), 0.0)
                    if not points or _distance(points[-1], point) > 1e-9:
                        points.append(point)
            if len(points) >= 2:
                self.cncPaths.append(CncPath(points, color))
        if not self.cncPaths:
            raise ValueError("SVG contains no stroked paths")

    def orderPartialCutsFirst(self):
        self.cncPaths.sort(key=lambda path: path.cut_fraction)

    def orderCncHolePathsFirst(self):
        def point_in_polygon(point, polygon):
            inside = False
            x, y = point.X, point.Y
            for first, second in zip(polygon, polygon[1:] + polygon[:1]):
                if ((first.Y > y) != (second.Y > y)):
                    cross_x = (second.X - first.X) * (y - first.Y) / (second.Y - first.Y) + first.X
                    if x < cross_x:
                        inside = not inside
            return inside

        def nesting_depth(path):
            if not path.is_closed:
                return 0
            probe = path.points3D[0]
            return sum(1 for other in self.cncPaths
                       if other is not path and other.is_closed
                       and point_in_polygon(probe, other.points3D[:-1]))

        self.cncPaths.sort(
            key=lambda path: (-nesting_depth(path), abs(signedArea(path.points3D)))
        )


class cncGcodeGeneratorClass:
    def __init__(self, cncPaths, materialThickness, depthBelowMaterial,
                 depthPerPass, cutFeedRate, safeHeight, tabHeight,
                 useMM=False):
        self.cncPaths = cncPaths
        self.materialThickness = _finite(materialThickness, "materialThickness")
        self.depthBelowMaterial = _finite(depthBelowMaterial, "depthBelowMaterial")
        self.depthPerPass = _finite(depthPerPass, "depthPerPass")
        self.cutFeedRate = _finite(cutFeedRate, "cutFeedRate")
        self.safeHeight = _finite(safeHeight, "safeHeight")
        self.tabHeight = _finite(tabHeight, "tabHeight")
        self.useMM = bool(useMM)
        self.gCodes = []
        self._validate()

    def _validate(self):
        if self.materialThickness <= 0:
            raise ValueError("materialThickness must be positive")
        if self.depthBelowMaterial < 0:
            raise ValueError("depthBelowMaterial cannot be negative")
        if self.depthPerPass <= 0:
            raise ValueError("depthPerPass must be positive")
        if self.cutFeedRate <= 0:
            raise ValueError("cutFeedRate must be positive")
        if self.safeHeight < 0:
            raise ValueError("safeHeight cannot be negative")
        if not 0 <= self.tabHeight <= self.materialThickness:
            raise ValueError("tabHeight must be between zero and materialThickness")
        if not getattr(self.cncPaths, "cncPaths", None):
            raise ValueError("at least one CNC path is required")

    @staticmethod
    def _fmt(value):
        value = 0.0 if abs(value) < 0.0000005 else value
        return ("%.5f" % value).rstrip("0").rstrip(".")

    def _depths(self, path):
        total = (self.materialThickness + self.depthBelowMaterial) * path.cut_fraction
        count = max(1, int(math.ceil(total / self.depthPerPass)))
        return [-min(total, self.depthPerPass * index) for index in range(1, count + 1)]

    @staticmethod
    def _split_intervals(points, boundaries):
        cumulative = [0.0]
        for first, second in zip(points, points[1:]):
            cumulative.append(cumulative[-1] + _distance(first, second))
        cuts = sorted({*cumulative, *[value for value in boundaries
                                     if 0.0 < value < cumulative[-1]]})
        result = []
        segment = 0
        for start, end in zip(cuts, cuts[1:]):
            while segment + 1 < len(cumulative) and cumulative[segment + 1] < end - 1e-10:
                segment += 1
            segment_length = cumulative[segment + 1] - cumulative[segment]
            ratio = 0.0 if segment_length == 0 else (end - cumulative[segment]) / segment_length
            first, second = points[segment], points[segment + 1]
            endpoint = Point3D(first.X + (second.X - first.X) * ratio,
                               first.Y + (second.Y - first.Y) * ratio, 0.0)
            result.append((start, end, endpoint))
        return result

    def _final_pass_intervals(self, path):
        points = path.points3D
        total = sum(_distance(first, second) for first, second in zip(points, points[1:]))
        spacing = self.cncPaths.distPerTab
        width = min(self.cncPaths.tabWidth, spacing) if spacing > 0 else 0.0
        if not path.is_closed or spacing <= 0 or width <= 0 or total <= spacing:
            return self._split_intervals(points, [])
        centers = [spacing * index for index in range(1, int(total // spacing) + 1)]
        boundaries = []
        for center in centers:
            boundaries.extend((center - width / 2.0, center + width / 2.0))
        return self._split_intervals(points, boundaries)

    def Generate(self):
        unit_scale = 25.4 if self.useMM else 1.0
        unit_code = "G21" if self.useMM else "G20"
        safe = self.safeHeight * unit_scale
        feed = self.cutFeedRate * unit_scale
        codes = [unit_code, "G90", "G17", "G94", "F" + self._fmt(feed),
                 "G0 Z" + self._fmt(safe)]
        for path in self.cncPaths.cncPaths:
            points = path.points3D
            depths = self._depths(path)
            for pass_index, depth in enumerate(depths):
                start = points[0]
                codes.append("G0 Z" + self._fmt(safe))
                codes.append("G0 X%s Y%s" % (self._fmt(start.X * unit_scale),
                                              self._fmt(start.Y * unit_scale)))
                current_depth = depth * unit_scale
                codes.append("G1 Z%s F%s" % (self._fmt(current_depth), self._fmt(feed)))
                final_pass = pass_index == len(depths) - 1
                intervals = (self._final_pass_intervals(path) if final_pass
                             else self._split_intervals(points, []))
                tab_depth = -max(0.0, self.materialThickness - self.tabHeight) * unit_scale
                for start_distance, end_distance, endpoint in intervals:
                    midpoint = (start_distance + end_distance) / 2.0
                    in_tab = final_pass and path.is_closed and self.cncPaths.distPerTab > 0 and any(
                        abs(midpoint - self.cncPaths.distPerTab * index) <= self.cncPaths.tabWidth / 2.0
                        for index in range(1, int(end_distance // self.cncPaths.distPerTab) + 2)
                    )
                    target_depth = max(current_depth, tab_depth) if in_tab else current_depth
                    if abs(target_depth - current_depth) > 1e-9:
                        codes.append("G1 Z%s" % self._fmt(target_depth))
                    codes.append("G1 X%s Y%s" % (self._fmt(endpoint.X * unit_scale),
                                                  self._fmt(endpoint.Y * unit_scale)))
                    if abs(target_depth - current_depth) > 1e-9:
                        codes.append("G1 Z%s" % self._fmt(current_depth))
        codes.extend(("G0 Z" + self._fmt(safe), "M2"))
        self.gCodes = codes
        return codes

    def Save(self, filename):
        if not self.gCodes:
            self.Generate()
        Path(filename).write_text("\n".join(self.gCodes) + "\n", encoding="ascii")
