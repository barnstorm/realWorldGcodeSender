import math

import pytest

from svgToGCode import Point3D, cncGcodeGeneratorClass, cncPathsClass


def _generator(paths, **overrides):
    values = dict(materialThickness=0.25, depthBelowMaterial=0.05,
                  depthPerPass=0.1, cutFeedRate=60, safeHeight=0.2,
                  tabHeight=0.05, useMM=False)
    values.update(overrides)
    return cncGcodeGeneratorClass(paths, **values)


def test_svg_loader_maps_viewbox_to_negative_machine_inches(tmp_path):
    svg = tmp_path / "square.svg"
    svg.write_text(
        '<svg width="2in" height="1in" viewBox="0 0 50.8 25.4" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<path d="M 0,0 L 50.8,0 L 50.8,25.4 L 0,25.4 Z" '
        'style="fill:none;stroke:#000000"/></svg>',
        encoding="ascii",
    )
    paths = cncPathsClass(inputSvgFile=svg, convertSvfToIn=True)
    points = paths.cncPaths[0].points3D
    assert min(point.X for point in points) == pytest.approx(-2.0)
    assert max(point.X for point in points) == pytest.approx(0.0)
    assert min(point.Y for point in points) == pytest.approx(-1.0)
    assert paths.cncPaths[0].is_closed


def test_generator_emits_bounded_multi_pass_job_with_tabs():
    square = [Point3D(0, 0), Point3D(-2, 0), Point3D(-2, -2),
              Point3D(0, -2), Point3D(0, 0)]
    paths = cncPathsClass(points3D=square, distPerTab=1.0, tabWidth=0.2)
    generator = _generator(paths)
    codes = generator.Generate()
    assert codes[:6] == ["G20", "G90", "G17", "G94", "F60", "G0 Z0.2"]
    assert codes[-1] == "M2"
    plunge_depths = [float(code.split("Z", 1)[1].split()[0])
                     for code in codes if code.startswith("G1 Z")]
    assert min(plunge_depths) == pytest.approx(-0.3)
    assert -0.2 in plunge_depths
    assert all(depth >= -0.3 for depth in plunge_depths)
    assert any(code == "G1 X-2 Y-2" for code in codes)


def test_hole_contours_are_ordered_before_outer_contours():
    outer = [Point3D(0, 0), Point3D(-4, 0), Point3D(-4, -4),
             Point3D(0, -4), Point3D(0, 0)]
    inner = [Point3D(-1, -1), Point3D(-2, -1), Point3D(-2, -2),
             Point3D(-1, -2), Point3D(-1, -1)]
    paths = cncPathsClass(points3D=outer)
    inner_path = cncPathsClass(points3D=inner).cncPaths[0]
    paths.cncPaths.append(inner_path)
    paths.orderCncHolePathsFirst()
    assert paths.cncPaths[0] is inner_path


@pytest.mark.parametrize(
    "overrides",
    [{"depthPerPass": 0}, {"safeHeight": -1}, {"materialThickness": math.nan}],
)
def test_generator_rejects_unsafe_parameters(overrides):
    paths = cncPathsClass(points3D=[Point3D(0, 0), Point3D(1, 0)])
    with pytest.raises(ValueError):
        _generator(paths, **overrides)


def test_repository_svg_generates_complete_dry_run():
    paths = cncPathsClass(inputSvgFile="puzzles2.svg", pointsPerCurve=30,
                          distPerTab=7.87, tabWidth=0.25,
                          cutterDiameter=0.125, convertSvfToIn=True)
    paths.orderCncHolePathsFirst()
    paths.orderPartialCutsFirst()
    generator = _generator(paths, materialThickness=0.471,
                           depthBelowMaterial=0.06, depthPerPass=0.107,
                           cutFeedRate=79, safeHeight=0.25, tabHeight=0.12)
    codes = generator.Generate()
    assert len(paths.cncPaths) >= 20
    assert len(codes) > 1000
    assert all("nan" not in code.lower() and "inf" not in code.lower() for code in codes)
    x_values = [float(word[1:]) for code in codes for word in code.split()
                if word.startswith("X")]
    y_values = [float(word[1:]) for code in codes for word in code.split()
                if word.startswith("Y")]
    assert min(x_values) >= -13.0
    assert max(x_values) <= 0.1
    assert min(y_values) >= -31.0
    assert max(y_values) <= 0.1


def test_qt_send_path_reaches_real_generator_without_serial():
    from app_config import Configuration
    from qt_ui.bed_coords import BedTransform
    from qt_ui.context import AppContext
    from qt_ui.machine_controller import MachineController
    from qt_ui.toolpath_loader import load_svg, svg_paths_as_tuples

    class FakeSender:
        def __init__(self):
            self.call = None

        def is_busy(self):
            return False

        def send_svf(self, *_args):
            raise AssertionError("run_async should own execution")

        def run_async(self, name, function, *args):
            self.call = (name, function, args)
            return True

    config = Configuration.get_default()
    context = AppContext(config, BedTransform.from_config(config))
    context.svg = load_svg("puzzles2.svg", config.cutting_parameters.cutter_diameter)
    context.svg_paths, context.svg_colors = svg_paths_as_tuples(context.svg)
    context.path_offsets = [[-10.0, -4.0] for _ in context.svg_paths]
    context.sender = FakeSender()

    MachineController(context).send_svg()
    name, _function, args = context.sender.call
    assert name == "send_svf"
    sent_paths = args[0]
    generator = cncGcodeGeneratorClass(
        sent_paths,
        config.cutting_parameters.material_thickness,
        config.cutting_parameters.depth_below_material,
        config.cutting_parameters.depth_per_pass,
        config.cutting_parameters.cut_feed_rate,
        config.cutting_parameters.safe_height,
        config.cutting_parameters.tab_height,
        useMM=False,
    )
    codes = generator.Generate()
    assert len(codes) > 1000
    assert codes[-1] == "M2"
