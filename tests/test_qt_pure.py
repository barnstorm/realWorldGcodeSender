import math

import pytest

from gerbil import Gerbil
from qt_ui.bed_coords import BedTransform
from qt_ui.grbl_caps import parse_boot_banner
from qt_ui.overlay_model import gcode_overlay, svg_overlay


def test_bed_transform_click_mapping_round_trips():
    transform = BedTransform(1400, -35, -39.17, 3.83)
    transform.cam_ref_center = (-10.0, -12.0)
    transform.ref_plate_measured = (-9.75, -12.2)

    for point in ((0, 0), (700, 700), (1399, 1200)):
        inches = transform.pixels_to_inches(*point)
        assert transform.inches_to_pixels(*inches) == pytest.approx(point)


def test_svg_overlay_uses_per_path_offsets_and_common_rotation_origin():
    result = svg_overlay(
        paths=[[(0, 0), (1, 0)], [(0, 0), (0, 1)]],
        colors=[(0, 0, 0), (0, 255, 0)],
        offsets=[(2, 3), (5, 6)],
        rotation_deg=90,
    )
    assert result[0]["points"][0] == pytest.approx((8, 3))
    assert result[1]["points"][0] == pytest.approx((5, 6))
    assert result[0]["powers"] == [0.0, 1.0]


def test_gcode_overlay_rotates_about_placement_point():
    result = gcode_overlay([(0, 0), (1, 0)], [0, 1], 4, -2, 90)[0]
    assert result["points"] == pytest.approx([(4, -2), (4, -1)])


@pytest.mark.parametrize(
    "banner,modern,fmt",
    [("Grbl 0.9j ['$' for help]", False, "comma"),
     ("Grbl 1.1h ['$' for help]", True, "pipe")],
)
def test_boot_banner_capabilities(banner, modern, fmt):
    caps = parse_boot_banner(banner)
    assert caps.modern is modern
    assert caps.report_format == fmt


def test_gerbil_parses_grbl_09_status():
    gerbil = Gerbil(lambda *_args: None)
    gerbil._update_state("<Idle,MPos:1.000,2.000,3.000,WPos:0.500,1.000,2.000>")
    assert gerbil.cmode == "Idle"
    assert gerbil.cmpos == (1, 2, 3)
    assert gerbil.cwpos == (0.5, 1, 2)


def test_gerbil_parses_grbl_11_status_with_wco():
    gerbil = Gerbil(lambda *_args: None)
    gerbil._update_state("<Idle|MPos:10.000,20.000,30.000|WCO:1.000,2.000,3.000|FS:0,0>")
    assert gerbil.cmode == "Idle"
    assert gerbil.cmpos == (10, 20, 30)
    assert gerbil.cwpos == (9, 18, 27)


def test_gerbil_ignores_malformed_status():
    gerbil = Gerbil(lambda *_args: None)
    before = (gerbil.cmode, gerbil.cmpos, gerbil.cwpos)
    gerbil._update_state("<not a status report>")
    assert (gerbil.cmode, gerbil.cmpos, gerbil.cwpos) == before

