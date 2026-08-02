"""Pure port of the monolith's bed-view coordinate math (no Qt, no cv2).

Scene coordinates in the Qt bed view are exactly the monolith's overlay pixel
space: a bedViewSizePixels-square image of the warped bed. Two mappings exist
in the monolith and are preserved here verbatim:

- ``pixels_to_inches`` (OverlayGcode._pixel_to_inches): click position ->
  machine inches, including the reference-plate correction that snaps the
  camera's estimate of the touch plate onto its measured location.
- ``phy_to_pixels`` (OverlayGcode.phyPointsToPixels): machine inches ->
  overlay pixels for drawing, deliberately WITHOUT the plate correction --
  overlay parity with what the matplotlib UI renders.

``inches_to_pixels`` is the exact inverse of ``pixels_to_inches`` and is used
for anything that must round-trip with clicks (e.g. the live router marker:
clicking where the marker sits should return the machine position).
"""

from typing import List, Sequence, Tuple

XY = Tuple[float, float]


class BedTransform:
    def __init__(self, bed_view_pixels: float, bed_size_y: float,
                 left_box_x: float, right_box_x: float):
        self.bed_view_pixels = float(bed_view_pixels)
        self.bed_size_y = float(bed_size_y)
        self.left_box_x = float(left_box_x)
        self.right_box_x = float(right_box_x)
        # reference-plate correction state (monolith: camRefCenter /
        # refPlateMeasuredLoc). Equal until a probe measures the plate.
        self.cam_ref_center: XY = (0.0, 0.0)
        self.ref_plate_measured: XY = (0.0, 0.0)

    @classmethod
    def from_config(cls, config) -> "BedTransform":
        """Build from an app_config.Configuration (get_bed_size /
        get_left_box_ref / get_right_box_ref accessors)."""
        return cls(
            bed_view_pixels=config.vision_settings.bed_view_size_pixels,
            bed_size_y=config.get_bed_size().Y,
            left_box_x=config.get_left_box_ref().X,
            right_box_x=config.get_right_box_ref().X,
        )

    # -- click mapping (with plate correction) ------------------------------

    def pixels_to_inches(self, px: float, py: float) -> XY:
        x = px / self.bed_view_pixels * (self.left_box_x - self.right_box_x)
        x = x + self.right_box_x
        y = py / self.bed_view_pixels * self.bed_size_y
        x = x - self.cam_ref_center[0] + self.ref_plate_measured[0]
        y = y - self.cam_ref_center[1] + self.ref_plate_measured[1]
        return x, y

    def inches_to_pixels(self, x: float, y: float) -> XY:
        x = x - self.ref_plate_measured[0] + self.cam_ref_center[0]
        y = y - self.ref_plate_measured[1] + self.cam_ref_center[1]
        px = (x - self.right_box_x) * self.bed_view_pixels / (self.left_box_x - self.right_box_x)
        py = y * self.bed_view_pixels / self.bed_size_y
        return px, py

    # -- overlay mapping (parity with phyPointsToPixels, no correction) -----

    def phy_to_pixels(self, x: float, y: float) -> XY:
        px = (x - self.right_box_x) * self.bed_view_pixels / (self.left_box_x - self.right_box_x)
        py = y * self.bed_view_pixels / self.bed_size_y
        return px, py

    def tool_width_pixels(self, cutter_diameter: float) -> int:
        """Overlay stroke width for the cutter (monolith toolWidth formula)."""
        return max(1, round(abs(cutter_diameter * self.bed_view_pixels /
                                (self.left_box_x - self.right_box_x))))

    # -- reference plate (port of OverlayGcode.set_ref_loc) -----------------

    def set_ref_from_pixels(self, ref_pixels: Sequence[Sequence[float]]) -> List[XY]:
        """Take the touch plate's corner pixels (bed-view space), record the
        camera's estimate of its center, and return the corners in inches
        (the monolith's refPoints, used to seed the probing sequence)."""
        ref_points = [self.pixels_to_inches(p[0], p[1]) for p in ref_pixels]
        avg_x = sum(p[0] for p in ref_points) / len(ref_points)
        avg_y = sum(p[1] for p in ref_points) / len(ref_points)
        self.cam_ref_center = (avg_x, avg_y)
        # measured location starts at the camera estimate; a probe refines it
        self.ref_plate_measured = (avg_x, avg_y)
        return ref_points
