"""Session state shared by the Qt views (no Qt imports -- plain state).

Mirrors the monolith's OverlayGcode session fields: loaded toolpath, GUI
placement (offsets / rotation / selected path), drawn points, the workpiece
frame, and the (optional) GCodeSender. Views read from here and call
MachineController; background jobs replace `workpiece_frame` by reference
swap (atomic), read back on the busy->idle edge.
"""

from typing import List, Optional, Tuple

from qt_ui import overlay_model
from qt_ui.bed_coords import BedTransform

XY = Tuple[float, float]


class AppContext:
    def __init__(self, config, transform: BedTransform):
        self.config = config
        self.transform = transform

        # bed imagery (BGR ndarrays) + calibration artifacts
        self.bed_image = None            # warped overhead view
        self.raw_frame = None            # annotated camera frame (calibration tab)
        self.calibration_info = None     # calibrate_bed detection details for the UI
        self.touch_plate_px: List[XY] = []   # plate corners, bed-view pixels
        self.ref_points: List[XY] = []       # plate corners, inches (probing seed)

        # loaded toolpath: exactly one of svg / gcode active
        self.svg = None                  # pristine cncPathsClass (send_svf input)
        self.svg_file: Optional[str] = None
        self.svg_paths: List[List[XY]] = []
        self.svg_colors: List = []
        self.path_offsets: List[List[float]] = []
        self.path_index: int = -1        # -1 = all paths
        self.gcode_file: Optional[str] = None
        self.gcode_points: List[XY] = []
        self.gcode_powers: List[float] = []

        # placement + drawing (monolith semantics)
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.rotation = 0.0              # degrees
        self.drawn_points: List[XY] = [] # inches

        # machine
        self.sender = None               # GCodeSender or None (offline mode)
        self.grbl_caps = None            # populated from the boot banner
        self.workpiece_frame = None      # WorkpieceFrame or None
        self.probe_targets = []          # last proposed ProbeTargets (preview)

    # -- overlay building ---------------------------------------------------

    def overlay_scene_paths(self):
        """Current toolpath as bed-view-pixel polylines (same math as send)."""
        if self.svg is not None:
            polys = overlay_model.svg_overlay(self.svg_paths, self.svg_colors,
                                              self.path_offsets, self.rotation)
        elif self.gcode_points:
            polys = overlay_model.gcode_overlay(self.gcode_points, self.gcode_powers,
                                                self.x_offset, self.y_offset,
                                                self.rotation)
        else:
            polys = []
        return overlay_model.to_scene(polys, self.transform)

    def drawn_scene_path(self):
        return overlay_model.to_scene(
            overlay_model.drawn_overlay(self.drawn_points), self.transform)

    def tool_width_px(self) -> int:
        return self.transform.tool_width_pixels(
            self.config.cutting_parameters.cutter_diameter)

    # -- probing support (port of OverlayGcode._workCoordPaths) -------------

    def work_coord_paths(self) -> List[List[XY]]:
        """Current cut paths in work coordinates, as the send paths emit them
        -- input for propose_z_targets."""
        paths: List[List[XY]] = []
        if self.drawn_points:
            rx, ry = self.transform.ref_plate_measured
            paths.append([(x - rx, y - ry) for x, y in self.drawn_points])
        if self.svg is not None:
            polys = overlay_model.svg_overlay(self.svg_paths, self.svg_colors,
                                              self.path_offsets, self.rotation)
            paths.extend(p["points"] for p in polys)
        elif self.gcode_points:
            paths.append(list(self.gcode_points))
        return paths
