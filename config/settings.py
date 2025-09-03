"""Configuration settings classes for the CNC G-code sender application."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from pathlib import Path


@dataclass
class Point3D:
    """Represents a 3D point in space."""
    x: float
    y: float
    z: float = 0.0
    
    def __repr__(self):
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"
    
    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(x=data["x"], y=data["y"], z=data.get("z", 0.0))


@dataclass
class MachineConfig:
    """CNC machine physical configuration."""
    
    # Machine type
    type: str = "grbl"
    
    # Connection settings
    port: str = "auto"
    baudrate: int = 115200
    
    # Bed dimensions (in inches)
    bed_size: Point3D = field(default_factory=lambda: Point3D(35.0, 35.0, 3.75))
    
    # Origin position
    origin_position: str = "back_right_top"
    origin_coordinates: Point3D = field(default_factory=lambda: Point3D(0.0, 0.0, 0.0))
    
    # Limits
    soft_limits: bool = True
    hard_limits: bool = True
    max_feedrate: float = 1000.0  # inches per minute
    max_acceleration: float = 100.0  # inches per second squared
    
    def to_dict(self):
        return {
            "type": self.type,
            "port": self.port,
            "baudrate": self.baudrate,
            "bed_size": self.bed_size.to_dict(),
            "origin_position": self.origin_position,
            "origin_coordinates": self.origin_coordinates.to_dict(),
            "soft_limits": self.soft_limits,
            "hard_limits": self.hard_limits,
            "max_feedrate": self.max_feedrate,
            "max_acceleration": self.max_acceleration
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            type=data.get("type", "grbl"),
            port=data.get("port", "auto"),
            baudrate=data.get("baudrate", 115200),
            bed_size=Point3D.from_dict(data.get("bed_size", {"x": 35.0, "y": 35.0, "z": 3.75})),
            origin_position=data.get("origin_position", "back_right_top"),
            origin_coordinates=Point3D.from_dict(data.get("origin_coordinates", {"x": 0, "y": 0, "z": 0})),
            soft_limits=data.get("soft_limits", True),
            hard_limits=data.get("hard_limits", True),
            max_feedrate=data.get("max_feedrate", 1000.0),
            max_acceleration=data.get("max_acceleration", 100.0)
        )


@dataclass
class MarkerConfig:
    """ChArUco marker configuration."""
    
    # Marker type
    type: str = "charuco"
    dictionary: str = "DICT_6X6_250"
    
    # Physical dimensions
    box_width: float = 0.745642857  # inches
    board_size: Tuple[int, int] = (10, 7)  # columns x rows
    
    # Marker positions on machine
    right_ref_position: Point3D = field(default_factory=lambda: Point3D(3.83, -34.2, -1.9025))
    left_ref_position: Point3D = field(default_factory=lambda: Point3D(-39.17, -34.2, -1.03))
    
    # Heights at different Y positions
    right_far_height: float = 2.38 - 0.471  # Height at Y=0
    left_far_height: float = 3.1375 - 0.471
    
    # ID to location mapping (stored as dict for easy serialization)
    id_locations: Dict[int, List[int]] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "type": self.type,
            "dictionary": self.dictionary,
            "box_width": self.box_width,
            "board_size": list(self.board_size),
            "right_ref_position": self.right_ref_position.to_dict(),
            "left_ref_position": self.left_ref_position.to_dict(),
            "right_far_height": self.right_far_height,
            "left_far_height": self.left_far_height,
            "id_locations": self.id_locations
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            type=data.get("type", "charuco"),
            dictionary=data.get("dictionary", "DICT_6X6_250"),
            box_width=data.get("box_width", 0.745642857),
            board_size=tuple(data.get("board_size", [10, 7])),
            right_ref_position=Point3D.from_dict(data.get("right_ref_position", {"x": 3.83, "y": -34.2, "z": -1.9025})),
            left_ref_position=Point3D.from_dict(data.get("left_ref_position", {"x": -39.17, "y": -34.2, "z": -1.03})),
            right_far_height=data.get("right_far_height", 1.909),
            left_far_height=data.get("left_far_height", 2.6665),
            id_locations=data.get("id_locations", {})
        )


@dataclass
class CameraConfig:
    """Camera configuration settings."""
    
    # Camera source
    source: str = "webcam"  # webcam, file, or network
    device_index: int = 0  # For webcam
    file_path: Optional[str] = None  # For file source
    
    # Resolution
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    
    # Camera settings
    exposure: str = "auto"  # auto or manual value
    brightness: float = 0.5
    contrast: float = 0.5
    
    # Display settings
    display_size: Tuple[int, int] = (1400, 1050)
    
    def to_dict(self):
        return {
            "source": self.source,
            "device_index": self.device_index,
            "file_path": self.file_path,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "exposure": self.exposure,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "display_size": list(self.display_size)
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            source=data.get("source", "webcam"),
            device_index=data.get("device_index", 0),
            file_path=data.get("file_path"),
            resolution=tuple(data.get("resolution", [1920, 1080])),
            fps=data.get("fps", 30),
            exposure=data.get("exposure", "auto"),
            brightness=data.get("brightness", 0.5),
            contrast=data.get("contrast", 0.5),
            display_size=tuple(data.get("display_size", [1400, 1050]))
        )


@dataclass
class CalibrationConfig:
    """Calibration settings."""
    
    # Marker configuration
    markers: MarkerConfig = field(default_factory=MarkerConfig)
    
    # Camera configuration
    camera: CameraConfig = field(default_factory=CameraConfig)
    
    # Calibration file paths
    calibration_dir: Path = field(default_factory=lambda: Path("calibration"))
    camera_matrix_file: str = "camera_matrix.npy"
    distortion_coeffs_file: str = "distortion_coeffs.npy"
    homography_file: str = "homography.npy"
    
    # Calibration flags
    auto_calibrate: bool = False
    save_calibration_images: bool = True
    
    def to_dict(self):
        return {
            "markers": self.markers.to_dict(),
            "camera": self.camera.to_dict(),
            "calibration_dir": str(self.calibration_dir),
            "camera_matrix_file": self.camera_matrix_file,
            "distortion_coeffs_file": self.distortion_coeffs_file,
            "homography_file": self.homography_file,
            "auto_calibrate": self.auto_calibrate,
            "save_calibration_images": self.save_calibration_images
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            markers=MarkerConfig.from_dict(data.get("markers", {})),
            camera=CameraConfig.from_dict(data.get("camera", {})),
            calibration_dir=Path(data.get("calibration_dir", "calibration")),
            camera_matrix_file=data.get("camera_matrix_file", "camera_matrix.npy"),
            distortion_coeffs_file=data.get("distortion_coeffs_file", "distortion_coeffs.npy"),
            homography_file=data.get("homography_file", "homography.npy"),
            auto_calibrate=data.get("auto_calibrate", False),
            save_calibration_images=data.get("save_calibration_images", True)
        )


@dataclass
class CuttingConfig:
    """Cutting operation parameters."""
    
    # Material settings
    material_thickness: float = 0.471  # inches
    material_type: str = "wood"
    
    # Tool settings
    cutter_diameter: float = 0.125  # inches
    cutter_type: str = "endmill"
    
    # Cutting parameters
    cut_depth_per_pass: float = 0.125  # inches
    feedrate: float = 400.0  # inches per minute
    plunge_rate: float = 100.0  # inches per minute
    spindle_speed: float = 10000.0  # RPM
    
    # Tab settings
    tab_height: float = 0.1  # inches
    tab_width: float = 0.25  # inches
    tabs_per_inch: float = 0.25
    auto_tabs: bool = True
    
    # Safety settings
    safe_z: float = 0.5  # inches above material
    clearance_height: float = 0.1  # inches above material for moves
    
    def to_dict(self):
        return {
            "material_thickness": self.material_thickness,
            "material_type": self.material_type,
            "cutter_diameter": self.cutter_diameter,
            "cutter_type": self.cutter_type,
            "cut_depth_per_pass": self.cut_depth_per_pass,
            "feedrate": self.feedrate,
            "plunge_rate": self.plunge_rate,
            "spindle_speed": self.spindle_speed,
            "tab_height": self.tab_height,
            "tab_width": self.tab_width,
            "tabs_per_inch": self.tabs_per_inch,
            "auto_tabs": self.auto_tabs,
            "safe_z": self.safe_z,
            "clearance_height": self.clearance_height
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            material_thickness=data.get("material_thickness", 0.471),
            material_type=data.get("material_type", "wood"),
            cutter_diameter=data.get("cutter_diameter", 0.125),
            cutter_type=data.get("cutter_type", "endmill"),
            cut_depth_per_pass=data.get("cut_depth_per_pass", 0.125),
            feedrate=data.get("feedrate", 400.0),
            plunge_rate=data.get("plunge_rate", 100.0),
            spindle_speed=data.get("spindle_speed", 10000.0),
            tab_height=data.get("tab_height", 0.1),
            tab_width=data.get("tab_width", 0.25),
            tabs_per_inch=data.get("tabs_per_inch", 0.25),
            auto_tabs=data.get("auto_tabs", True),
            safe_z=data.get("safe_z", 0.5),
            clearance_height=data.get("clearance_height", 0.1)
        )


@dataclass
class ApplicationConfig:
    """Main application configuration."""
    
    # Component configurations
    machine: MachineConfig = field(default_factory=MachineConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    cutting: CuttingConfig = field(default_factory=CuttingConfig)
    
    # Application settings
    ui_backend: str = "matplotlib"  # matplotlib, qt, or web
    theme: str = "default"
    language: str = "en"
    
    # File paths
    project_dir: Path = field(default_factory=lambda: Path("projects"))
    gcode_dir: Path = field(default_factory=lambda: Path("gcode"))
    svg_dir: Path = field(default_factory=lambda: Path("svg"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # Debug settings
    debug_mode: bool = False
    simulation_mode: bool = False
    verbose_logging: bool = False
    
    def to_dict(self):
        return {
            "machine": self.machine.to_dict(),
            "calibration": self.calibration.to_dict(),
            "cutting": self.cutting.to_dict(),
            "ui_backend": self.ui_backend,
            "theme": self.theme,
            "language": self.language,
            "project_dir": str(self.project_dir),
            "gcode_dir": str(self.gcode_dir),
            "svg_dir": str(self.svg_dir),
            "log_dir": str(self.log_dir),
            "debug_mode": self.debug_mode,
            "simulation_mode": self.simulation_mode,
            "verbose_logging": self.verbose_logging
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            machine=MachineConfig.from_dict(data.get("machine", {})),
            calibration=CalibrationConfig.from_dict(data.get("calibration", {})),
            cutting=CuttingConfig.from_dict(data.get("cutting", {})),
            ui_backend=data.get("ui_backend", "matplotlib"),
            theme=data.get("theme", "default"),
            language=data.get("language", "en"),
            project_dir=Path(data.get("project_dir", "projects")),
            gcode_dir=Path(data.get("gcode_dir", "gcode")),
            svg_dir=Path(data.get("svg_dir", "svg")),
            log_dir=Path(data.get("log_dir", "logs")),
            debug_mode=data.get("debug_mode", False),
            simulation_mode=data.get("simulation_mode", False),
            verbose_logging=data.get("verbose_logging", False)
        )