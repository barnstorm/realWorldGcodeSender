"""
Configuration management for realWorldGcodeSender

This module provides centralized configuration management with JSON file persistence
and environment-specific overrides.
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


class Point3D:
    """Simple 3D point class for configuration compatibility"""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.X = x
        self.Y = y
        self.Z = z
    
    def __str__(self):
        return f"Point3D({self.X}, {self.Y}, {self.Z})"
    
    def __repr__(self):
        return self.__str__()


@dataclass
class PhysicalSetup:
    """Physical CNC machine setup parameters"""
    box_width: float = 0.745642857 * 1.01  # ChArUco marker box width in inches
    bed_size_x: float = -35.0  # Bed X dimension (negative from machine origin)
    bed_size_y: float = -35.0  # Bed Y dimension (negative from machine origin)  
    bed_size_z: float = -3.75  # Bed Z dimension (negative from machine origin)
    
    # Reference box positions relative to machine origin
    right_box_ref_x: float = 4.0 - 0.17
    right_box_ref_y: float = -34.0 - 0.2
    right_box_ref_z_offset: float = 2.3975  # Offset from bed surface
    
    left_box_ref_x: float = -39.0 - 0.17
    left_box_ref_y: float = -34.0 - 0.2
    left_box_ref_z_offset: float = 3.2  # Offset from bed surface
    
    # Heights at far end for slope calculation
    right_box_far_height: float = 2.38  # Height at Y=0 end
    left_box_far_height: float = 3.1375  # Height at Y=0 end


@dataclass
class CuttingParameters:
    """Cutting tool and material parameters"""
    material_thickness: float = 0.471  # Default material thickness in inches
    cutter_diameter: float = 0.125  # Default cutter diameter in inches
    cut_feed_rate: float = 79  # Cutting feed rate
    depth_per_pass: float = 0.107  # Depth per cutting pass
    depth_below_material: float = 0.06  # Cut depth below material surface
    safe_height: float = 0.25  # Safe Z height for rapid moves
    tab_height: float = 0.12  # Height of holding tabs


@dataclass
class VisionSettings:
    """Computer vision and display settings"""
    bed_view_size_pixels: int = 1400  # Display size for bed view
    camera_device_index: int = 1  # Camera device index (0, 1, 2, etc.)
    camera_width: int = 1280  # Camera capture width
    camera_height: int = 800  # Camera capture height


@dataclass
class CommunicationSettings:
    """CNC communication parameters"""
    com_port: str = "COM4"  # Serial port for CNC communication
    baud_rate: int = 115200  # Serial communication baud rate
    auto_detect_port: bool = False  # Automatically detect CNC port


@dataclass
class ProbingSettings:
    """Touch probe and reference plate settings"""
    plate_height: float = 0.472441  # Touch plate height (12mm)
    plate_width: float = 2.75591  # Touch plate width (70mm)
    probe_feed_rate_fast: float = 5.9  # Fast probing feed rate
    probe_feed_rate_slow: float = 1.5  # Slow probing feed rate
    dist_to_notch: float = 1.39173  # Distance to notch on touch plate


@dataclass
class Configuration:
    """Complete application configuration"""
    physical_setup: PhysicalSetup
    cutting_parameters: CuttingParameters
    vision_settings: VisionSettings
    communication_settings: CommunicationSettings
    probing_settings: ProbingSettings

    @classmethod
    def load_from_file(cls, config_file: str = "config.json") -> "Configuration":
        """Load configuration from JSON file"""
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    return cls.from_dict(data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
        
        return cls.get_default()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Configuration":
        """Create configuration from dictionary"""
        return cls(
            physical_setup=PhysicalSetup(**data.get("physical_setup", {})),
            cutting_parameters=CuttingParameters(**data.get("cutting_parameters", {})),
            vision_settings=VisionSettings(**data.get("vision_settings", {})),
            communication_settings=CommunicationSettings(**data.get("communication_settings", {})),
            probing_settings=ProbingSettings(**data.get("probing_settings", {}))
        )
    
    @classmethod
    def get_default(cls) -> "Configuration":
        """Get default configuration"""
        return cls(
            physical_setup=PhysicalSetup(),
            cutting_parameters=CuttingParameters(),
            vision_settings=VisionSettings(),
            communication_settings=CommunicationSettings(),
            probing_settings=ProbingSettings()
        )
    
    def save_to_file(self, config_file: str = "config.json") -> None:
        """Save configuration to JSON file"""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
        except IOError as e:
            print(f"Error saving config file: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "physical_setup": asdict(self.physical_setup),
            "cutting_parameters": asdict(self.cutting_parameters),
            "vision_settings": asdict(self.vision_settings),
            "communication_settings": asdict(self.communication_settings),
            "probing_settings": asdict(self.probing_settings)
        }
    
    # Helper methods to generate derived values for backward compatibility
    def get_bed_size(self) -> Point3D:
        """Get bed size as Point3D object"""
        return Point3D(
            self.physical_setup.bed_size_x,
            self.physical_setup.bed_size_y,
            self.physical_setup.bed_size_z
        )
    
    def get_right_box_ref(self) -> Point3D:
        """Get right reference box position"""
        return Point3D(
            self.physical_setup.right_box_ref_x,
            self.physical_setup.right_box_ref_y,
            self.physical_setup.bed_size_z + self.physical_setup.right_box_ref_z_offset - self.cutting_parameters.material_thickness
        )
    
    def get_left_box_ref(self) -> Point3D:
        """Get left reference box position"""
        return Point3D(
            self.physical_setup.left_box_ref_x,
            self.physical_setup.left_box_ref_y,
            self.physical_setup.bed_size_z + self.physical_setup.left_box_ref_z_offset - self.cutting_parameters.material_thickness
        )
    
    def get_right_slope(self) -> float:
        """Calculate right side slope for marker height adjustment"""
        right_box_ref_z = self.physical_setup.bed_size_z + self.physical_setup.right_box_ref_z_offset - self.cutting_parameters.material_thickness
        return (self.physical_setup.right_box_far_height - self.cutting_parameters.material_thickness - (right_box_ref_z - self.physical_setup.bed_size_z)) / 20.0
    
    def get_left_slope(self) -> float:
        """Calculate left side slope for marker height adjustment"""
        left_box_ref_z = self.physical_setup.bed_size_z + self.physical_setup.left_box_ref_z_offset - self.cutting_parameters.material_thickness
        return (self.physical_setup.left_box_far_height - self.cutting_parameters.material_thickness - (left_box_ref_z - self.physical_setup.bed_size_z)) / 20.0


# Global configuration instance
_config: Optional[Configuration] = None


def get_config() -> Configuration:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Configuration.load_from_file()
    return _config


def reload_config() -> Configuration:
    """Reload configuration from file"""
    global _config
    _config = Configuration.load_from_file()
    return _config


def save_config() -> None:
    """Save the current configuration to file"""
    if _config is not None:
        _config.save_to_file()