"""Unit tests for configuration management."""

import unittest
import tempfile
from pathlib import Path
import yaml
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import (
    Point3D, MachineConfig, MarkerConfig, CameraConfig, 
    CalibrationConfig, CuttingConfig, ApplicationConfig
)
from config.loader import ConfigLoader


class TestPoint3D(unittest.TestCase):
    """Test Point3D class."""
    
    def test_creation(self):
        """Test Point3D creation."""
        p = Point3D(1.0, 2.0, 3.0)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 2.0)
        self.assertEqual(p.z, 3.0)
    
    def test_default_z(self):
        """Test default z value."""
        p = Point3D(1.0, 2.0)
        self.assertEqual(p.z, 0.0)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        p = Point3D(1.0, 2.0, 3.0)
        d = p.to_dict()
        self.assertEqual(d, {"x": 1.0, "y": 2.0, "z": 3.0})
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"x": 1.0, "y": 2.0, "z": 3.0}
        p = Point3D.from_dict(d)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 2.0)
        self.assertEqual(p.z, 3.0)


class TestMachineConfig(unittest.TestCase):
    """Test MachineConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MachineConfig()
        self.assertEqual(config.type, "grbl")
        self.assertEqual(config.baudrate, 115200)
        self.assertTrue(config.soft_limits)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = MachineConfig()
        d = config.to_dict()
        self.assertIn("type", d)
        self.assertIn("baudrate", d)
        self.assertIn("bed_size", d)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "type": "custom",
            "baudrate": 9600,
            "soft_limits": False
        }
        config = MachineConfig.from_dict(d)
        self.assertEqual(config.type, "custom")
        self.assertEqual(config.baudrate, 9600)
        self.assertFalse(config.soft_limits)


class TestConfigLoader(unittest.TestCase):
    """Test ConfigLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.loader = ConfigLoader(self.temp_path)
    
    def test_load_yaml(self):
        """Test loading YAML configuration."""
        # Create test YAML file
        yaml_file = self.temp_path / "test.yaml"
        config_data = {
            "machine": {
                "type": "test",
                "baudrate": 57600
            }
        }
        with open(yaml_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load configuration
        config = self.loader.load(yaml_file)
        self.assertEqual(config.machine.type, "test")
        self.assertEqual(config.machine.baudrate, 57600)
    
    def test_load_json(self):
        """Test loading JSON configuration."""
        # Create test JSON file
        json_file = self.temp_path / "test.json"
        config_data = {
            "machine": {
                "type": "json_test",
                "port": "COM5"
            }
        }
        with open(json_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load configuration
        config = self.loader.load(json_file)
        self.assertEqual(config.machine.type, "json_test")
        self.assertEqual(config.machine.port, "COM5")
    
    def test_save_yaml(self):
        """Test saving YAML configuration."""
        config = ApplicationConfig()
        config.machine.type = "save_test"
        
        yaml_file = self.temp_path / "save_test.yaml"
        self.loader.save(config, yaml_file)
        
        # Verify file was created
        self.assertTrue(yaml_file.exists())
        
        # Load and verify
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        self.assertEqual(data["machine"]["type"], "save_test")
    
    def test_validation(self):
        """Test configuration validation."""
        config = ApplicationConfig()
        
        # Valid configuration
        self.assertTrue(self.loader.validate(config))
        
        # Invalid baudrate
        config.machine.baudrate = -1
        self.assertFalse(self.loader.validate(config))
        
        # Invalid bed size
        config.machine.baudrate = 115200
        config.machine.bed_size.x = -10
        self.assertFalse(self.loader.validate(config))


class TestApplicationConfig(unittest.TestCase):
    """Test ApplicationConfig class."""
    
    def test_complete_config(self):
        """Test complete application configuration."""
        config = ApplicationConfig()
        
        # Check all sub-configurations exist
        self.assertIsNotNone(config.machine)
        self.assertIsNotNone(config.calibration)
        self.assertIsNotNone(config.cutting)
        
        # Check default values
        self.assertEqual(config.ui_backend, "matplotlib")
        self.assertEqual(config.language, "en")
        self.assertFalse(config.debug_mode)
    
    def test_roundtrip(self):
        """Test configuration roundtrip (to_dict and from_dict)."""
        config1 = ApplicationConfig()
        config1.machine.type = "roundtrip_test"
        config1.cutting.feedrate = 123.45
        config1.debug_mode = True
        
        # Convert to dict and back
        d = config1.to_dict()
        config2 = ApplicationConfig.from_dict(d)
        
        # Verify values preserved
        self.assertEqual(config2.machine.type, "roundtrip_test")
        self.assertEqual(config2.cutting.feedrate, 123.45)
        self.assertTrue(config2.debug_mode)


if __name__ == '__main__':
    unittest.main()