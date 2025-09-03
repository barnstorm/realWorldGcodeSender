"""Basic vision module tests that don't require ArUco."""

import unittest
import numpy as np
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.calibration.homography import CoordinateTransformer
from vision.calibration.calibration_manager import CalibrationManager
from vision.detection.probe_finder import ProbeDetector
from vision.camera.static import StaticImageCamera
from config.settings import Point3D


class TestCoordinateTransformer(unittest.TestCase):
    """Test coordinate transformation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'bed_size': {'x': 35.0, 'y': 35.0, 'z': 3.75},
            'box_width': 0.75,
            'right_ref_position': {'x': 3.83, 'y': -34.2, 'z': -1.9},
            'left_ref_position': {'x': -39.17, 'y': -34.2, 'z': -1.03}
        }
        self.transformer = CoordinateTransformer(self.config)
    
    def test_transformer_initialization(self):
        """Test transformer initialization."""
        self.assertFalse(self.transformer.is_calibrated)
        self.assertIsNotNone(self.transformer.bed_size)
        self.assertEqual(self.transformer.box_width, 0.75)
        self.assertIsNotNone(self.transformer.right_box_ref)
        self.assertIsNotNone(self.transformer.left_box_ref)
    
    def test_uncalibrated_transformations(self):
        """Test transformations on uncalibrated transformer."""
        result = self.transformer.pixel_to_physical((100, 100))
        self.assertIsNone(result)
        
        result = self.transformer.physical_to_pixel(Point3D(1, 1, 0))
        self.assertIsNone(result)
    
    def test_get_calibration_info(self):
        """Test calibration info retrieval."""
        info = self.transformer.get_calibration_info()
        
        self.assertIn('is_calibrated', info)
        self.assertIn('bed_size', info)
        self.assertIn('box_width', info)
        self.assertFalse(info['is_calibrated'])
        
        # Check Point3D serialization
        bed_dict = info['bed_size']
        self.assertIn('x', bed_dict)
        self.assertIn('y', bed_dict)
        self.assertIn('z', bed_dict)
        self.assertEqual(bed_dict['x'], 35.0)


class TestProbeDetector(unittest.TestCase):
    """Test probe plate detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'min_area': 100,
            'max_area': 10000,
            'aspect_ratio_tolerance': 0.3,
            'circularity_threshold': 0.5,
            'use_color_detection': False
        }
        self.detector = ProbeDetector(self.config)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.min_area, 100)
        self.assertEqual(self.detector.max_area, 10000)
        self.assertFalse(self.detector.use_color_detection)
        self.assertEqual(self.detector.aspect_ratio_tolerance, 0.3)
        self.assertEqual(self.detector.circularity_threshold, 0.5)
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        processed = self.detector._preprocess_image(image)
        
        self.assertEqual(len(processed.shape), 2)  # Should be grayscale
        self.assertEqual(processed.dtype, np.uint8)
        self.assertEqual(processed.shape, (480, 640))
    
    def test_detect_empty_image(self):
        """Test detection on empty image."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = self.detector.detect_probe_targets(image)
        self.assertEqual(len(result.targets), 0)
        self.assertIsNone(result.best_target)
        self.assertIsNotNone(result.image_with_annotations)  # Always creates annotated image
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # Create dummy corners for a square
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).reshape(-1, 1, 2)
        
        # Test with ideal parameters
        confidence = self.detector._calculate_confidence(
            area=5000,  # Mid-range area
            aspect_ratio=1.0,  # Perfect square
            circularity=0.9,  # Very circular
            corners=corners
        )
        
        self.assertGreater(confidence, 0.7)  # Should be high confidence
        self.assertLessEqual(confidence, 1.0)
        self.assertGreaterEqual(confidence, 0.0)
    
    def test_get_detection_info(self):
        """Test detection info retrieval."""
        info = self.detector.get_detection_info()
        
        self.assertIn('min_area', info)
        self.assertIn('max_area', info)
        self.assertIn('use_color_detection', info)
        self.assertIn('probe_dimensions', info)
        
        self.assertEqual(info['min_area'], 100)
        self.assertEqual(info['max_area'], 10000)


class TestStaticImageCamera(unittest.TestCase):
    """Test static image camera."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test images using numpy (avoiding OpenCV dependency)
        self.test_images = []
        for i in range(3):
            img_path = self.temp_path / f"test_{i}.png"
            # Create a simple test image as raw data
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Use cv2 to save if available, otherwise just create the file
            try:
                import cv2
                cv2.imwrite(str(img_path), test_image)
                self.test_images.append(img_path)
            except:
                # Create empty files for testing structure
                with open(img_path, 'wb') as f:
                    f.write(b'fake_image_data')
                self.test_images.append(img_path)
        
        self.config = {
            'file_path': str(self.temp_path),
            'auto_discover': True
        }
        self.camera = StaticImageCamera(self.config)
    
    def test_camera_initialization(self):
        """Test camera initialization."""
        self.assertFalse(self.camera.is_open)
        self.assertTrue(self.camera.auto_discover)
        self.assertEqual(self.camera.current_index, 0)
        self.assertEqual(len(self.camera.image_list), 0)  # Not opened yet
    
    def test_get_camera_info_unopened(self):
        """Test camera info on unopened camera."""
        info = self.camera.get_camera_info()
        
        self.assertIn('source', info)
        self.assertIn('total_images', info)
        self.assertIn('current_index', info)
        self.assertEqual(info['source'], 'static')
        self.assertEqual(info['total_images'], 0)  # Not opened yet


class TestCalibrationManager(unittest.TestCase):
    """Test calibration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = CalibrationManager(Path(self.temp_dir))
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        self.assertTrue(self.manager.calibration_dir.exists())
        self.assertEqual(len(self.manager.sessions), 0)
        
        # Check file paths are set correctly
        self.assertTrue(str(self.manager.homography_file).endswith('homography.npz'))
        self.assertTrue(str(self.manager.camera_matrix_file).endswith('camera_matrix.npy'))
    
    def test_get_calibration_status(self):
        """Test calibration status retrieval."""
        status = self.manager.get_calibration_status()
        
        self.assertIn('homography_available', status)
        self.assertIn('camera_calibration_available', status)
        self.assertIn('total_sessions', status)
        self.assertIn('calibration_directory', status)
        
        self.assertFalse(status['homography_available'])
        self.assertFalse(status['camera_calibration_available'])
        self.assertEqual(status['total_sessions'], 0)
    
    def test_session_history(self):
        """Test session history management."""
        # Initially empty
        self.assertEqual(len(self.manager.sessions), 0)
        
        # Save and load should work without errors
        self.manager.save_session_history()
        self.manager.load_session_history()
        
        # Should still be empty
        self.assertEqual(len(self.manager.sessions), 0)
        
        # Check files exist
        self.assertTrue(self.manager.session_log_file.exists())


if __name__ == '__main__':
    unittest.main()