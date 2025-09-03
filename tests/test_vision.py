"""Unit tests for vision module."""

import unittest
import numpy as np
import cv2
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision.calibration.marker_detector import ChArucoDetector, DetectedMarker
from vision.calibration.homography import CoordinateTransformer
from vision.calibration.calibration_manager import CalibrationManager
from vision.detection.probe_finder import ProbeDetector
from vision.camera.static import StaticImageCamera
from config.settings import Point3D


class TestChArucoDetector(unittest.TestCase):
    """Test ChArUco marker detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'dictionary': 'DICT_4X4_100',
            'box_width': 0.75,
            'id_locations': {
                0: [2, 21], 1: [2, 19], 2: [2, 17],
                33: [0, 20], 34: [0, 18], 35: [0, 16]
            }
        }
        self.detector = ChArucoDetector(self.config)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector.aruco_dict)
        self.assertIsNotNone(self.detector.detector_params)
        self.assertEqual(self.detector.box_width, 0.75)
    
    def test_sort_box_points(self):
        """Test box point sorting."""
        # Create a simple square
        points = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32)
        
        # Test right side sorting
        sorted_right = self.detector._sort_box_points(points, right_side=True)
        self.assertEqual(sorted_right.shape, (4, 2))
        
        # Test left side sorting  
        sorted_left = self.detector._sort_box_points(points, right_side=False)
        self.assertEqual(sorted_left.shape, (4, 2))
    
    def test_calculate_center(self):
        """Test center point calculation."""
        corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        center = self.detector._calculate_center(corners)
        
        self.assertIsInstance(center, Point3D)
        self.assertAlmostEqual(center.x, 5.0, places=1)
        self.assertAlmostEqual(center.y, 5.0, places=1)
    
    def test_get_physical_location(self):
        """Test physical location retrieval."""
        # Test valid ID
        location = self.detector._get_physical_location(0)
        self.assertIsNotNone(location)
        self.assertEqual(location.y, 21.0)
        self.assertEqual(location.z, 2.0)
        
        # Test invalid ID
        location = self.detector._get_physical_location(999)
        self.assertIsNone(location)
    
    def test_detect_markers_empty_image(self):
        """Test marker detection on empty image."""
        # Create blank image
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = self.detector.detect_markers(image)
        self.assertEqual(len(result.markers), 0)
        self.assertEqual(len(result.left_markers), 0)
        self.assertEqual(len(result.right_markers), 0)
    
    def test_filter_markers_by_side(self):
        """Test marker filtering by side."""
        # Create test markers
        markers = [
            DetectedMarker(id=0, corners=np.zeros((4, 2)), center=Point3D(0, 0)),
            DetectedMarker(id=33, corners=np.zeros((4, 2)), center=Point3D(0, 0)),
            DetectedMarker(id=50, corners=np.zeros((4, 2)), center=Point3D(0, 0)),
        ]
        
        # Test right side filter
        right_markers = self.detector.filter_markers_by_side(markers, right_side=True)
        self.assertEqual(len(right_markers), 1)
        self.assertEqual(right_markers[0].id, 0)
        
        # Test left side filter
        left_markers = self.detector.filter_markers_by_side(markers, right_side=False)
        self.assertEqual(len(left_markers), 2)


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


class TestProbeDetector(unittest.TestCase):
    """Test probe plate detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'min_area': 100,
            'max_area': 10000,
            'aspect_ratio_tolerance': 0.3,
            'circularity_threshold': 0.5
        }
        self.detector = ProbeDetector(self.config)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.min_area, 100)
        self.assertEqual(self.detector.max_area, 10000)
        self.assertFalse(self.detector.use_color_detection)
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        processed = self.detector._preprocess_image(image)
        
        self.assertEqual(len(processed.shape), 2)  # Should be grayscale
        self.assertEqual(processed.dtype, np.uint8)
    
    def test_detect_empty_image(self):
        """Test detection on empty image."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = self.detector.detect_probe_targets(image)
        self.assertEqual(len(result.targets), 0)
        self.assertIsNone(result.best_target)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # Test with ideal parameters
        confidence = self.detector._calculate_confidence(
            area=5000,  # Mid-range area
            aspect_ratio=1.0,  # Perfect square
            circularity=0.9,  # Very circular
            corners=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).reshape(-1, 1, 2)
        )
        
        self.assertGreater(confidence, 0.7)  # Should be high confidence
        self.assertLessEqual(confidence, 1.0)


class TestStaticImageCamera(unittest.TestCase):
    """Test static image camera."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test images
        self.test_images = []
        for i in range(3):
            img_path = self.temp_path / f"test_{i}.jpg"
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), test_image)
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
    
    def test_open_directory(self):
        """Test opening directory with images."""
        success = self.camera.open()
        
        self.assertTrue(success)
        self.assertTrue(self.camera.is_open)
        self.assertEqual(len(self.camera.image_list), 3)
    
    def test_capture_image(self):
        """Test image capture."""
        self.camera.open()
        
        image = self.camera.capture()
        self.assertIsNotNone(image)
        self.assertEqual(len(image.shape), 3)  # Should be BGR
        self.assertEqual(image.shape[:2], (480, 640))
    
    def test_navigation(self):
        """Test image navigation."""
        self.camera.open()
        
        # Test next image
        next_img = self.camera.next_image()
        self.assertIsNotNone(next_img)
        self.assertEqual(self.camera.current_index, 1)
        
        # Test previous image
        prev_img = self.camera.previous_image()
        self.assertIsNotNone(prev_img)
        self.assertEqual(self.camera.current_index, 0)
    
    def test_get_camera_info(self):
        """Test camera info retrieval."""
        self.camera.open()
        
        info = self.camera.get_camera_info()
        
        self.assertIn('source', info)
        self.assertIn('total_images', info)
        self.assertIn('current_index', info)
        self.assertEqual(info['source'], 'static')
        self.assertEqual(info['total_images'], 3)


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
    
    def test_get_calibration_status(self):
        """Test calibration status retrieval."""
        status = self.manager.get_calibration_status()
        
        self.assertIn('homography_available', status)
        self.assertIn('camera_calibration_available', status)
        self.assertIn('total_sessions', status)
        self.assertFalse(status['homography_available'])
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


class TestVisionIntegration(unittest.TestCase):
    """Integration tests for vision components."""
    
    def test_detector_transformer_integration(self):
        """Test integration between detector and transformer."""
        detector_config = {
            'dictionary': 'DICT_4X4_100',
            'box_width': 0.75,
            'id_locations': {0: [2, 21], 33: [0, 20]}
        }
        detector = ChArucoDetector(detector_config)
        
        transformer_config = {
            'bed_size': {'x': 35.0, 'y': 35.0, 'z': 3.75},
            'box_width': 0.75
        }
        transformer = CoordinateTransformer(transformer_config)
        
        # Should be able to create instances without errors
        self.assertIsNotNone(detector)
        self.assertIsNotNone(transformer)
        
        # Transformer should not be calibrated initially
        self.assertFalse(transformer.is_calibrated)


if __name__ == '__main__':
    unittest.main()