#!/usr/bin/env python3
"""Demo script to test the vision module functionality."""

import numpy as np
import tempfile
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

from vision.camera.static import StaticImageCamera
from vision.detection.probe_finder import ProbeDetector
from vision.calibration.homography import CoordinateTransformer
from vision.calibration.calibration_manager import CalibrationManager
from config.settings import Point3D
from utils.logging_config import setup_logging


def create_test_image(width=640, height=480):
    """Create a test image with some geometric shapes."""
    import cv2
    
    # Create blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colored rectangles that could look like probe plates
    cv2.rectangle(image, (100, 100), (150, 150), (0, 255, 0), -1)  # Green square
    cv2.rectangle(image, (300, 200), (360, 260), (255, 0, 0), -1)  # Blue square
    cv2.circle(image, (500, 300), 30, (0, 0, 255), -1)  # Red circle
    
    # Add some noise
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image


def test_static_camera():
    """Test static camera functionality."""
    print("\n=== Testing Static Camera ===")
    
    # Create temporary directory with test images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test images
        for i in range(3):
            img_path = temp_path / f"test_image_{i}.jpg"
            test_img = create_test_image()
            
            try:
                import cv2
                cv2.imwrite(str(img_path), test_img)
            except ImportError:
                # Create dummy file if cv2 not available
                with open(img_path, 'wb') as f:
                    f.write(b'dummy_image_data')
        
        # Test camera
        config = {'file_path': str(temp_path), 'auto_discover': True}
        camera = StaticImageCamera(config)
        
        # Open camera
        success = camera.open()
        print(f"Camera open: {success}")
        print(f"Found {camera.get_image_count()} images")
        
        # Test navigation
        if success and camera.get_image_count() > 1:
            print(f"Current index: {camera.get_current_index()}")
            
            next_img = camera.next_image()
            if next_img is not None:
                print(f"Next image shape: {next_img.shape}")
            
            prev_img = camera.previous_image()
            if prev_img is not None:
                print(f"Previous image shape: {prev_img.shape}")
        
        # Test info
        info = camera.get_camera_info()
        print(f"Camera info: {info}")
        
        camera.close()


def test_probe_detector():
    """Test probe detector functionality."""
    print("\n=== Testing Probe Detector ===")
    
    config = {
        'min_area': 500,
        'max_area': 5000,
        'aspect_ratio_tolerance': 0.5,
        'circularity_threshold': 0.3,
        'use_color_detection': False
    }
    
    detector = ProbeDetector(config)
    
    # Create test image
    test_image = create_test_image()
    
    # Detect probe targets
    result = detector.detect_probe_targets(test_image, annotate=True)
    
    print(f"Found {len(result.targets)} probe targets")
    
    for i, target in enumerate(result.targets):
        print(f"Target {i}: center=({target.center.x:.1f}, {target.center.y:.1f}), "
              f"area={target.area:.1f}, confidence={target.confidence:.2f}")
    
    if result.best_target:
        best = result.best_target
        print(f"Best target: center=({best.center.x:.1f}, {best.center.y:.1f}), "
              f"confidence={best.confidence:.2f}")
    
    # Test info
    info = detector.get_detection_info()
    print(f"Detector info: {info}")


def test_coordinate_transformer():
    """Test coordinate transformer functionality."""
    print("\n=== Testing Coordinate Transformer ===")
    
    config = {
        'bed_size': {'x': 35.0, 'y': 35.0, 'z': 3.75},
        'box_width': 0.75,
        'right_ref_position': {'x': 3.83, 'y': -34.2, 'z': -1.9},
        'left_ref_position': {'x': -39.17, 'y': -34.2, 'z': -1.03}
    }
    
    transformer = CoordinateTransformer(config)
    
    print(f"Transformer calibrated: {transformer.is_calibrated}")
    print(f"Bed size: {transformer.bed_size}")
    
    # Test uncalibrated transformations
    result = transformer.pixel_to_physical((320, 240))
    print(f"Uncalibrated pixel to physical: {result}")
    
    # Test info
    info = transformer.get_calibration_info()
    print(f"Calibration info: {info}")


def test_calibration_manager():
    """Test calibration manager functionality."""
    print("\n=== Testing Calibration Manager ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CalibrationManager(Path(temp_dir))
        
        print(f"Calibration directory: {manager.calibration_dir}")
        print(f"Session count: {len(manager.sessions)}")
        
        # Test status
        status = manager.get_calibration_status()
        print(f"Calibration status: {status}")
        
        # Test session history
        manager.save_session_history()
        manager.load_session_history()
        
        print(f"Session history loaded: {len(manager.sessions)} sessions")


def test_integration():
    """Test integration between components."""
    print("\n=== Testing Integration ===")
    
    # Create detector and transformer
    detector_config = {'min_area': 100, 'max_area': 10000}
    detector = ProbeDetector(detector_config)
    
    transformer_config = {
        'bed_size': {'x': 35.0, 'y': 35.0, 'z': 3.75},
        'box_width': 0.75
    }
    transformer = CoordinateTransformer(transformer_config)
    
    # Create test image
    test_image = create_test_image()
    
    # Detect with coordinate transformer (uncalibrated)
    result = detector.detect_probe_targets(test_image, transformer, annotate=False)
    
    print(f"Integration test: found {len(result.targets)} targets")
    
    for i, target in enumerate(result.targets):
        print(f"Target {i}: physical_location={target.physical_location}")


def main():
    """Run all vision module tests."""
    # Set up logging
    setup_logging(console_output=True, file_output=False, log_level="INFO")
    
    print("Starting Vision Module Demo")
    print("=" * 50)
    
    try:
        test_static_camera()
        test_probe_detector()
        test_coordinate_transformer()
        test_calibration_manager()
        test_integration()
        
        print("\n" + "=" * 50)
        print("[SUCCESS] Vision Module Demo Completed Successfully!")
        print("[SUCCESS] Phase 2: Vision Module Implementation Complete!")
        
    except Exception as e:
        print(f"\n[ERROR] Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()