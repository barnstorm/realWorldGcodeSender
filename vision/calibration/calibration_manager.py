"""Calibration management system for storing and loading calibration data."""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import cv2

from .marker_detector import ChArucoDetector, MarkerDetectionResult
from .homography import CoordinateTransformer
from config.settings import Point3D

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSession:
    """Information about a calibration session."""
    timestamp: str
    image_count: int
    marker_count: int
    left_markers: int
    right_markers: int
    reprojection_error: float
    calibration_quality: str
    notes: Optional[str] = None


@dataclass
class CameraCalibration:
    """Camera intrinsic calibration data."""
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple
    reprojection_error: float
    calibration_date: str


class CalibrationManager:
    """Manages calibration data storage, loading, and validation."""
    
    def __init__(self, calibration_dir: Path):
        """Initialize calibration manager.
        
        Args:
            calibration_dir: Directory to store calibration files.
        """
        self.calibration_dir = Path(calibration_dir)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.homography_file = self.calibration_dir / "homography.npz"
        self.camera_matrix_file = self.calibration_dir / "camera_matrix.npy"
        self.distortion_file = self.calibration_dir / "distortion_coeffs.npy"
        self.session_log_file = self.calibration_dir / "calibration_sessions.json"
        self.config_file = self.calibration_dir / "calibration_config.json"
        
        # Session history
        self.sessions: List[CalibrationSession] = []
        self.load_session_history()
        
        logger.info(f"Calibration manager initialized: {self.calibration_dir}")
    
    def save_homography_calibration(self, transformer: CoordinateTransformer,
                                   session_info: Dict[str, Any]) -> bool:
        """Save homography calibration data.
        
        Args:
            transformer: Calibrated coordinate transformer.
            session_info: Information about the calibration session.
        
        Returns:
            True if successful.
        """
        if not transformer.is_calibrated:
            logger.error("Cannot save uncalibrated transformer")
            return False
        
        try:
            # Save homography matrices
            calibration_data = {
                'bed_homography': transformer.bed_homography,
                'bed_inverse_homography': transformer.bed_inverse_homography,
                'left_homography_matrix': transformer.left_homography.homography_matrix,
                'left_homography_inverse': transformer.left_homography.inverse_matrix,
                'left_reprojection_error': transformer.left_homography.reprojection_error,
                'right_homography_matrix': transformer.right_homography.homography_matrix,
                'right_homography_inverse': transformer.right_homography.inverse_matrix,
                'right_reprojection_error': transformer.right_homography.reprojection_error,
                'calibration_config': transformer.config,
                'timestamp': datetime.now().isoformat(),
                'session_info': session_info
            }
            
            np.savez_compressed(self.homography_file, **calibration_data)
            
            # Log session
            self._log_calibration_session(session_info, transformer)
            
            logger.info(f"Homography calibration saved to {self.homography_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving homography calibration: {e}")
            return False
    
    def load_homography_calibration(self, transformer: CoordinateTransformer) -> bool:
        """Load homography calibration data.
        
        Args:
            transformer: Coordinate transformer to load calibration into.
        
        Returns:
            True if successful.
        """
        if not self.homography_file.exists():
            logger.warning(f"Homography calibration file not found: {self.homography_file}")
            return False
        
        try:
            data = np.load(self.homography_file, allow_pickle=True)
            
            # Load matrices
            transformer.bed_homography = data['bed_homography']
            transformer.bed_inverse_homography = data['bed_inverse_homography']
            
            # Reconstruct homography results
            from .homography import HomographyResult
            
            transformer.left_homography = HomographyResult(
                homography_matrix=data['left_homography_matrix'],
                inverse_matrix=data['left_homography_inverse'],
                reprojection_error=float(data['left_reprojection_error']),
                valid=True
            )
            
            transformer.right_homography = HomographyResult(
                homography_matrix=data['right_homography_matrix'],
                inverse_matrix=data['right_homography_inverse'],
                reprojection_error=float(data['right_reprojection_error']),
                valid=True
            )
            
            # Update configuration if available
            if 'calibration_config' in data:
                config_dict = data['calibration_config'].item()
                transformer.config.update(config_dict)
            
            transformer.is_calibrated = True
            
            logger.info(f"Homography calibration loaded from {self.homography_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading homography calibration: {e}")
            return False
    
    def save_camera_calibration(self, camera_matrix: np.ndarray,
                               distortion_coeffs: np.ndarray,
                               image_size: tuple,
                               reprojection_error: float) -> bool:
        """Save camera intrinsic calibration data.
        
        Args:
            camera_matrix: Camera intrinsic matrix.
            distortion_coeffs: Distortion coefficients.
            image_size: Image size (width, height).
            reprojection_error: Calibration reprojection error.
        
        Returns:
            True if successful.
        """
        try:
            # Save camera matrix
            np.save(self.camera_matrix_file, camera_matrix)
            
            # Save distortion coefficients
            np.save(self.distortion_file, distortion_coeffs)
            
            # Save calibration metadata
            calibration_data = CameraCalibration(
                camera_matrix=camera_matrix,
                distortion_coefficients=distortion_coeffs,
                image_size=image_size,
                reprojection_error=reprojection_error,
                calibration_date=datetime.now().isoformat()
            )
            
            # Save metadata as JSON (excluding numpy arrays)
            metadata = {
                'image_size': image_size,
                'reprojection_error': reprojection_error,
                'calibration_date': calibration_data.calibration_date
            }
            
            metadata_file = self.calibration_dir / "camera_calibration_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Camera calibration saved with error: {reprojection_error:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving camera calibration: {e}")
            return False
    
    def load_camera_calibration(self) -> Optional[CameraCalibration]:
        """Load camera intrinsic calibration data.
        
        Returns:
            CameraCalibration object or None if not found.
        """
        if not (self.camera_matrix_file.exists() and self.distortion_file.exists()):
            logger.warning("Camera calibration files not found")
            return None
        
        try:
            # Load matrices
            camera_matrix = np.load(self.camera_matrix_file)
            distortion_coeffs = np.load(self.distortion_file)
            
            # Load metadata
            metadata_file = self.calibration_dir / "camera_calibration_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    'image_size': (1920, 1080),
                    'reprojection_error': 0.0,
                    'calibration_date': 'unknown'
                }
            
            calibration = CameraCalibration(
                camera_matrix=camera_matrix,
                distortion_coefficients=distortion_coeffs,
                image_size=tuple(metadata['image_size']),
                reprojection_error=metadata['reprojection_error'],
                calibration_date=metadata['calibration_date']
            )
            
            logger.info(f"Camera calibration loaded (error: {calibration.reprojection_error:.2f})")
            return calibration
            
        except Exception as e:
            logger.error(f"Error loading camera calibration: {e}")
            return None
    
    def perform_full_calibration(self, images: List[np.ndarray],
                                detector: ChArucoDetector) -> tuple:
        """Perform complete calibration from multiple images.
        
        Args:
            images: List of calibration images.
            detector: ChArUco detector instance.
        
        Returns:
            Tuple of (transformer, session_success) where transformer is the
            calibrated CoordinateTransformer and session_success is bool.
        """
        logger.info(f"Starting full calibration with {len(images)} images")
        
        # Initialize transformer
        transformer = CoordinateTransformer(detector.config)
        
        # Collect all marker detections
        all_left_markers = []
        all_right_markers = []
        total_markers = 0
        
        for i, image in enumerate(images):
            try:
                result = detector.detect_markers(image, annotate=False)
                
                if not detector.validate_detection(result):
                    logger.warning(f"Invalid detection in image {i}")
                    continue
                
                all_left_markers.extend(result.left_markers)
                all_right_markers.extend(result.right_markers)
                total_markers += len(result.markers)
                
                logger.debug(f"Image {i}: {len(result.left_markers)} left, "
                           f"{len(result.right_markers)} right markers")
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                continue
        
        if not all_left_markers or not all_right_markers:
            logger.error("Insufficient markers for calibration")
            return transformer, False
        
        # Calibrate transformer
        image_shape = images[0].shape[:2]
        success = transformer.calibrate_from_markers(
            all_left_markers, all_right_markers, image_shape
        )
        
        if not success:
            logger.error("Transformer calibration failed")
            return transformer, False
        
        # Create session info
        avg_error = (transformer.left_homography.reprojection_error + 
                    transformer.right_homography.reprojection_error) / 2
        
        session_info = {
            'image_count': len(images),
            'marker_count': total_markers,
            'left_markers': len(all_left_markers),
            'right_markers': len(all_right_markers),
            'reprojection_error': avg_error
        }
        
        # Save calibration
        self.save_homography_calibration(transformer, session_info)
        
        logger.info(f"Full calibration completed with error: {avg_error:.2f}")
        return transformer, True
    
    def _log_calibration_session(self, session_info: Dict[str, Any],
                                transformer: CoordinateTransformer) -> None:
        """Log calibration session information.
        
        Args:
            session_info: Session information dictionary.
            transformer: Calibrated transformer.
        """
        try:
            # Determine calibration quality
            avg_error = (transformer.left_homography.reprojection_error + 
                        transformer.right_homography.reprojection_error) / 2
            
            if avg_error < 2.0:
                quality = "excellent"
            elif avg_error < 5.0:
                quality = "good"
            elif avg_error < 10.0:
                quality = "fair"
            else:
                quality = "poor"
            
            session = CalibrationSession(
                timestamp=datetime.now().isoformat(),
                image_count=session_info.get('image_count', 0),
                marker_count=session_info.get('marker_count', 0),
                left_markers=session_info.get('left_markers', 0),
                right_markers=session_info.get('right_markers', 0),
                reprojection_error=avg_error,
                calibration_quality=quality,
                notes=session_info.get('notes')
            )
            
            self.sessions.append(session)
            self.save_session_history()
            
        except Exception as e:
            logger.error(f"Error logging calibration session: {e}")
    
    def save_session_history(self) -> None:
        """Save session history to file."""
        try:
            sessions_data = [asdict(session) for session in self.sessions]
            with open(self.session_log_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session history: {e}")
    
    def load_session_history(self) -> None:
        """Load session history from file."""
        if not self.session_log_file.exists():
            return
        
        try:
            with open(self.session_log_file, 'r') as f:
                sessions_data = json.load(f)
            
            self.sessions = [CalibrationSession(**data) for data in sessions_data]
            logger.info(f"Loaded {len(self.sessions)} calibration sessions")
            
        except Exception as e:
            logger.error(f"Error loading session history: {e}")
            self.sessions = []
    
    def validate_calibration(self, transformer: CoordinateTransformer,
                           max_error: float = 10.0) -> bool:
        """Validate calibration quality.
        
        Args:
            transformer: Transformer to validate.
            max_error: Maximum acceptable reprojection error.
        
        Returns:
            True if calibration is valid.
        """
        if not transformer.is_calibrated:
            logger.error("Transformer is not calibrated")
            return False
        
        left_error = transformer.left_homography.reprojection_error
        right_error = transformer.right_homography.reprojection_error
        avg_error = (left_error + right_error) / 2
        
        if avg_error > max_error:
            logger.warning(f"Calibration error too high: {avg_error:.2f} > {max_error}")
            return False
        
        # Test coordinate transformations
        try:
            # Test pixel to physical conversion
            test_pixel = (100, 100)
            physical = transformer.pixel_to_physical(test_pixel)
            
            if physical is None:
                logger.error("Pixel to physical conversion failed")
                return False
            
            # Test round-trip conversion
            back_to_pixel = transformer.physical_to_pixel(physical)
            if back_to_pixel is None:
                logger.error("Physical to pixel conversion failed")
                return False
            
            # Check round-trip error
            pixel_error = np.sqrt((test_pixel[0] - back_to_pixel[0])**2 + 
                                (test_pixel[1] - back_to_pixel[1])**2)
            
            if pixel_error > max_error:
                logger.warning(f"Round-trip conversion error too high: {pixel_error:.2f}")
                return False
            
            logger.info(f"Calibration validation passed (avg error: {avg_error:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Error during calibration validation: {e}")
            return False
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get calibration status information.
        
        Returns:
            Dictionary with calibration status.
        """
        status = {
            'homography_available': self.homography_file.exists(),
            'camera_calibration_available': (self.camera_matrix_file.exists() and 
                                           self.distortion_file.exists()),
            'total_sessions': len(self.sessions),
            'calibration_directory': str(self.calibration_dir)
        }
        
        # Add latest session info if available
        if self.sessions:
            latest = self.sessions[-1]
            status['latest_session'] = {
                'timestamp': latest.timestamp,
                'quality': latest.calibration_quality,
                'error': latest.reprojection_error
            }
        
        return status
    
    def cleanup_old_calibrations(self, keep_count: int = 5) -> None:
        """Clean up old calibration files, keeping only the most recent.
        
        Args:
            keep_count: Number of recent calibrations to keep.
        """
        if len(self.sessions) <= keep_count:
            return
        
        try:
            # Keep only recent sessions
            self.sessions = self.sessions[-keep_count:]
            self.save_session_history()
            
            # Archive old calibration files
            archive_dir = self.calibration_dir / "archive"
            archive_dir.mkdir(exist_ok=True)
            
            logger.info(f"Cleaned up old calibrations, kept {keep_count} recent ones")
            
        except Exception as e:
            logger.error(f"Error cleaning up calibrations: {e}")