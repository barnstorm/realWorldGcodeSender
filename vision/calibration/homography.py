"""Homography and coordinate transformation for CNC bed calibration."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from config.settings import Point3D
from .marker_detector import DetectedMarker

logger = logging.getLogger(__name__)


@dataclass
class CalibrationPoints:
    """Calibration point pairs for homography computation."""
    pixel_points: np.ndarray
    physical_points: np.ndarray
    side: str  # 'left' or 'right'


@dataclass
class HomographyResult:
    """Result of homography computation."""
    homography_matrix: np.ndarray
    inverse_matrix: np.ndarray
    reprojection_error: float
    valid: bool = True


class CoordinateTransformer:
    """Handles coordinate transformations between pixel and physical coordinates."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize coordinate transformer.
        
        Args:
            config: Calibration configuration dictionary.
        """
        self.config = config
        
        # Physical bed configuration
        self.bed_size = Point3D.from_dict(config.get('bed_size', {'x': 35.0, 'y': 35.0, 'z': 3.75}))
        self.box_width = config.get('box_width', 0.745642857)
        
        # Reference box positions
        right_ref = config.get('right_ref_position', {'x': 3.83, 'y': -34.2, 'z': -1.9025})
        left_ref = config.get('left_ref_position', {'x': -39.17, 'y': -34.2, 'z': -1.03})
        
        self.right_box_ref = Point3D.from_dict(right_ref)
        self.left_box_ref = Point3D.from_dict(left_ref)
        
        # Homography matrices
        self.left_homography = None
        self.right_homography = None
        self.bed_homography = None
        self.bed_inverse_homography = None
        
        # Calibration state
        self.is_calibrated = False
        
        logger.info("Coordinate transformer initialized")
    
    def calibrate_from_markers(self, left_markers: List[DetectedMarker], 
                             right_markers: List[DetectedMarker], 
                             image_shape: Tuple[int, int]) -> bool:
        """Calibrate coordinate transformation from detected markers.
        
        Args:
            left_markers: List of left side markers.
            right_markers: List of right side markers.
            image_shape: Shape of the image (height, width).
        
        Returns:
            True if calibration successful.
        """
        try:
            # Compute homographies for each side
            left_result = self._compute_side_homography(left_markers, 'left')
            right_result = self._compute_side_homography(right_markers, 'right')
            
            if not (left_result.valid and right_result.valid):
                logger.error("Failed to compute side homographies")
                return False
            
            self.left_homography = left_result
            self.right_homography = right_result
            
            # Compute bed homography
            bed_result = self._compute_bed_homography(image_shape)
            if not bed_result.valid:
                logger.error("Failed to compute bed homography")
                return False
            
            self.bed_homography = bed_result.homography_matrix
            self.bed_inverse_homography = bed_result.inverse_matrix
            
            self.is_calibrated = True
            logger.info("Coordinate transformation calibrated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
    
    def pixel_to_physical(self, pixel_point: Tuple[float, float]) -> Optional[Point3D]:
        """Convert pixel coordinates to physical coordinates.
        
        Args:
            pixel_point: Pixel coordinates (x, y).
        
        Returns:
            Physical coordinates or None if not calibrated.
        """
        if not self.is_calibrated:
            logger.warning("Transformer not calibrated")
            return None
        
        try:
            # Convert pixel to bed coordinates
            pixel_array = np.array([pixel_point], dtype=np.float32).reshape(-1, 1, 2)
            bed_coords = cv2.perspectiveTransform(pixel_array, self.bed_inverse_homography)
            
            bed_x = bed_coords[0][0][0]
            bed_y = bed_coords[0][0][1]
            
            # Convert bed coordinates to physical coordinates
            # This assumes the bed coordinate system maps directly to physical inches
            physical_x = bed_x / 100.0  # Convert from bed pixels to inches
            physical_y = bed_y / 100.0
            physical_z = 0.0  # Z coordinate from bed surface
            
            return Point3D(physical_x, physical_y, physical_z)
            
        except Exception as e:
            logger.error(f"Error converting pixel to physical coordinates: {e}")
            return None
    
    def physical_to_pixel(self, physical_point: Point3D) -> Optional[Tuple[float, float]]:
        """Convert physical coordinates to pixel coordinates.
        
        Args:
            physical_point: Physical coordinates.
        
        Returns:
            Pixel coordinates or None if not calibrated.
        """
        if not self.is_calibrated:
            logger.warning("Transformer not calibrated")
            return None
        
        try:
            # Convert physical to bed coordinates
            bed_x = physical_point.x * 100.0  # Convert inches to bed pixels
            bed_y = physical_point.y * 100.0
            
            # Convert bed coordinates to pixel coordinates
            bed_array = np.array([(bed_x, bed_y)], dtype=np.float32).reshape(-1, 1, 2)
            pixel_coords = cv2.perspectiveTransform(bed_array, self.bed_homography)
            
            pixel_x = float(pixel_coords[0][0][0])
            pixel_y = float(pixel_coords[0][0][1])
            
            return (pixel_x, pixel_y)
            
        except Exception as e:
            logger.error(f"Error converting physical to pixel coordinates: {e}")
            return None
    
    def transform_path_to_pixels(self, path_points: List[Point3D]) -> List[Tuple[float, float]]:
        """Transform a list of physical path points to pixel coordinates.
        
        Args:
            path_points: List of physical coordinates.
        
        Returns:
            List of pixel coordinates.
        """
        pixel_points = []
        
        for point in path_points:
            pixel_point = self.physical_to_pixel(point)
            if pixel_point:
                pixel_points.append(pixel_point)
        
        return pixel_points
    
    def _compute_side_homography(self, markers: List[DetectedMarker], 
                                side: str) -> HomographyResult:
        """Compute homography for one side of the machine.
        
        Args:
            markers: List of markers for this side.
            side: Side identifier ('left' or 'right').
        
        Returns:
            HomographyResult with computed matrices.
        """
        if len(markers) < 4:
            logger.error(f"Need at least 4 markers for {side} side, got {len(markers)}")
            return HomographyResult(np.eye(3), np.eye(3), float('inf'), False)
        
        try:
            # Extract pixel and physical points
            pixel_points = []
            physical_points = []
            
            for marker in markers:
                if marker.physical_location:
                    # Use marker center for pixel coordinates
                    pixel_points.append([marker.center.x, marker.center.y])
                    
                    # Use physical location
                    phys_loc = marker.physical_location
                    physical_points.append([phys_loc.z, phys_loc.y])  # Note: z,y order
            
            if len(pixel_points) < 4:
                logger.error(f"Need at least 4 markers with physical locations for {side} side")
                return HomographyResult(np.eye(3), np.eye(3), float('inf'), False)
            
            pixel_array = np.array(pixel_points, dtype=np.float32)
            physical_array = np.array(physical_points, dtype=np.float32)
            
            # Compute homography
            homography, status = cv2.findHomography(
                physical_array, pixel_array,
                cv2.RANSAC, 5.0
            )
            
            if homography is None:
                logger.error(f"Failed to compute homography for {side} side")
                return HomographyResult(np.eye(3), np.eye(3), float('inf'), False)
            
            # Compute inverse
            inverse_homography = np.linalg.inv(homography)
            
            # Compute reprojection error
            reprojected = cv2.perspectiveTransform(
                physical_array.reshape(-1, 1, 2), homography
            )
            error = np.mean(np.sqrt(np.sum((reprojected.squeeze() - pixel_array) ** 2, axis=1)))
            
            logger.info(f"{side} side homography computed with error: {error:.2f} pixels")
            
            return HomographyResult(homography, inverse_homography, error, True)
            
        except Exception as e:
            logger.error(f"Error computing {side} side homography: {e}")
            return HomographyResult(np.eye(3), np.eye(3), float('inf'), False)
    
    def _compute_bed_homography(self, image_shape: Tuple[int, int]) -> HomographyResult:
        """Compute homography for the entire bed.
        
        Args:
            image_shape: Shape of the image (height, width).
        
        Returns:
            HomographyResult with bed homography matrices.
        """
        try:
            height, width = image_shape[:2]
            
            # Get bed corner points in physical coordinates
            bed_corners_physical = np.array([
                [self.right_box_ref.z, 0.0],  # Right side, Y=0
                [self.bed_size.z, 0.0],       # Left side, Y=0  
                [self.bed_size.z, self.bed_size.y],  # Left side, Y=max
                [self.right_box_ref.z, self.bed_size.y]  # Right side, Y=max
            ], dtype=np.float32)
            
            # Transform to pixel coordinates using side homographies
            right_points = cv2.perspectiveTransform(
                bed_corners_physical[[0, 3]].reshape(-1, 1, 2), 
                self.right_homography.homography_matrix
            )
            
            left_points = cv2.perspectiveTransform(
                bed_corners_physical[[1, 2]].reshape(-1, 1, 2), 
                self.left_homography.homography_matrix
            )
            
            # Combine points
            bed_corners_pixels = np.array([
                right_points[0][0],  # Right, Y=0
                left_points[0][0],   # Left, Y=0
                left_points[1][0],   # Left, Y=max
                right_points[1][0]   # Right, Y=max
            ], dtype=np.float32)
            
            # Define bed coordinate system (overhead view)
            bed_view_size = 1400  # From original code
            bed_corners_normalized = np.array([
                [bed_view_size, 0.0],
                [bed_view_size, float(width)],
                [0.0, 0.0],
                [0.0, float(width)]
            ], dtype=np.float32)
            
            # Compute homographies
            bed_to_pixel, _ = cv2.findHomography(
                bed_corners_normalized, bed_corners_pixels
            )
            pixel_to_bed, _ = cv2.findHomography(
                bed_corners_pixels, bed_corners_normalized
            )
            
            if bed_to_pixel is None or pixel_to_bed is None:
                logger.error("Failed to compute bed homography")
                return HomographyResult(np.eye(3), np.eye(3), float('inf'), False)
            
            # Compute reprojection error
            reprojected = cv2.perspectiveTransform(
                bed_corners_normalized.reshape(-1, 1, 2), bed_to_pixel
            )
            error = np.mean(np.sqrt(np.sum((reprojected.squeeze() - bed_corners_pixels) ** 2, axis=1)))
            
            logger.info(f"Bed homography computed with error: {error:.2f} pixels")
            
            return HomographyResult(bed_to_pixel, pixel_to_bed, error, True)
            
        except Exception as e:
            logger.error(f"Error computing bed homography: {e}")
            return HomographyResult(np.eye(3), np.eye(3), float('inf'), False)
    
    def draw_bed_outline(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255), 
                        thickness: int = 3) -> np.ndarray:
        """Draw bed outline on image.
        
        Args:
            image: Image to draw on.
            color: Line color (BGR).
            thickness: Line thickness.
        
        Returns:
            Image with bed outline drawn.
        """
        if not self.is_calibrated:
            return image
        
        try:
            # Define bed corners in normalized coordinates
            height, width = image.shape[:2]
            bed_corners = np.array([
                [1400, 0.0],
                [1400, float(width)],
                [0.0, 0.0],
                [0.0, float(width)]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            # Transform to pixel coordinates
            pixel_corners = cv2.perspectiveTransform(bed_corners, self.bed_homography)
            
            # Draw lines
            points = pixel_corners.squeeze().astype(np.int32)
            cv2.polylines(image, [points], True, color, thickness)
            
            return image
            
        except Exception as e:
            logger.error(f"Error drawing bed outline: {e}")
            return image
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get calibration information.
        
        Returns:
            Dictionary with calibration status and errors.
        """
        info = {
            'is_calibrated': self.is_calibrated,
            'bed_size': self.bed_size.to_dict(),
            'box_width': self.box_width
        }
        
        if self.left_homography:
            info['left_error'] = self.left_homography.reprojection_error
            
        if self.right_homography:
            info['right_error'] = self.right_homography.reprojection_error
        
        return info
    
    def save_calibration(self, file_path: str) -> bool:
        """Save calibration data to file.
        
        Args:
            file_path: Path to save calibration data.
        
        Returns:
            True if successful.
        """
        if not self.is_calibrated:
            logger.error("Cannot save uncalibrated transformer")
            return False
        
        try:
            calibration_data = {
                'bed_homography': self.bed_homography.tolist(),
                'bed_inverse_homography': self.bed_inverse_homography.tolist(),
                'left_homography': self.left_homography.homography_matrix.tolist(),
                'right_homography': self.right_homography.homography_matrix.tolist(),
                'config': self.config
            }
            
            np.savez(file_path, **calibration_data)
            logger.info(f"Calibration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
            return False
    
    def load_calibration(self, file_path: str) -> bool:
        """Load calibration data from file.
        
        Args:
            file_path: Path to load calibration data from.
        
        Returns:
            True if successful.
        """
        try:
            data = np.load(file_path, allow_pickle=True)
            
            self.bed_homography = np.array(data['bed_homography'])
            self.bed_inverse_homography = np.array(data['bed_inverse_homography'])
            
            # Reconstruct homography results
            left_matrix = np.array(data['left_homography'])
            right_matrix = np.array(data['right_homography'])
            
            self.left_homography = HomographyResult(
                left_matrix, np.linalg.inv(left_matrix), 0.0, True
            )
            self.right_homography = HomographyResult(
                right_matrix, np.linalg.inv(right_matrix), 0.0, True
            )
            
            self.is_calibrated = True
            logger.info(f"Calibration loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")
            return False