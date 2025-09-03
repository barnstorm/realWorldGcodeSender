"""ChArUco marker detection and processing."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from config.settings import Point3D

logger = logging.getLogger(__name__)


@dataclass
class DetectedMarker:
    """Detected marker information."""
    id: int
    corners: np.ndarray
    center: Point3D
    physical_location: Optional[Point3D] = None


@dataclass
class MarkerDetectionResult:
    """Result of marker detection operation."""
    markers: List[DetectedMarker]
    left_markers: List[DetectedMarker]
    right_markers: List[DetectedMarker]
    image_with_annotations: Optional[np.ndarray] = None


class ChArucoDetector:
    """ChArUco marker detector for CNC bed calibration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ChArUco detector.
        
        Args:
            config: Marker configuration dictionary.
        """
        self.config = config
        
        # ArUco dictionary and detector
        dict_name = config.get('dictionary', 'DICT_4X4_100')
        try:
            # Try new OpenCV 4.7+ API first
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
            self.detector_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            try:
                # Fall back to older API
                self.aruco_dict = cv2.aruco.Dictionary_get(getattr(cv2.aruco, dict_name))
                self.detector_params = cv2.aruco.DetectorParameters_create()
            except AttributeError:
                # Last fallback - ArUco might be in contrib
                import cv2.aruco as aruco
                self.aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
                self.detector_params = aruco.DetectorParameters()
        
        # Physical configuration
        self.box_width = config.get('box_width', 0.745642857)
        self.id_to_location = config.get('id_locations', {})
        
        # Generate default ID locations if not provided
        if not self.id_to_location:
            self.id_to_location = self._generate_default_id_locations()
        
        logger.info(f"ChArUco detector initialized with {dict_name}")
    
    def detect_markers(self, image: np.ndarray, 
                      annotate: bool = True) -> MarkerDetectionResult:
        """Detect ChArUco markers in image.
        
        Args:
            image: Input image (BGR or grayscale).
            annotate: If True, draw annotations on image.
        
        Returns:
            MarkerDetectionResult with detected markers.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect ArUco markers
        try:
            # Try new OpenCV 4.7+ API
            boxes, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.detector_params
            )
        except:
            # Fall back to older API
            boxes, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict)
        
        markers = []
        left_markers = []
        right_markers = []
        
        if ids is not None:
            for box, marker_id in zip(boxes, ids.flatten()):
                # Create marker object
                corners = self._sort_box_points(box[0])
                center = self._calculate_center(corners)
                
                marker = DetectedMarker(
                    id=int(marker_id),
                    corners=corners,
                    center=center,
                    physical_location=self._get_physical_location(marker_id)
                )
                
                markers.append(marker)
                
                # Categorize by side (IDs < 33 are right side, >= 33 are left side)
                if marker_id < 33:
                    right_markers.append(marker)
                elif 33 <= marker_id <= 65:
                    left_markers.append(marker)
        
        # Create annotated image if requested
        annotated_image = None
        if annotate and len(image.shape) == 3:
            annotated_image = image.copy()
            self._draw_annotations(annotated_image, markers)
        
        result = MarkerDetectionResult(
            markers=markers,
            left_markers=left_markers,
            right_markers=right_markers,
            image_with_annotations=annotated_image
        )
        
        logger.info(f"Detected {len(markers)} markers "
                   f"({len(left_markers)} left, {len(right_markers)} right)")
        
        return result
    
    def get_marker_by_id(self, markers: List[DetectedMarker], 
                        marker_id: int) -> Optional[DetectedMarker]:
        """Get marker by ID.
        
        Args:
            markers: List of markers to search.
            marker_id: ID to find.
        
        Returns:
            Marker with matching ID or None.
        """
        for marker in markers:
            if marker.id == marker_id:
                return marker
        return None
    
    def _sort_box_points(self, points: np.ndarray, right_side: bool = True) -> np.ndarray:
        """Sort box corner points in consistent order.
        
        Args:
            points: Array of 4 corner points.
            right_side: True if marker is on right side of machine.
        
        Returns:
            Sorted points array.
        """
        # Sort by X coordinate first
        sorted_x = sorted(points, key=lambda k: k[0])
        
        # Sort left and right pairs by Y coordinate
        left_two = sorted(sorted_x[0:2], key=lambda k: k[1])
        right_two = sorted(sorted_x[2:4], key=lambda k: k[1])
        
        if right_side:
            # For right side: min_z_min_y, min_z_max_y, max_z_max_y, max_z_min_y
            return np.array([left_two[1], left_two[0], right_two[0], right_two[1]])
        else:
            # For left side: order is different
            return np.array([right_two[1], right_two[0], left_two[0], left_two[1]])
    
    def _calculate_center(self, corners: np.ndarray) -> Point3D:
        """Calculate center point of marker.
        
        Args:
            corners: Corner points of marker.
        
        Returns:
            Center point as Point3D.
        """
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        return Point3D(float(center_x), float(center_y))
    
    def _get_physical_location(self, marker_id: int) -> Optional[Point3D]:
        """Get physical location of marker in machine coordinates.
        
        Args:
            marker_id: Marker ID.
        
        Returns:
            Physical location or None if not found.
        """
        if marker_id in self.id_to_location:
            loc = self.id_to_location[marker_id]
            if len(loc) >= 2:
                # Convert from [z, y] to Point3D (assuming x=0 for side markers)
                return Point3D(0.0, float(loc[1]), float(loc[0]))
        return None
    
    def _draw_annotations(self, image: np.ndarray, markers: List[DetectedMarker]) -> None:
        """Draw marker annotations on image.
        
        Args:
            image: Image to annotate (modified in place).
            markers: List of detected markers.
        """
        for marker in markers:
            # Draw marker outline
            cv2.aruco.drawDetectedMarkers(image, [marker.corners.reshape(1, -1, 2)], 
                                         np.array([marker.id]))
            
            # Draw marker ID
            center = (int(marker.center.x), int(marker.center.y))
            cv2.putText(image, str(marker.id), 
                       (center[0] + 10, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 255, 255), 2)
    
    def _generate_default_id_locations(self) -> Dict[int, List[float]]:
        """Generate default ID to location mapping.
        
        Returns:
            Dictionary mapping marker IDs to [z, y] locations.
        """
        # This is the original mapping from the code
        return {
            0: [2, 21], 1: [2, 19], 2: [2, 17], 3: [2, 16], 4: [2, 13],
            5: [2, 11], 6: [2, 9], 7: [2, 7], 8: [2, 5], 9: [2, 3], 10: [2, 1],
            11: [1, 20], 12: [1, 18], 13: [1, 16], 14: [1, 14], 15: [1, 12],
            16: [1, 10], 17: [1, 8], 18: [1, 6], 19: [1, 4], 20: [1, 2], 21: [1, 0],
            22: [0, 21], 23: [0, 19], 24: [0, 17], 25: [0, 15], 26: [0, 13],
            27: [0, 11], 28: [0, 9], 29: [0, 7], 30: [0, 5], 31: [0, 3], 32: [0, 1],
            33: [0, 20], 34: [0, 18], 35: [0, 16], 36: [0, 14], 37: [0, 12],
            38: [0, 10], 39: [0, 8], 40: [0, 6], 41: [0, 4], 42: [0, 2], 43: [0, 0],
            44: [1, 21], 45: [1, 19], 46: [1, 17], 47: [1, 15], 48: [1, 13],
            49: [1, 11], 50: [1, 9], 51: [1, 7], 52: [1, 5], 53: [1, 3], 54: [1, 1],
            55: [2, 20], 56: [2, 18], 57: [2, 16], 58: [2, 14], 59: [2, 12],
            60: [2, 10], 61: [2, 8], 62: [2, 6], 63: [2, 4], 64: [2, 2], 65: [2, 0]
        }
    
    def filter_markers_by_side(self, markers: List[DetectedMarker], 
                              right_side: bool = True) -> List[DetectedMarker]:
        """Filter markers by machine side.
        
        Args:
            markers: List of markers to filter.
            right_side: True for right side, False for left side.
        
        Returns:
            Filtered list of markers.
        """
        if right_side:
            return [m for m in markers if m.id < 33]
        else:
            return [m for m in markers if 33 <= m.id <= 65]
    
    def validate_detection(self, result: MarkerDetectionResult, 
                          min_markers: int = 4) -> bool:
        """Validate marker detection result.
        
        Args:
            result: Detection result to validate.
            min_markers: Minimum number of markers required.
        
        Returns:
            True if detection is valid.
        """
        total_markers = len(result.markers)
        left_markers = len(result.left_markers)
        right_markers = len(result.right_markers)
        
        is_valid = (
            total_markers >= min_markers and
            left_markers >= 2 and
            right_markers >= 2
        )
        
        if not is_valid:
            logger.warning(f"Invalid detection: {total_markers} total markers, "
                          f"{left_markers} left, {right_markers} right")
        
        return is_valid