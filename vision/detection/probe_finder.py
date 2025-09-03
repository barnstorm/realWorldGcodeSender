"""Probe plate detection for automatic probing operations."""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from dataclasses import dataclass
from config.settings import Point3D

logger = logging.getLogger(__name__)


@dataclass
class ProbeTarget:
    """Detected probe target information."""
    center: Point3D
    corners: np.ndarray
    area: float
    confidence: float
    physical_location: Optional[Point3D] = None


@dataclass
class ProbeDetectionResult:
    """Result of probe detection operation."""
    targets: List[ProbeTarget]
    best_target: Optional[ProbeTarget]
    image_with_annotations: Optional[np.ndarray] = None


class ProbeDetector:
    """Detects probe plates and targets in camera images."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize probe detector.
        
        Args:
            config: Probe detection configuration dictionary.
        """
        self.config = config
        
        # Detection parameters
        self.min_area = config.get('min_area', 1000)  # Minimum probe area in pixels
        self.max_area = config.get('max_area', 50000)  # Maximum probe area in pixels
        self.aspect_ratio_tolerance = config.get('aspect_ratio_tolerance', 0.3)
        self.circularity_threshold = config.get('circularity_threshold', 0.7)
        
        # Color detection parameters (for colored probe plates)
        self.use_color_detection = config.get('use_color_detection', False)
        self.probe_color_hsv = config.get('probe_color_hsv', [0, 100, 100])  # Red by default
        self.color_tolerance = config.get('color_tolerance', [10, 50, 50])
        
        # Edge detection parameters
        self.edge_threshold1 = config.get('edge_threshold1', 50)
        self.edge_threshold2 = config.get('edge_threshold2', 150)
        self.blur_kernel_size = config.get('blur_kernel_size', 5)
        
        # Probe plate dimensions (if known)
        self.probe_width = config.get('probe_width', 1.0)  # inches
        self.probe_height = config.get('probe_height', 1.0)  # inches
        
        logger.info("Probe detector initialized")
    
    def detect_probe_targets(self, image: np.ndarray, 
                           coordinate_transformer=None,
                           annotate: bool = True) -> ProbeDetectionResult:
        """Detect probe targets in image.
        
        Args:
            image: Input image (BGR or grayscale).
            coordinate_transformer: Optional coordinate transformer for physical locations.
            annotate: If True, create annotated image.
        
        Returns:
            ProbeDetectionResult with detected targets.
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Detect contours
            contours = self._find_contours(processed_image)
            
            # Filter and analyze contours
            targets = self._analyze_contours(contours, coordinate_transformer)
            
            # Find best target
            best_target = self._select_best_target(targets)
            
            # Create annotated image
            annotated_image = None
            if annotate:
                annotated_image = self._create_annotated_image(image, targets, best_target)
            
            result = ProbeDetectionResult(
                targets=targets,
                best_target=best_target,
                image_with_annotations=annotated_image
            )
            
            logger.info(f"Detected {len(targets)} probe targets")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting probe targets: {e}")
            return ProbeDetectionResult([], None, None)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for probe detection.
        
        Args:
            image: Input image.
        
        Returns:
            Preprocessed image.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Apply adaptive threshold or edge detection based on configuration
        if self.use_color_detection:
            # Use color-based detection
            processed = self._detect_by_color(image)
        else:
            # Use edge-based detection
            processed = cv2.Canny(blurred, self.edge_threshold1, self.edge_threshold2)
        
        return processed
    
    def _detect_by_color(self, image: np.ndarray) -> np.ndarray:
        """Detect probe plate by color.
        
        Args:
            image: Input BGR image.
        
        Returns:
            Binary mask of detected color.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color range
        probe_color = np.array(self.probe_color_hsv)
        tolerance = np.array(self.color_tolerance)
        
        lower_bound = probe_color - tolerance
        upper_bound = probe_color + tolerance
        
        # Create mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _find_contours(self, processed_image: np.ndarray) -> List[np.ndarray]:
        """Find contours in processed image.
        
        Args:
            processed_image: Preprocessed binary image.
        
        Returns:
            List of contours.
        """
        contours, _ = cv2.findContours(
            processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
    def _analyze_contours(self, contours: List[np.ndarray], 
                         coordinate_transformer=None) -> List[ProbeTarget]:
        """Analyze contours to find probe targets.
        
        Args:
            contours: List of contours to analyze.
            coordinate_transformer: Optional coordinate transformer.
        
        Returns:
            List of probe targets.
        """
        targets = []
        
        for contour in contours:
            try:
                # Basic measurements
                area = cv2.contourArea(contour)
                
                # Filter by area
                if not (self.min_area <= area <= self.max_area):
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Filter by aspect ratio (should be roughly square for probe plate)
                if not (1 - self.aspect_ratio_tolerance <= aspect_ratio <= 1 + self.aspect_ratio_tolerance):
                    continue
                
                # Calculate circularity/rectangularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Filter by circularity (probe plates should be reasonably circular/square)
                if circularity < self.circularity_threshold:
                    continue
                
                # Get center point
                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                
                center_x = moments["m10"] / moments["m00"]
                center_y = moments["m01"] / moments["m00"]
                center = Point3D(center_x, center_y, 0)
                
                # Get corner points
                corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # Calculate confidence based on shape characteristics
                confidence = self._calculate_confidence(area, aspect_ratio, circularity, corners)
                
                # Get physical location if transformer available
                physical_location = None
                if coordinate_transformer and coordinate_transformer.is_calibrated:
                    physical_location = coordinate_transformer.pixel_to_physical((center_x, center_y))
                
                target = ProbeTarget(
                    center=center,
                    corners=corners,
                    area=area,
                    confidence=confidence,
                    physical_location=physical_location
                )
                
                targets.append(target)
                
            except Exception as e:
                logger.debug(f"Error analyzing contour: {e}")
                continue
        
        return targets
    
    def _calculate_confidence(self, area: float, aspect_ratio: float, 
                            circularity: float, corners: np.ndarray) -> float:
        """Calculate confidence score for probe target.
        
        Args:
            area: Contour area.
            aspect_ratio: Width to height ratio.
            circularity: Circularity measure.
            corners: Corner points.
        
        Returns:
            Confidence score (0.0 to 1.0).
        """
        confidence = 0.0
        
        # Area score (prefer medium-sized targets)
        optimal_area = (self.min_area + self.max_area) / 2
        area_score = 1.0 - abs(area - optimal_area) / optimal_area
        confidence += 0.3 * max(0, area_score)
        
        # Aspect ratio score (prefer square targets)
        aspect_score = 1.0 - abs(aspect_ratio - 1.0)
        confidence += 0.3 * max(0, aspect_score)
        
        # Circularity score
        confidence += 0.3 * circularity
        
        # Corner count score (prefer 4 corners for rectangular probe plates)
        corner_count = len(corners)
        if corner_count == 4:
            confidence += 0.1
        elif corner_count >= 6:  # Could be circular
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _select_best_target(self, targets: List[ProbeTarget]) -> Optional[ProbeTarget]:
        """Select the best probe target from candidates.
        
        Args:
            targets: List of probe targets.
        
        Returns:
            Best target or None if no targets.
        """
        if not targets:
            return None
        
        # Sort by confidence and return the best
        targets.sort(key=lambda t: t.confidence, reverse=True)
        return targets[0]
    
    def _create_annotated_image(self, image: np.ndarray, 
                              targets: List[ProbeTarget],
                              best_target: Optional[ProbeTarget]) -> np.ndarray:
        """Create annotated image with probe targets.
        
        Args:
            image: Original image.
            targets: List of probe targets.
            best_target: Best probe target.
        
        Returns:
            Annotated image.
        """
        annotated = image.copy()
        
        # Draw all targets
        for i, target in enumerate(targets):
            color = (0, 255, 0) if target == best_target else (0, 255, 255)
            
            # Draw contour
            if len(target.corners) > 0:
                cv2.drawContours(annotated, [target.corners], -1, color, 2)
            
            # Draw center
            center = (int(target.center.x), int(target.center.y))
            cv2.circle(annotated, center, 5, color, -1)
            
            # Draw confidence
            text = f"T{i}: {target.confidence:.2f}"
            cv2.putText(annotated, text, 
                       (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw physical location if available
            if target.physical_location:
                phys_text = f"({target.physical_location.x:.2f}, {target.physical_location.y:.2f})"
                cv2.putText(annotated, phys_text,
                           (center[0] + 10, center[1] + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Highlight best target
        if best_target:
            center = (int(best_target.center.x), int(best_target.center.y))
            cv2.circle(annotated, center, 20, (0, 255, 0), 3)
            cv2.putText(annotated, "BEST TARGET", 
                       (center[0] - 50, center[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated
    
    def auto_find_probe_plate(self, image: np.ndarray, 
                             coordinate_transformer=None) -> Optional[Point3D]:
        """Automatically find the best probe plate location.
        
        Args:
            image: Input image.
            coordinate_transformer: Optional coordinate transformer.
        
        Returns:
            Physical location of best probe target or None.
        """
        result = self.detect_probe_targets(image, coordinate_transformer, annotate=False)
        
        if result.best_target and result.best_target.physical_location:
            logger.info(f"Auto-found probe plate at {result.best_target.physical_location}")
            return result.best_target.physical_location
        
        logger.warning("No suitable probe target found")
        return None
    
    def validate_probe_target(self, target: ProbeTarget, 
                            min_confidence: float = 0.5) -> bool:
        """Validate if a probe target is suitable for probing.
        
        Args:
            target: Probe target to validate.
            min_confidence: Minimum confidence threshold.
        
        Returns:
            True if target is valid for probing.
        """
        if target.confidence < min_confidence:
            logger.warning(f"Probe target confidence too low: {target.confidence}")
            return False
        
        if target.area < self.min_area or target.area > self.max_area:
            logger.warning(f"Probe target area out of range: {target.area}")
            return False
        
        return True
    
    def get_detection_info(self) -> Dict[str, Any]:
        """Get detector configuration information.
        
        Returns:
            Dictionary with detector settings.
        """
        return {
            'min_area': self.min_area,
            'max_area': self.max_area,
            'aspect_ratio_tolerance': self.aspect_ratio_tolerance,
            'circularity_threshold': self.circularity_threshold,
            'use_color_detection': self.use_color_detection,
            'probe_dimensions': {
                'width': self.probe_width,
                'height': self.probe_height
            }
        }