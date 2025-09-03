"""Abstract base class for camera interfaces."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CameraInterface(ABC):
    """Abstract camera interface for different camera sources."""
    
    def __init__(self, config: dict):
        """Initialize camera interface.
        
        Args:
            config: Camera configuration dictionary.
        """
        self.config = config
        self.is_open = False
        self.resolution = tuple(config.get('resolution', [1920, 1080]))
        self.fps = config.get('fps', 30)
        
    @abstractmethod
    def open(self) -> bool:
        """Open camera connection.
        
        Returns:
            True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close camera connection."""
        pass
    
    @abstractmethod
    def capture(self) -> Optional[np.ndarray]:
        """Capture a single frame.
        
        Returns:
            Image as numpy array or None if capture failed.
        """
        pass
    
    @abstractmethod
    def get_live_stream(self) -> Any:
        """Get live stream handle.
        
        Returns:
            Stream handle for continuous capture.
        """
        pass
    
    def is_available(self) -> bool:
        """Check if camera is available.
        
        Returns:
            True if camera is available and ready.
        """
        return self.is_open
    
    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution.
        
        Args:
            width: Image width in pixels.
            height: Image height in pixels.
        
        Returns:
            True if successful, False otherwise.
        """
        self.resolution = (width, height)
        return True
    
    def set_fps(self, fps: int) -> bool:
        """Set camera frame rate.
        
        Args:
            fps: Frames per second.
        
        Returns:
            True if successful, False otherwise.
        """
        self.fps = fps
        return True
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get current resolution.
        
        Returns:
            Tuple of (width, height).
        """
        return self.resolution
    
    def get_fps(self) -> int:
        """Get current frame rate.
        
        Returns:
            Frames per second.
        """
        return self.fps
    
    def save_image(self, image: np.ndarray, file_path: Path) -> bool:
        """Save image to file.
        
        Args:
            image: Image array to save.
            file_path: Path to save image to.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            import cv2
            cv2.imwrite(str(file_path), image)
            logger.info(f"Image saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()