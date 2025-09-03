"""Static image camera implementation."""

import cv2
import numpy as np
from typing import Optional, Any, List, Generator
from pathlib import Path
import logging
import glob
from .base import CameraInterface

logger = logging.getLogger(__name__)


class StaticImageCamera(CameraInterface):
    """Static image camera implementation for loading images from files."""
    
    def __init__(self, config: dict):
        """Initialize static image camera.
        
        Args:
            config: Camera configuration dictionary.
        """
        super().__init__(config)
        self.file_path = config.get('file_path')
        self.image_list = []
        self.current_index = 0
        self.current_image = None
        
        # Auto-discovery of images in directory
        self.auto_discover = config.get('auto_discover', True)
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        if self.file_path:
            self.file_path = Path(self.file_path)
    
    def open(self) -> bool:
        """Open image source.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            if not self.file_path:
                logger.error("No file path specified")
                return False
            
            if self.file_path.is_file():
                # Single image file
                self.image_list = [self.file_path]
                logger.info(f"Loaded single image: {self.file_path}")
                
            elif self.file_path.is_dir() and self.auto_discover:
                # Directory - discover images
                self.image_list = self._discover_images(self.file_path)
                logger.info(f"Discovered {len(self.image_list)} images in {self.file_path}")
                
            else:
                logger.error(f"Invalid file path: {self.file_path}")
                return False
            
            if not self.image_list:
                logger.error("No images found")
                return False
            
            # Load first image to verify and get dimensions
            first_image = self._load_image(self.image_list[0])
            if first_image is None:
                logger.error("Failed to load first image")
                return False
            
            self.current_image = first_image
            self.resolution = (first_image.shape[1], first_image.shape[0])
            self.is_open = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening static image source: {e}")
            return False
    
    def close(self) -> None:
        """Close image source."""
        self.image_list = []
        self.current_image = None
        self.current_index = 0
        self.is_open = False
        logger.info("Static image camera closed")
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture current image.
        
        Returns:
            Current image as numpy array or None if failed.
        """
        if not self.is_available() or not self.image_list:
            return None
        
        try:
            # Return copy of current image
            if self.current_image is not None:
                return self.current_image.copy()
            
            # Load current image if not cached
            current_path = self.image_list[self.current_index]
            image = self._load_image(current_path)
            if image is not None:
                self.current_image = image
                return image.copy()
            
            return None
            
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            return None
    
    def get_live_stream(self) -> Generator[np.ndarray, None, None]:
        """Get stream of all images (cycling through list).
        
        Yields:
            Images as numpy arrays.
        """
        if not self.is_available():
            return
        
        try:
            while True:
                for i in range(len(self.image_list)):
                    self.set_image_index(i)
                    image = self.capture()
                    if image is not None:
                        yield image
                    else:
                        break
                        
        except Exception as e:
            logger.error(f"Error in image stream: {e}")
    
    def next_image(self) -> Optional[np.ndarray]:
        """Load next image in sequence.
        
        Returns:
            Next image or None if at end.
        """
        if not self.is_available() or not self.image_list:
            return None
        
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.current_image = None  # Force reload
            return self.capture()
        
        return None
    
    def previous_image(self) -> Optional[np.ndarray]:
        """Load previous image in sequence.
        
        Returns:
            Previous image or None if at beginning.
        """
        if not self.is_available() or not self.image_list:
            return None
        
        if self.current_index > 0:
            self.current_index -= 1
            self.current_image = None  # Force reload
            return self.capture()
        
        return None
    
    def set_image_index(self, index: int) -> bool:
        """Set current image by index.
        
        Args:
            index: Image index to set.
        
        Returns:
            True if successful.
        """
        if not self.is_available() or not (0 <= index < len(self.image_list)):
            return False
        
        self.current_index = index
        self.current_image = None  # Force reload
        return True
    
    def set_image_path(self, file_path: Path) -> bool:
        """Set image by file path.
        
        Args:
            file_path: Path to image file.
        
        Returns:
            True if successful.
        """
        if not self.is_available():
            return False
        
        try:
            file_path = Path(file_path)
            if file_path in self.image_list:
                self.current_index = self.image_list.index(file_path)
                self.current_image = None  # Force reload
                return True
            else:
                # Try to load new image
                image = self._load_image(file_path)
                if image is not None:
                    self.image_list = [file_path]
                    self.current_index = 0
                    self.current_image = image
                    return True
                    
        except Exception as e:
            logger.error(f"Error setting image path: {e}")
        
        return False
    
    def _load_image(self, file_path: Path) -> Optional[np.ndarray]:
        """Load image from file.
        
        Args:
            file_path: Path to image file.
        
        Returns:
            Image as numpy array or None if failed.
        """
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                logger.error(f"Failed to load image: {file_path}")
                return None
            
            logger.debug(f"Loaded image: {file_path} ({image.shape[1]}x{image.shape[0]})")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None
    
    def _discover_images(self, directory: Path) -> List[Path]:
        """Discover image files in directory.
        
        Args:
            directory: Directory to search.
        
        Returns:
            List of image file paths.
        """
        images = []
        
        try:
            for ext in self.supported_formats:
                # Search for files with each extension (case insensitive)
                pattern = str(directory / f"*{ext}")
                files = glob.glob(pattern, recursive=False)
                
                # Also try uppercase
                pattern_upper = str(directory / f"*{ext.upper()}")
                files.extend(glob.glob(pattern_upper, recursive=False))
                
                images.extend([Path(f) for f in files])
            
            # Remove duplicates and sort
            images = list(set(images))
            images.sort()
            
            logger.debug(f"Discovered {len(images)} images with extensions: {self.supported_formats}")
            
        except Exception as e:
            logger.error(f"Error discovering images: {e}")
        
        return images
    
    def get_image_list(self) -> List[Path]:
        """Get list of available images.
        
        Returns:
            List of image file paths.
        """
        return self.image_list.copy()
    
    def get_current_index(self) -> int:
        """Get current image index.
        
        Returns:
            Current index.
        """
        return self.current_index
    
    def get_current_path(self) -> Optional[Path]:
        """Get current image file path.
        
        Returns:
            Current image path or None.
        """
        if 0 <= self.current_index < len(self.image_list):
            return self.image_list[self.current_index]
        return None
    
    def get_image_count(self) -> int:
        """Get total number of images.
        
        Returns:
            Number of images.
        """
        return len(self.image_list)
    
    def reload_directory(self) -> bool:
        """Reload images from directory.
        
        Returns:
            True if successful.
        """
        if not self.file_path or not self.file_path.is_dir():
            return False
        
        try:
            old_count = len(self.image_list)
            self.image_list = self._discover_images(self.file_path)
            new_count = len(self.image_list)
            
            # Reset index if necessary
            if self.current_index >= new_count:
                self.current_index = max(0, new_count - 1)
                self.current_image = None
            
            logger.info(f"Reloaded directory: {old_count} -> {new_count} images")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading directory: {e}")
            return False
    
    def get_camera_info(self) -> dict:
        """Get camera information.
        
        Returns:
            Dictionary with camera properties.
        """
        info = {
            'source': 'static',
            'file_path': str(self.file_path) if self.file_path else None,
            'current_index': self.current_index,
            'total_images': len(self.image_list),
            'resolution': self.resolution,
            'supported_formats': self.supported_formats
        }
        
        current_path = self.get_current_path()
        if current_path:
            info['current_image'] = str(current_path)
            
        return info