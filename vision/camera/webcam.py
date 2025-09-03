"""Webcam camera implementation."""

import cv2
import numpy as np
from typing import Optional, Any, Generator, List
import logging
import time
import threading
from queue import Queue, Empty
from .base import CameraInterface

logger = logging.getLogger(__name__)


class WebcamCamera(CameraInterface):
    """Webcam camera implementation using OpenCV."""
    
    def __init__(self, config: dict):
        """Initialize webcam camera.
        
        Args:
            config: Camera configuration dictionary.
        """
        super().__init__(config)
        self.device_index = config.get('device_index', 0)
        self.cap = None
        self.last_frame = None
        self.frame_count = 0
        
        # Streaming support
        self.streaming = False
        self.stream_thread = None
        self.frame_queue = Queue(maxsize=2)  # Small buffer
        
        # Camera settings
        self.brightness = config.get('brightness', 0.5)
        self.contrast = config.get('contrast', 0.5)
        self.exposure = config.get('exposure', 'auto')
    
    def open(self) -> bool:
        """Open webcam connection.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(self.device_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.device_index}")
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Apply camera settings
            self._apply_settings()
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Webcam opened: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            
            self.is_open = True
            return True
            
        except Exception as e:
            logger.error(f"Error opening webcam: {e}")
            return False
    
    def close(self) -> None:
        """Close webcam connection."""
        self.stop_stream()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_open = False
        logger.info("Webcam closed")
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture a single frame.
        
        Returns:
            Image as numpy array or None if capture failed.
        """
        if not self.is_available():
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy()
                self.frame_count += 1
                return frame
            else:
                logger.warning("Failed to capture frame")
                return None
                
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def get_live_stream(self) -> Generator[np.ndarray, None, None]:
        """Get live stream generator.
        
        Yields:
            Image frames as numpy arrays.
        """
        if not self.is_available():
            return
        
        try:
            while self.is_open:
                frame = self.capture()
                if frame is not None:
                    yield frame
                else:
                    break
                    
        except Exception as e:
            logger.error(f"Error in live stream: {e}")
    
    def start_stream(self) -> bool:
        """Start streaming in background thread.
        
        Returns:
            True if streaming started successfully.
        """
        if self.streaming or not self.is_available():
            return False
        
        try:
            self.streaming = True
            self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
            self.stream_thread.start()
            logger.info("Background streaming started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            self.streaming = False
            return False
    
    def stop_stream(self) -> None:
        """Stop background streaming."""
        if self.streaming:
            self.streaming = False
            if self.stream_thread:
                self.stream_thread.join(timeout=1.0)
            
            # Clear frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    break
            
            logger.info("Background streaming stopped")
    
    def get_latest_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get latest frame from background stream.
        
        Args:
            timeout: Maximum time to wait for frame.
        
        Returns:
            Latest frame or None.
        """
        if not self.streaming:
            return self.capture()
        
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _stream_worker(self) -> None:
        """Background thread worker for streaming."""
        frame_interval = 1.0 / self.fps
        last_capture_time = 0
        
        while self.streaming and self.is_open:
            current_time = time.time()
            
            # Rate limiting
            if current_time - last_capture_time < frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            frame = self.capture()
            if frame is not None:
                # Add frame to queue (non-blocking, drop oldest if full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                
                try:
                    self.frame_queue.put_nowait(frame.copy())
                    last_capture_time = current_time
                except:
                    pass  # Queue full, drop frame
    
    def _apply_settings(self) -> None:
        """Apply camera settings."""
        if not self.cap:
            return
        
        try:
            # Set brightness
            if hasattr(cv2, 'CAP_PROP_BRIGHTNESS'):
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
            
            # Set contrast
            if hasattr(cv2, 'CAP_PROP_CONTRAST'):
                self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast)
            
            # Set exposure
            if hasattr(cv2, 'CAP_PROP_EXPOSURE'):
                if self.exposure == 'auto':
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                else:
                    try:
                        exposure_val = float(self.exposure)
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)
                    except ValueError:
                        logger.warning(f"Invalid exposure value: {self.exposure}")
            
            logger.debug("Camera settings applied")
            
        except Exception as e:
            logger.warning(f"Error applying camera settings: {e}")
    
    def set_brightness(self, brightness: float) -> bool:
        """Set camera brightness.
        
        Args:
            brightness: Brightness value (0.0 to 1.0).
        
        Returns:
            True if successful.
        """
        if not self.cap or not (0.0 <= brightness <= 1.0):
            return False
        
        try:
            self.brightness = brightness
            if hasattr(cv2, 'CAP_PROP_BRIGHTNESS'):
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            return True
        except Exception as e:
            logger.error(f"Error setting brightness: {e}")
            return False
    
    def set_contrast(self, contrast: float) -> bool:
        """Set camera contrast.
        
        Args:
            contrast: Contrast value (0.0 to 1.0).
        
        Returns:
            True if successful.
        """
        if not self.cap or not (0.0 <= contrast <= 1.0):
            return False
        
        try:
            self.contrast = contrast
            if hasattr(cv2, 'CAP_PROP_CONTRAST'):
                self.cap.set(cv2.CAP_PROP_CONTRAST, contrast)
            return True
        except Exception as e:
            logger.error(f"Error setting contrast: {e}")
            return False
    
    def set_exposure(self, exposure) -> bool:
        """Set camera exposure.
        
        Args:
            exposure: Exposure value ('auto' or numeric value).
        
        Returns:
            True if successful.
        """
        if not self.cap:
            return False
        
        try:
            self.exposure = exposure
            self._apply_settings()
            return True
        except Exception as e:
            logger.error(f"Error setting exposure: {e}")
            return False
    
    def get_camera_info(self) -> dict:
        """Get camera information.
        
        Returns:
            Dictionary with camera properties.
        """
        if not self.cap:
            return {}
        
        try:
            info = {
                'device_index': self.device_index,
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'brightness': self.brightness,
                'contrast': self.contrast,
                'exposure': self.exposure,
                'frame_count': self.frame_count,
                'is_streaming': self.streaming
            }
            
            # Add additional properties if available
            if hasattr(cv2, 'CAP_PROP_FOURCC'):
                fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                info['fourcc'] = chr(fourcc & 255) + chr((fourcc >> 8) & 255) + \
                               chr((fourcc >> 16) & 255) + chr((fourcc >> 24) & 255)
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting camera info: {e}")
            return {}
    
    @staticmethod
    def list_available_cameras() -> List[int]:
        """List available camera indices.
        
        Returns:
            List of available camera device indices.
        """
        available = []
        
        # Test cameras 0-9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
        
        logger.info(f"Found {len(available)} available cameras: {available}")
        return available