"""Device detection utilities for finding COM ports and cameras."""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import platform

logger = logging.getLogger(__name__)


@dataclass
class SerialPortInfo:
    """Information about a serial port."""
    port: str
    description: str
    manufacturer: Optional[str] = None
    vendor_id: Optional[str] = None
    product_id: Optional[str] = None
    serial_number: Optional[str] = None
    
    def __str__(self):
        return f"{self.port} - {self.description}"


@dataclass
class CameraInfo:
    """Information about a camera device."""
    index: int
    name: str
    resolution: Optional[Tuple[int, int]] = None
    fps: Optional[float] = None
    available: bool = True
    
    def __str__(self):
        if self.resolution:
            return f"Camera {self.index}: {self.name} ({self.resolution[0]}x{self.resolution[1]})"
        return f"Camera {self.index}: {self.name}"


class DeviceDetector:
    """Detects and lists available devices (COM ports, cameras)."""
    
    @staticmethod
    def get_serial_ports() -> List[SerialPortInfo]:
        """Get list of available serial ports.
        
        Returns:
            List of SerialPortInfo objects.
        """
        ports = []
        
        try:
            import serial.tools.list_ports
            
            for port_info in serial.tools.list_ports.comports():
                port = SerialPortInfo(
                    port=port_info.device,
                    description=port_info.description or "Unknown Device",
                    manufacturer=getattr(port_info, 'manufacturer', None),
                    vendor_id=getattr(port_info, 'vid', None),
                    product_id=getattr(port_info, 'pid', None),
                    serial_number=getattr(port_info, 'serial_number', None)
                )
                ports.append(port)
                
            logger.info(f"Found {len(ports)} serial ports")
            
        except ImportError:
            logger.warning("pyserial not available - cannot detect serial ports")
        except Exception as e:
            logger.error(f"Error detecting serial ports: {e}")
        
        return ports
    
    @staticmethod
    def get_cameras() -> List[CameraInfo]:
        """Get list of available cameras.
        
        Returns:
            List of CameraInfo objects.
        """
        cameras = []
        
        try:
            import cv2
            
            # Test camera indices 0-9
            for i in range(10):
                cap = cv2.VideoCapture(i)
                
                if cap.isOpened():
                    # Try to read a frame to verify camera works
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        # Try to get camera name (may not be available on all platforms)
                        name = f"Camera {i}"
                        if hasattr(cv2, 'CAP_PROP_BACKEND_NAME'):
                            try:
                                backend = cap.get(cv2.CAP_PROP_BACKEND_NAME)
                                if backend:
                                    name = f"Camera {i} ({backend})"
                            except:
                                pass
                        
                        camera = CameraInfo(
                            index=i,
                            name=name,
                            resolution=(width, height) if width > 0 and height > 0 else None,
                            fps=fps if fps > 0 else None,
                            available=True
                        )
                        cameras.append(camera)
                
                cap.release()
            
            logger.info(f"Found {len(cameras)} cameras")
            
        except ImportError:
            logger.warning("OpenCV not available - cannot detect cameras")
        except Exception as e:
            logger.error(f"Error detecting cameras: {e}")
        
        return cameras
    
    @staticmethod
    def test_serial_port(port: str, baudrate: int = 115200, timeout: float = 1.0) -> bool:
        """Test if a serial port can be opened.
        
        Args:
            port: Port name to test.
            baudrate: Baud rate to use.
            timeout: Connection timeout.
        
        Returns:
            True if port can be opened.
        """
        try:
            import serial
            
            with serial.Serial(port, baudrate, timeout=timeout) as ser:
                # Try to read any available data
                ser.read(1)
                return True
                
        except ImportError:
            logger.warning("pyserial not available - cannot test serial port")
            return False
        except Exception as e:
            logger.debug(f"Cannot open port {port}: {e}")
            return False
    
    @staticmethod
    def test_camera(index: int) -> bool:
        """Test if a camera can be opened and used.
        
        Args:
            index: Camera index to test.
        
        Returns:
            True if camera works.
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                return False
            
            # Try to capture a frame
            ret, frame = cap.read()
            cap.release()
            
            return ret and frame is not None
            
        except ImportError:
            logger.warning("OpenCV not available - cannot test camera")
            return False
        except Exception as e:
            logger.debug(f"Cannot test camera {index}: {e}")
            return False
    
    @staticmethod
    def get_recommended_serial_port() -> Optional[str]:
        """Get recommended serial port for CNC connection.
        
        Returns:
            Recommended port name or None.
        """
        ports = DeviceDetector.get_serial_ports()
        
        if not ports:
            return None
        
        # Prioritize by common CNC controller patterns
        cnc_keywords = ['arduino', 'ch340', 'cp210', 'ftdi', 'prolific', 'usb']
        
        for port in ports:
            desc_lower = port.description.lower()
            manufacturer_lower = (port.manufacturer or '').lower()
            
            for keyword in cnc_keywords:
                if keyword in desc_lower or keyword in manufacturer_lower:
                    logger.info(f"Recommended port: {port.port} ({port.description})")
                    return port.port
        
        # Default to first available port
        logger.info(f"No CNC-specific port found, using first available: {ports[0].port}")
        return ports[0].port
    
    @staticmethod
    def get_recommended_camera() -> Optional[int]:
        """Get recommended camera index.
        
        Returns:
            Recommended camera index or None.
        """
        cameras = DeviceDetector.get_cameras()
        
        if not cameras:
            return None
        
        # Prefer higher resolution cameras
        cameras.sort(key=lambda c: (c.resolution[0] * c.resolution[1] if c.resolution else 0), reverse=True)
        
        recommended = cameras[0]
        logger.info(f"Recommended camera: {recommended}")
        return recommended.index
    
    @staticmethod
    def get_system_info() -> Dict[str, str]:
        """Get system information.
        
        Returns:
            Dictionary with system info.
        """
        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'machine': platform.machine(),
            'processor': platform.processor(),
        }
        
        try:
            import cv2
            info['opencv_version'] = cv2.__version__
            info['opencv_has_aruco'] = str(hasattr(cv2, 'aruco'))
        except ImportError:
            info['opencv_version'] = 'Not installed'
            info['opencv_has_aruco'] = 'False'
        
        try:
            import serial
            info['pyserial_version'] = serial.__version__
        except ImportError:
            info['pyserial_version'] = 'Not installed'
        
        try:
            import numpy as np
            info['numpy_version'] = np.__version__
        except ImportError:
            info['numpy_version'] = 'Not installed'
        
        return info


# Convenience functions
def list_serial_ports() -> List[str]:
    """Get simple list of serial port names.
    
    Returns:
        List of port names.
    """
    return [port.port for port in DeviceDetector.get_serial_ports()]


def list_cameras() -> List[int]:
    """Get simple list of camera indices.
    
    Returns:
        List of camera indices.
    """
    return [cam.index for cam in DeviceDetector.get_cameras()]


def auto_detect_devices() -> Dict[str, Optional[str]]:
    """Auto-detect recommended devices.
    
    Returns:
        Dictionary with recommended device settings.
    """
    return {
        'serial_port': DeviceDetector.get_recommended_serial_port(),
        'camera_index': DeviceDetector.get_recommended_camera(),
    }