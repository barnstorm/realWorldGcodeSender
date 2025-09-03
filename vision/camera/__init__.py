"""Camera module for different camera sources."""

from .base import CameraInterface
from .webcam import WebcamCamera  
from .static import StaticImageCamera

__all__ = ['CameraInterface', 'WebcamCamera', 'StaticImageCamera']