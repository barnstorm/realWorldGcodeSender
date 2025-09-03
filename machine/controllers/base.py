"""Abstract base class for machine controllers."""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MachineState(Enum):
    """Machine state enumeration."""
    DISCONNECTED = "disconnected"
    IDLE = "idle"
    RUN = "run"
    HOLD = "hold"
    JOG = "jog"
    ALARM = "alarm"
    DOOR = "door"
    CHECK = "check"
    HOME = "home"
    SLEEP = "sleep"


@dataclass
class MachinePosition:
    """Machine position data."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Work coordinate position
    work_x: float = 0.0
    work_y: float = 0.0
    work_z: float = 0.0
    
    # Machine coordinate position
    machine_x: float = 0.0
    machine_y: float = 0.0
    machine_z: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'work_x': self.work_x, 'work_y': self.work_y, 'work_z': self.work_z,
            'machine_x': self.machine_x, 'machine_y': self.machine_y, 'machine_z': self.machine_z
        }


@dataclass
class MachineStatus:
    """Machine status information."""
    state: MachineState = MachineState.DISCONNECTED
    position: MachinePosition = None
    feedrate: float = 0.0
    spindle_speed: float = 0.0
    current_line: int = 0
    total_lines: int = 0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.position is None:
            self.position = MachinePosition()


class MachineInterface(ABC):
    """Abstract machine controller interface."""
    
    def __init__(self, config: dict):
        """Initialize machine interface.
        
        Args:
            config: Machine configuration dictionary.
        """
        self.config = config
        self.is_connected = False
        self.status = MachineStatus()
        self.status_callbacks: List[Callable] = []
        
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the machine.
        
        Returns:
            True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the machine."""
        pass
    
    @abstractmethod
    def send_command(self, command: str) -> Optional[str]:
        """Send a command to the machine.
        
        Args:
            command: Command string to send.
        
        Returns:
            Response from machine or None.
        """
        pass
    
    @abstractmethod
    def execute_gcode(self, gcode: List[str]) -> bool:
        """Execute G-code program.
        
        Args:
            gcode: List of G-code commands.
        
        Returns:
            True if execution started successfully.
        """
        pass
    
    @abstractmethod
    def pause(self) -> bool:
        """Pause current operation.
        
        Returns:
            True if successful.
        """
        pass
    
    @abstractmethod
    def resume(self) -> bool:
        """Resume paused operation.
        
        Returns:
            True if successful.
        """
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Stop current operation.
        
        Returns:
            True if successful.
        """
        pass
    
    @abstractmethod
    def home(self, axes: Optional[List[str]] = None) -> bool:
        """Home the machine.
        
        Args:
            axes: List of axes to home ('X', 'Y', 'Z'). None homes all.
        
        Returns:
            True if homing started successfully.
        """
        pass
    
    @abstractmethod
    def probe(self, axis: str = 'Z', distance: float = -1.0, 
             feedrate: float = 10.0) -> Optional[float]:
        """Probe for surface.
        
        Args:
            axis: Axis to probe along.
            distance: Maximum probe distance.
            feedrate: Probe feedrate.
        
        Returns:
            Probe position or None if failed.
        """
        pass
    
    @abstractmethod
    def jog(self, axis: str, distance: float, feedrate: float) -> bool:
        """Jog machine.
        
        Args:
            axis: Axis to jog ('X', 'Y', 'Z').
            distance: Distance to jog (positive or negative).
            feedrate: Jog feedrate.
        
        Returns:
            True if jog command sent successfully.
        """
        pass
    
    @abstractmethod
    def get_position(self) -> MachinePosition:
        """Get current machine position.
        
        Returns:
            Current position.
        """
        pass
    
    @abstractmethod
    def get_status(self) -> MachineStatus:
        """Get machine status.
        
        Returns:
            Current machine status.
        """
        pass
    
    @abstractmethod
    def set_work_zero(self, axes: Optional[List[str]] = None) -> bool:
        """Set work coordinate zero.
        
        Args:
            axes: Axes to zero. None zeros all.
        
        Returns:
            True if successful.
        """
        pass
    
    @abstractmethod
    def set_feedrate(self, feedrate: float) -> bool:
        """Set feedrate override.
        
        Args:
            feedrate: Feedrate in units per minute.
        
        Returns:
            True if successful.
        """
        pass
    
    @abstractmethod
    def set_spindle_speed(self, speed: float) -> bool:
        """Set spindle speed.
        
        Args:
            speed: Spindle speed in RPM.
        
        Returns:
            True if successful.
        """
        pass
    
    def is_idle(self) -> bool:
        """Check if machine is idle.
        
        Returns:
            True if machine is idle and ready for commands.
        """
        return self.status.state == MachineState.IDLE
    
    def is_running(self) -> bool:
        """Check if machine is running a program.
        
        Returns:
            True if machine is running.
        """
        return self.status.state == MachineState.RUN
    
    def is_alarm(self) -> bool:
        """Check if machine is in alarm state.
        
        Returns:
            True if machine is in alarm.
        """
        return self.status.state == MachineState.ALARM
    
    def register_status_callback(self, callback: Callable) -> None:
        """Register a status update callback.
        
        Args:
            callback: Function to call on status updates.
        """
        self.status_callbacks.append(callback)
        logger.debug("Registered status callback")
    
    def unregister_status_callback(self, callback: Callable) -> None:
        """Unregister a status update callback.
        
        Args:
            callback: Function to remove.
        """
        try:
            self.status_callbacks.remove(callback)
            logger.debug("Unregistered status callback")
        except ValueError:
            logger.warning("Callback not found in status callbacks")
    
    def notify_status_change(self) -> None:
        """Notify all registered callbacks of status change."""
        for callback in self.status_callbacks:
            try:
                callback(self.status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()