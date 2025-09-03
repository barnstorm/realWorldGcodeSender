"""Main application coordinator for the CNC G-code sender."""

import logging
from pathlib import Path
from typing import Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.loader import ConfigLoader
from config.settings import ApplicationConfig
from utils.logging_config import setup_logging
from core.event_bus import get_event_bus, Events, Event

logger = logging.getLogger(__name__)


class CNCApplication:
    """Main application coordinator."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize the application.
        
        Args:
            config_file: Optional configuration file to load.
        """
        self.config: Optional[ApplicationConfig] = None
        self.config_loader = ConfigLoader()
        self.event_bus = get_event_bus()
        
        # Component placeholders
        self.camera = None
        self.machine = None
        self.ui = None
        
        # Initialize configuration
        self._load_config(config_file)
        
        # Set up logging
        self._setup_logging()
        
        # Subscribe to core events
        self._setup_event_handlers()
        
        logger.info("CNC Application initialized")
    
    def _load_config(self, config_file: Optional[Path] = None) -> None:
        """Load application configuration.
        
        Args:
            config_file: Optional configuration file.
        """
        try:
            self.config = self.config_loader.load(config_file)
            
            # Validate configuration
            if not self.config_loader.validate(self.config):
                logger.warning("Configuration validation failed, using defaults")
                self.config = ApplicationConfig()
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
            self.config = ApplicationConfig()
    
    def _setup_logging(self) -> None:
        """Set up application logging."""
        setup_logging(
            log_dir=self.config.log_dir,
            log_level="DEBUG" if self.config.debug_mode else "INFO",
            verbose=self.config.verbose_logging
        )
    
    def _setup_event_handlers(self) -> None:
        """Set up core event handlers."""
        # Machine events
        self.event_bus.subscribe(Events.MACHINE_ERROR, self._handle_machine_error)
        self.event_bus.subscribe(Events.MACHINE_ALARM, self._handle_machine_alarm)
        
        # Camera events
        self.event_bus.subscribe(Events.CAMERA_ERROR, self._handle_camera_error)
        
        # Execution events
        self.event_bus.subscribe(Events.EXECUTION_COMPLETED, self._handle_execution_completed)
    
    def initialize_camera(self) -> bool:
        """Initialize camera module.
        
        Returns:
            True if successful.
        """
        try:
            # Import camera module based on configuration
            if self.config.calibration.camera.source == "webcam":
                from vision.camera.webcam import WebcamCamera
                self.camera = WebcamCamera(self.config.calibration.camera.to_dict())
            elif self.config.calibration.camera.source == "file":
                from vision.camera.static import StaticImageCamera
                self.camera = StaticImageCamera(self.config.calibration.camera.to_dict())
            else:
                logger.error(f"Unknown camera source: {self.config.calibration.camera.source}")
                return False
            
            if self.camera.open():
                self.event_bus.emit(Events.CAMERA_CONNECTED)
                logger.info("Camera initialized successfully")
                return True
            else:
                logger.error("Failed to open camera")
                return False
                
        except ImportError as e:
            logger.error(f"Camera module not implemented yet: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def initialize_machine(self) -> bool:
        """Initialize machine controller.
        
        Returns:
            True if successful.
        """
        try:
            # Import machine module based on configuration
            if self.config.machine.type == "grbl":
                from machine.controllers.grbl import GRBLController
                self.machine = GRBLController(self.config.machine.to_dict())
            elif self.config.simulation_mode:
                from machine.controllers.simulation import SimulationController
                self.machine = SimulationController(self.config.machine.to_dict())
            else:
                logger.error(f"Unknown machine type: {self.config.machine.type}")
                return False
            
            if self.machine.connect():
                self.event_bus.emit(Events.MACHINE_CONNECTED)
                logger.info("Machine controller initialized successfully")
                return True
            else:
                logger.error("Failed to connect to machine")
                return False
                
        except ImportError as e:
            logger.error(f"Machine module not implemented yet: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize machine: {e}")
            return False
    
    def initialize_ui(self) -> bool:
        """Initialize user interface.
        
        Returns:
            True if successful.
        """
        try:
            # Import UI module based on configuration
            if self.config.ui_backend == "matplotlib":
                from ui.interfaces.matplotlib_ui import MatplotlibUI
                self.ui = MatplotlibUI({"theme": self.config.theme})
            elif self.config.ui_backend == "qt":
                from ui.interfaces.qt_ui import QtUI
                self.ui = QtUI({"theme": self.config.theme})
            elif self.config.ui_backend == "web":
                from ui.interfaces.web_ui import WebUI
                self.ui = WebUI({"theme": self.config.theme})
            else:
                logger.error(f"Unknown UI backend: {self.config.ui_backend}")
                return False
            
            if self.ui.initialize():
                self.event_bus.emit(Events.UI_INITIALIZED)
                logger.info("UI initialized successfully")
                return True
            else:
                logger.error("Failed to initialize UI")
                return False
                
        except ImportError as e:
            logger.error(f"UI module not implemented yet: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize UI: {e}")
            return False
    
    def run(self) -> None:
        """Run the main application."""
        logger.info("Starting CNC application")
        
        # Initialize components
        if not self.config.simulation_mode:
            if not self.initialize_camera():
                logger.warning("Running without camera")
            
            if not self.initialize_machine():
                logger.warning("Running without machine connection")
        
        if not self.initialize_ui():
            logger.error("Cannot run without UI")
            return
        
        try:
            # Run UI event loop
            self.ui.run()
            
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self) -> None:
        """Shut down the application cleanly."""
        logger.info("Shutting down application")
        
        # Stop UI
        if self.ui:
            try:
                self.ui.stop()
                self.event_bus.emit(Events.UI_CLOSED)
            except Exception as e:
                logger.error(f"Error stopping UI: {e}")
        
        # Disconnect machine
        if self.machine:
            try:
                self.machine.disconnect()
                self.event_bus.emit(Events.MACHINE_DISCONNECTED)
            except Exception as e:
                logger.error(f"Error disconnecting machine: {e}")
        
        # Close camera
        if self.camera:
            try:
                self.camera.close()
                self.event_bus.emit(Events.CAMERA_DISCONNECTED)
            except Exception as e:
                logger.error(f"Error closing camera: {e}")
        
        # Stop event bus
        self.event_bus.stop()
        
        logger.info("Application shutdown complete")
    
    def _handle_machine_error(self, event: Event) -> None:
        """Handle machine error events.
        
        Args:
            event: Error event.
        """
        logger.error(f"Machine error: {event.data}")
        if self.ui:
            self.ui.show_message(f"Machine Error: {event.data}", "error")
    
    def _handle_machine_alarm(self, event: Event) -> None:
        """Handle machine alarm events.
        
        Args:
            event: Alarm event.
        """
        logger.warning(f"Machine alarm: {event.data}")
        if self.ui:
            self.ui.show_message(f"Machine Alarm: {event.data}", "warning")
    
    def _handle_camera_error(self, event: Event) -> None:
        """Handle camera error events.
        
        Args:
            event: Error event.
        """
        logger.error(f"Camera error: {event.data}")
        if self.ui:
            self.ui.show_message(f"Camera Error: {event.data}", "error")
    
    def _handle_execution_completed(self, event: Event) -> None:
        """Handle execution completed events.
        
        Args:
            event: Completion event.
        """
        logger.info("Execution completed")
        if self.ui:
            self.ui.show_message("Job completed successfully!", "info")


def main():
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CNC G-code Sender Application")
    parser.add_argument(
        "--config", 
        type=Path, 
        help="Configuration file path"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    parser.add_argument(
        "--simulation", 
        action="store_true", 
        help="Run in simulation mode"
    )
    
    args = parser.parse_args()
    
    # Create and run application
    app = CNCApplication(args.config)
    
    # Override config with command line arguments
    if args.debug:
        app.config.debug_mode = True
    if args.simulation:
        app.config.simulation_mode = True
    
    app.run()


if __name__ == "__main__":
    main()