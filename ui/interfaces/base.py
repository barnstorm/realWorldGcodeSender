"""Abstract base class for UI interfaces."""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict, List, Tuple
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UIInterface(ABC):
    """Abstract UI interface for different UI backends."""
    
    def __init__(self, config: dict):
        """Initialize UI interface.
        
        Args:
            config: UI configuration dictionary.
        """
        self.config = config
        self.is_running = False
        self.event_handlers: Dict[str, List[Callable]] = {}
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the UI system.
        
        Returns:
            True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def run(self) -> None:
        """Run the main UI event loop."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the UI and clean up resources."""
        pass
    
    @abstractmethod
    def display_image(self, image: np.ndarray, window_name: str = "main") -> None:
        """Display an image in the UI.
        
        Args:
            image: Image array to display.
            window_name: Name of the window to display in.
        """
        pass
    
    @abstractmethod
    def display_path(self, path_points: List[Tuple[float, float]], 
                    window_name: str = "main") -> None:
        """Display a path overlay.
        
        Args:
            path_points: List of (x, y) coordinates.
            window_name: Name of the window to display in.
        """
        pass
    
    @abstractmethod
    def show_message(self, message: str, message_type: str = "info") -> None:
        """Show a message to the user.
        
        Args:
            message: Message text.
            message_type: Type of message (info, warning, error).
        """
        pass
    
    @abstractmethod
    def show_dialog(self, title: str, message: str, 
                   dialog_type: str = "info") -> Optional[str]:
        """Show a dialog and get user response.
        
        Args:
            title: Dialog title.
            message: Dialog message.
            dialog_type: Type of dialog (info, question, input).
        
        Returns:
            User response or None.
        """
        pass
    
    @abstractmethod
    def get_file_path(self, title: str = "Select File", 
                     file_types: Optional[List[Tuple[str, str]]] = None,
                     save: bool = False) -> Optional[Path]:
        """Show file dialog.
        
        Args:
            title: Dialog title.
            file_types: List of (description, extension) tuples.
            save: True for save dialog, False for open dialog.
        
        Returns:
            Selected file path or None.
        """
        pass
    
    @abstractmethod
    def update_status(self, status: Dict[str, Any]) -> None:
        """Update status display.
        
        Args:
            status: Status information dictionary.
        """
        pass
    
    @abstractmethod
    def create_settings_panel(self, settings: Dict[str, Any]) -> Any:
        """Create settings configuration panel.
        
        Args:
            settings: Settings dictionary.
        
        Returns:
            Settings panel widget/handle.
        """
        pass
    
    def register_event_handler(self, event: str, handler: Callable) -> None:
        """Register an event handler.
        
        Args:
            event: Event name (e.g., 'click', 'key_press').
            handler: Callback function.
        """
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
        logger.debug(f"Registered handler for event: {event}")
    
    def unregister_event_handler(self, event: str, handler: Callable) -> None:
        """Unregister an event handler.
        
        Args:
            event: Event name.
            handler: Callback function to remove.
        """
        if event in self.event_handlers:
            try:
                self.event_handlers[event].remove(handler)
                logger.debug(f"Unregistered handler for event: {event}")
            except ValueError:
                logger.warning(f"Handler not found for event: {event}")
    
    def trigger_event(self, event: str, *args, **kwargs) -> None:
        """Trigger an event and call all registered handlers.
        
        Args:
            event: Event name.
            *args: Positional arguments for handlers.
            **kwargs: Keyword arguments for handlers.
        """
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in event handler for {event}: {e}")
    
    def set_theme(self, theme: str) -> None:
        """Set UI theme.
        
        Args:
            theme: Theme name.
        """
        self.config['theme'] = theme
        logger.info(f"UI theme set to: {theme}")
    
    def get_theme(self) -> str:
        """Get current UI theme.
        
        Returns:
            Current theme name.
        """
        return self.config.get('theme', 'default')