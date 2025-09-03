"""Matplotlib-based UI implementation."""

import logging
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import tkinter as tk
from tkinter import messagebox, filedialog
from .base import UIInterface

logger = logging.getLogger(__name__)


class MatplotlibUI(UIInterface):
    """Matplotlib-based user interface implementation."""
    
    def __init__(self, config: dict):
        """Initialize Matplotlib UI.
        
        Args:
            config: UI configuration dictionary.
        """
        super().__init__(config)
        self.figure = None
        self.ax = None
        self.canvas = None
        self.root = None
        self.current_image = None
        self.path_lines = []
        
    def initialize(self) -> bool:
        """Initialize the UI system.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Create the main figure
            self.figure = plt.figure(figsize=(12, 8))
            self.figure.suptitle("CNC G-Code Sender - Refactored Version")
            
            # Create main plot area
            self.ax = self.figure.add_subplot(111)
            self.ax.set_xlabel("X (inches)")
            self.ax.set_ylabel("Y (inches)")
            self.ax.grid(True, alpha=0.3)
            self.ax.set_aspect('equal')
            
            # Set up event handlers
            self.figure.canvas.mpl_connect('button_press_event', self._on_click)
            self.figure.canvas.mpl_connect('key_press_event', self._on_key_press)
            self.figure.canvas.mpl_connect('close_event', self._on_close)
            
            self.is_running = True
            logger.info("Matplotlib UI initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Matplotlib UI: {e}")
            return False
    
    def run(self) -> None:
        """Run the main UI event loop."""
        if not self.is_running:
            if not self.initialize():
                return
        
        try:
            logger.info("Starting Matplotlib UI event loop")
            
            # Add some sample content
            self.ax.text(0.5, 0.5, "CNC G-Code Sender\nRefactored Architecture\n\nPhase 1: Foundation Complete", 
                        transform=self.ax.transAxes,
                        fontsize=16, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            # Show plot
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in UI event loop: {e}")
        finally:
            self.is_running = False
    
    def stop(self) -> None:
        """Stop the UI and clean up resources."""
        self.is_running = False
        if self.figure:
            plt.close(self.figure)
        logger.info("Matplotlib UI stopped")
    
    def display_image(self, image: np.ndarray, window_name: str = "main") -> None:
        """Display an image in the UI.
        
        Args:
            image: Image array to display.
            window_name: Name of the window to display in.
        """
        if not self.ax:
            return
        
        # Clear current display
        self.ax.clear()
        
        # Display image
        self.current_image = self.ax.imshow(image, aspect='equal')
        self.ax.set_title(f"Camera View - {window_name}")
        
        # Refresh display
        self.figure.canvas.draw()
    
    def display_path(self, path_points: List[Tuple[float, float]], 
                    window_name: str = "main") -> None:
        """Display a path overlay.
        
        Args:
            path_points: List of (x, y) coordinates.
            window_name: Name of the window to display in.
        """
        if not self.ax or not path_points:
            return
        
        # Extract x and y coordinates
        x_coords = [p[0] for p in path_points]
        y_coords = [p[1] for p in path_points]
        
        # Plot path
        line, = self.ax.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.7)
        self.path_lines.append(line)
        
        # Refresh display
        self.figure.canvas.draw()
    
    def show_message(self, message: str, message_type: str = "info") -> None:
        """Show a message to the user.
        
        Args:
            message: Message text.
            message_type: Type of message (info, warning, error).
        """
        logger.log(
            logging.INFO if message_type == "info" else 
            logging.WARNING if message_type == "warning" else 
            logging.ERROR,
            f"UI Message [{message_type}]: {message}"
        )
        
        # Also show in the plot title
        if self.ax:
            current_title = self.ax.get_title()
            self.ax.set_title(f"{current_title}\n[{message_type.upper()}] {message}")
            self.figure.canvas.draw()
    
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
        # Create temporary tkinter root if needed
        temp_root = False
        if not self.root:
            self.root = tk.Tk()
            self.root.withdraw()
            temp_root = True
        
        try:
            if dialog_type == "info":
                messagebox.showinfo(title, message)
                return "ok"
            elif dialog_type == "question":
                result = messagebox.askyesno(title, message)
                return "yes" if result else "no"
            elif dialog_type == "input":
                # Simple input dialog
                result = tk.simpledialog.askstring(title, message)
                return result
            else:
                return None
                
        finally:
            if temp_root and self.root:
                self.root.destroy()
                self.root = None
    
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
        # Create temporary tkinter root if needed
        temp_root = False
        if not self.root:
            self.root = tk.Tk()
            self.root.withdraw()
            temp_root = True
        
        try:
            if file_types is None:
                file_types = [("All Files", "*.*")]
            
            if save:
                file_path = filedialog.asksaveasfilename(
                    title=title,
                    filetypes=file_types
                )
            else:
                file_path = filedialog.askopenfilename(
                    title=title,
                    filetypes=file_types
                )
            
            return Path(file_path) if file_path else None
            
        finally:
            if temp_root and self.root:
                self.root.destroy()
                self.root = None
    
    def update_status(self, status: Dict[str, Any]) -> None:
        """Update status display.
        
        Args:
            status: Status information dictionary.
        """
        if not self.figure:
            return
        
        # Create status text
        status_text = "\n".join([f"{k}: {v}" for k, v in status.items()])
        
        # Update figure text
        self.figure.text(0.02, 0.98, status_text,
                         transform=self.figure.transFigure,
                         fontsize=10, va='top',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Refresh display
        self.figure.canvas.draw()
    
    def create_settings_panel(self, settings: Dict[str, Any]) -> Any:
        """Create settings configuration panel.
        
        Args:
            settings: Settings dictionary.
        
        Returns:
            Settings panel widget/handle.
        """
        # For now, just log that settings panel was requested
        logger.info(f"Settings panel requested with {len(settings)} settings")
        return None
    
    def _on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes == self.ax:
            self.trigger_event('click', x=event.xdata, y=event.ydata, button=event.button)
            logger.debug(f"Click at ({event.xdata:.2f}, {event.ydata:.2f})")
    
    def _on_key_press(self, event):
        """Handle key press events."""
        self.trigger_event('key_press', key=event.key)
        logger.debug(f"Key pressed: {event.key}")
        
        # Handle some basic keys
        if event.key == 'escape' or event.key == 'q':
            self.stop()
    
    def _on_close(self, event):
        """Handle window close event."""
        logger.info("UI window closed")
        self.is_running = False
        self.trigger_event('window_closed')