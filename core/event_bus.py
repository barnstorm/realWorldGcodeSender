"""Event bus system for decoupled communication between modules."""

import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import threading
from queue import Queue, Empty
import weakref

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """Event data structure."""
    name: str
    data: Any = None
    timestamp: datetime = None
    priority: EventPriority = EventPriority.NORMAL
    source: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventBus:
    """Central event bus for application-wide event handling."""
    
    def __init__(self, async_mode: bool = True):
        """Initialize event bus.
        
        Args:
            async_mode: If True, events are processed asynchronously.
        """
        self.async_mode = async_mode
        self._subscribers: Dict[str, List[weakref.ref]] = {}
        self._event_queue: Queue = Queue()
        self._running = False
        self._worker_thread = None
        self._lock = threading.RLock()
        
        if async_mode:
            self.start()
    
    def start(self) -> None:
        """Start the event processing thread."""
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
            self._worker_thread.start()
            logger.info("Event bus started in async mode")
    
    def stop(self) -> None:
        """Stop the event processing thread."""
        if self._running:
            self._running = False
            # Add sentinel event to wake up thread
            self._event_queue.put(None)
            if self._worker_thread:
                self._worker_thread.join(timeout=5)
            logger.info("Event bus stopped")
    
    def subscribe(self, event_name: str, handler: Callable, 
                 weak: bool = True) -> None:
        """Subscribe to an event.
        
        Args:
            event_name: Name of the event to subscribe to.
                       Use '*' to subscribe to all events.
            handler: Callback function to handle the event.
            weak: If True, use weak reference to handler.
        """
        with self._lock:
            if event_name not in self._subscribers:
                self._subscribers[event_name] = []
            
            # Store as weak reference if requested
            if weak:
                ref = weakref.ref(handler, lambda r: self._clean_dead_refs(event_name))
            else:
                ref = weakref.ref(handler)
            
            self._subscribers[event_name].append(ref)
            logger.debug(f"Subscribed to event '{event_name}'")
    
    def unsubscribe(self, event_name: str, handler: Callable) -> None:
        """Unsubscribe from an event.
        
        Args:
            event_name: Name of the event.
            handler: Handler to remove.
        """
        with self._lock:
            if event_name in self._subscribers:
                # Remove matching weak references
                self._subscribers[event_name] = [
                    ref for ref in self._subscribers[event_name]
                    if ref() is not handler
                ]
                
                # Clean up empty subscription lists
                if not self._subscribers[event_name]:
                    del self._subscribers[event_name]
                
                logger.debug(f"Unsubscribed from event '{event_name}'")
    
    def publish(self, event: Event) -> None:
        """Publish an event.
        
        Args:
            event: Event to publish.
        """
        if self.async_mode:
            self._event_queue.put(event)
        else:
            self._dispatch_event(event)
    
    def emit(self, event_name: str, data: Any = None, 
            priority: EventPriority = EventPriority.NORMAL,
            source: Optional[str] = None) -> None:
        """Convenience method to emit an event.
        
        Args:
            event_name: Name of the event.
            data: Event data.
            priority: Event priority.
            source: Event source identifier.
        """
        event = Event(
            name=event_name,
            data=data,
            priority=priority,
            source=source
        )
        self.publish(event)
    
    def _process_events(self) -> None:
        """Process events from the queue (runs in separate thread)."""
        while self._running:
            try:
                # Get event with timeout to allow checking _running flag
                event = self._event_queue.get(timeout=0.1)
                
                if event is None:  # Sentinel value
                    break
                
                self._dispatch_event(event)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def _dispatch_event(self, event: Event) -> None:
        """Dispatch an event to all subscribers.
        
        Args:
            event: Event to dispatch.
        """
        with self._lock:
            # Get handlers for this specific event
            handlers = self._get_handlers(event.name)
            
            # Also get wildcard handlers
            if '*' in self._subscribers:
                handlers.extend(self._get_handlers('*'))
        
        # Call handlers outside of lock to prevent deadlocks
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for '{event.name}': {e}")
    
    def _get_handlers(self, event_name: str) -> List[Callable]:
        """Get active handlers for an event.
        
        Args:
            event_name: Name of the event.
        
        Returns:
            List of handler functions.
        """
        handlers = []
        if event_name in self._subscribers:
            # Get actual handlers from weak references
            for ref in self._subscribers[event_name]:
                handler = ref()
                if handler is not None:
                    handlers.append(handler)
        return handlers
    
    def _clean_dead_refs(self, event_name: str) -> None:
        """Clean up dead weak references.
        
        Args:
            event_name: Event name to clean.
        """
        with self._lock:
            if event_name in self._subscribers:
                self._subscribers[event_name] = [
                    ref for ref in self._subscribers[event_name]
                    if ref() is not None
                ]
                
                if not self._subscribers[event_name]:
                    del self._subscribers[event_name]
    
    def wait_for_event(self, event_name: str, timeout: Optional[float] = None) -> Optional[Event]:
        """Wait for a specific event to occur.
        
        Args:
            event_name: Name of the event to wait for.
            timeout: Maximum time to wait in seconds.
        
        Returns:
            The event if received, None if timeout.
        """
        received_event = None
        event_received = threading.Event()
        
        def handler(event: Event):
            nonlocal received_event
            received_event = event
            event_received.set()
        
        self.subscribe(event_name, handler, weak=False)
        
        try:
            if event_received.wait(timeout):
                return received_event
            return None
        finally:
            self.unsubscribe(event_name, handler)
    
    def clear(self) -> None:
        """Clear all subscriptions."""
        with self._lock:
            self._subscribers.clear()
            # Clear the queue
            while not self._event_queue.empty():
                try:
                    self._event_queue.get_nowait()
                except Empty:
                    break
        logger.info("Event bus cleared")
    
    def get_subscription_count(self, event_name: Optional[str] = None) -> int:
        """Get number of subscriptions.
        
        Args:
            event_name: Specific event name or None for total.
        
        Returns:
            Number of subscriptions.
        """
        with self._lock:
            if event_name:
                return len(self._get_handlers(event_name))
            else:
                return sum(len(self._get_handlers(name)) 
                          for name in self._subscribers.keys())
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance.
    
    Returns:
        Global EventBus instance.
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (mainly for testing)."""
    global _global_event_bus
    if _global_event_bus:
        _global_event_bus.stop()
        _global_event_bus = None


# Common event names as constants
class Events:
    """Common event name constants."""
    
    # Machine events
    MACHINE_CONNECTED = "machine.connected"
    MACHINE_DISCONNECTED = "machine.disconnected"
    MACHINE_STATE_CHANGED = "machine.state_changed"
    MACHINE_POSITION_CHANGED = "machine.position_changed"
    MACHINE_ERROR = "machine.error"
    MACHINE_ALARM = "machine.alarm"
    
    # Camera events
    CAMERA_CONNECTED = "camera.connected"
    CAMERA_DISCONNECTED = "camera.disconnected"
    CAMERA_FRAME_CAPTURED = "camera.frame_captured"
    CAMERA_ERROR = "camera.error"
    
    # Calibration events
    CALIBRATION_STARTED = "calibration.started"
    CALIBRATION_COMPLETED = "calibration.completed"
    CALIBRATION_FAILED = "calibration.failed"
    MARKERS_DETECTED = "calibration.markers_detected"
    
    # Path events
    PATH_LOADED = "path.loaded"
    PATH_MODIFIED = "path.modified"
    PATH_CLEARED = "path.cleared"
    PATH_OPTIMIZED = "path.optimized"
    
    # UI events
    UI_INITIALIZED = "ui.initialized"
    UI_CLOSED = "ui.closed"
    UI_CLICK = "ui.click"
    UI_KEY_PRESS = "ui.key_press"
    UI_FILE_SELECTED = "ui.file_selected"
    
    # Execution events
    EXECUTION_STARTED = "execution.started"
    EXECUTION_PAUSED = "execution.paused"
    EXECUTION_RESUMED = "execution.resumed"
    EXECUTION_STOPPED = "execution.stopped"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_PROGRESS = "execution.progress"