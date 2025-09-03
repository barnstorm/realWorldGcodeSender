"""Logging configuration for the CNC application."""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if sys.platform == 'win32':
            # Enable ANSI colors on Windows
            import os
            os.system('')
        
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    verbose: bool = False
) -> None:
    """Set up application logging.
    
    Args:
        log_dir: Directory to store log files. Defaults to 'logs'.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console_output: Enable console logging.
        file_output: Enable file logging.
        verbose: Enable verbose output (sets DEBUG level).
    """
    if log_dir is None:
        log_dir = Path("logs")
    
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set log level
    if verbose:
        log_level = "DEBUG"
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if file_output:
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"cnc_{timestamp}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Also create a latest.log symlink/copy for easy access
        latest_log = log_dir / "latest.log"
        if latest_log.exists():
            latest_log.unlink()
        
        # Create a separate handler for latest.log
        latest_handler = logging.FileHandler(latest_log)
        latest_handler.setLevel(level)
        latest_handler.setFormatter(file_formatter)
        root_logger.addHandler(latest_handler)
    
    # Log startup message
    logging.info("=" * 60)
    logging.info("CNC G-Code Sender Application Started")
    logging.info(f"Log Level: {log_level}")
    logging.info(f"Log Directory: {log_dir}")
    logging.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Name of the logger (usually __name__).
    
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for temporary log level changes."""
    
    def __init__(self, level: str):
        """Initialize log context.
        
        Args:
            level: Temporary log level.
        """
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.previous_level = None
        self.logger = logging.getLogger()
    
    def __enter__(self):
        """Enter context and save current level."""
        self.previous_level = self.logger.level
        self.logger.setLevel(self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous level."""
        self.logger.setLevel(self.previous_level)


class OperationLogger:
    """Logger for tracking long-running operations."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """Initialize operation logger.
        
        Args:
            operation_name: Name of the operation.
            logger: Logger instance to use.
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
    
    def __enter__(self):
        """Start operation timing."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log operation completion or failure."""
        duration = datetime.now() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation_name} in {duration.total_seconds():.2f} seconds"
            )
        else:
            self.logger.error(
                f"Failed {self.operation_name} after {duration.total_seconds():.2f} seconds: {exc_val}"
            )
    
    def progress(self, message: str, percentage: Optional[float] = None):
        """Log progress update.
        
        Args:
            message: Progress message.
            percentage: Optional completion percentage (0-100).
        """
        if percentage is not None:
            self.logger.info(f"{self.operation_name}: [{percentage:.1f}%] {message}")
        else:
            self.logger.info(f"{self.operation_name}: {message}")