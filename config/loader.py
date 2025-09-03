"""Configuration loader for YAML and JSON configuration files."""

import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from .settings import ApplicationConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages application configuration."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files.
                       Defaults to 'config/defaults' in project root.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent / "defaults"
        self.config_dir = Path(config_dir)
        self.config: Optional[ApplicationConfig] = None
        
    def load(self, config_file: Optional[Path] = None) -> ApplicationConfig:
        """Load configuration from file.
        
        Args:
            config_file: Path to configuration file (YAML or JSON).
                        If None, loads from default locations.
        
        Returns:
            ApplicationConfig object with loaded settings.
        """
        if config_file:
            config_data = self._load_file(config_file)
        else:
            config_data = self._load_defaults()
        
        # Merge with environment variables if needed
        config_data = self._merge_env_vars(config_data)
        
        # Create configuration object
        self.config = ApplicationConfig.from_dict(config_data)
        
        logger.info(f"Configuration loaded successfully")
        return self.config
    
    def save(self, config: ApplicationConfig, file_path: Path) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration object to save.
            file_path: Path to save configuration to.
        """
        config_data = config.to_dict()
        
        if file_path.suffix == ".yaml" or file_path.suffix == ".yml":
            with open(file_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        elif file_path.suffix == ".json":
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Configuration saved to {file_path}")
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from a single file.
        
        Args:
            file_path: Path to configuration file.
        
        Returns:
            Dictionary with configuration data.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f) or {}
            elif file_path.suffix == ".json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration from defaults directory.
        
        Returns:
            Merged dictionary of all default configurations.
        """
        config_data = {}
        
        # Load each default configuration file
        default_files = [
            "machine.yaml",
            "calibration.yaml",
            "cutting.yaml",
            "application.yaml"
        ]
        
        for filename in default_files:
            file_path = self.config_dir / filename
            if file_path.exists():
                try:
                    file_data = self._load_file(file_path)
                    # Merge into main config
                    config_data.update(file_data)
                    logger.debug(f"Loaded default config: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
        
        return config_data
    
    def _merge_env_vars(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge environment variables into configuration.
        
        Environment variables should be prefixed with 'CNC_' and use
        double underscores for nested values.
        Example: CNC_MACHINE__PORT=COM3
        
        Args:
            config_data: Current configuration dictionary.
        
        Returns:
            Configuration dictionary with environment variables merged.
        """
        import os
        
        prefix = "CNC_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Split by double underscore for nested values
                keys = config_key.split("__")
                
                # Navigate to the correct position in config
                current = config_data
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                # Set the value (attempt to parse as JSON for complex types)
                try:
                    current[keys[-1]] = json.loads(value)
                except json.JSONDecodeError:
                    current[keys[-1]] = value
                
                logger.debug(f"Loaded environment variable: {key}")
        
        return config_data
    
    def validate(self, config: Optional[ApplicationConfig] = None) -> bool:
        """Validate configuration for correctness.
        
        Args:
            config: Configuration to validate. Uses loaded config if None.
        
        Returns:
            True if configuration is valid, False otherwise.
        """
        if config is None:
            config = self.config
        
        if config is None:
            logger.error("No configuration to validate")
            return False
        
        # Validate machine settings
        if config.machine.baudrate <= 0:
            logger.error("Invalid baudrate")
            return False
        
        if config.machine.bed_size.x <= 0 or config.machine.bed_size.y <= 0:
            logger.error("Invalid bed size")
            return False
        
        # Validate cutting settings
        if config.cutting.cutter_diameter <= 0:
            logger.error("Invalid cutter diameter")
            return False
        
        if config.cutting.feedrate <= 0:
            logger.error("Invalid feedrate")
            return False
        
        # Validate calibration settings
        if config.calibration.markers.box_width <= 0:
            logger.error("Invalid marker box width")
            return False
        
        return True
    
    def get_config(self) -> Optional[ApplicationConfig]:
        """Get the currently loaded configuration.
        
        Returns:
            Current configuration or None if not loaded.
        """
        return self.config
    
    def reload(self) -> ApplicationConfig:
        """Reload configuration from files.
        
        Returns:
            Reloaded configuration.
        """
        logger.info("Reloading configuration")
        return self.load()