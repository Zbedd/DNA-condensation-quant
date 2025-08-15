"""
Simple configuration management for DNA condensation quantification pipeline.
Loads settings from config.yaml in the same directory.
"""
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None


class Config:
    """
    Configuration class that loads from config.yaml and supports dev overrides.
    If `dev_mode: true` is in config.yaml, it merges settings from dev_config.yaml.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration from YAML files."""
        if config_file is None:
            self.config_path = Path(__file__).parent / 'config.yaml'
        else:
            self.config_path = Path(config_file)
            
        self.dev_config_path = self.config_path.parent / 'dev_config.yaml'
        
        # Load base configuration
        self._config = self._load_config_file(self.config_path)
        
        # Check for dev mode and apply overrides
        if self.get('dev_mode', False):
            print("DEV MODE: Attempting to load dev_config.yaml...")
            try:
                dev_config = self._load_config_file(self.dev_config_path)
                self._config.update(dev_config)
                print("✅ DEV MODE: Successfully loaded and applied settings from dev_config.yaml.")
            except FileNotFoundError:
                print("⚠️ DEV MODE: dev_mode is true, but dev_config.yaml was not found.")
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from a single YAML file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        if yaml is None:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")
        
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        
        return config_data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
    
    def pop(self, key: str, default: Any = None) -> Any:
        """Remove and return configuration value."""
        return self._config.pop(key, default)
        
    def create_output_directories(self) -> None:
        """Create output and temp directories if they don't exist."""
        for path_key in ['output_path', 'temp_path']:
            path = self.get(path_key)
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self._config})"


# Global configuration instance - loads from config.yaml automatically
config = Config()