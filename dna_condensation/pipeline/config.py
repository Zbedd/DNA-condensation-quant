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
                self._deep_merge(self._config, dev_config)
                print("✅ DEV MODE: Successfully loaded and applied settings from dev_config.yaml.")
            except FileNotFoundError:
                print("⚠️ DEV MODE: dev_mode is true, but dev_config.yaml was not found.")
    
    def _deep_merge(self, source: Dict[str, Any], destination: Dict[str, Any]) -> None:
        """
        Recursively merge dictionaries.
        
        `destination` is merged into `source`.
        """
        for key, value in destination.items():
            if isinstance(value, dict) and key in source and isinstance(source.get(key), dict):
                self._deep_merge(source[key], value)
            else:
                source[key] = value

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

    # Convenience helpers
    def get_nuclear_channel_index(self) -> int:
        """
        Return the nuclear DNA channel index based on input_source.

        - For input_source == 'nd2': reads nd2_selection_settings.nuclear_channel_index
        - For input_source == 'bbbc022': reads bbbc022_settings.nuclear_channel_index
        Falls back to 0 if not set.
        """
        src = str(self.get('input_source', 'nd2')).lower()
        if src == 'nd2':
            nd2_cfg = self.get('nd2_selection_settings', {}) or {}
            return int(nd2_cfg.get('nuclear_channel_index', 0))
        elif src == 'bbbc022':
            bcfg = self.get('bbbc022_settings', {}) or {}
            return int(bcfg.get('nuclear_channel_index', 0))
        # Unknown source → conservative default
        return 0


# Global configuration instance - loads from config.yaml automatically
config = Config()