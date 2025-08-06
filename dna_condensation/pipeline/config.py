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
    """Simple configuration class that loads from config.yaml."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration from config.yaml file."""
        if config_file is None:
            config_file = Path(__file__).parent / 'config.yaml'
        
        self._config = self._load_config(config_file)
    
    def _load_config(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        if yaml is None:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        
        return config_data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
        
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