from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Imports
import os
from typing import List, Optional
from nd2reader import ND2Reader
from dna_condensation.pipeline.config import config

def get_nd2_objects(path: Optional[str] = None) -> List[ND2Reader]:
    """
    Return a list of ND2Reader objects for all .nd2 files in the specified path.
    
    Args:
        path: Directory path containing .nd2 files. If None, uses config raw_nd2_path.
        
    Returns:
        List of ND2Reader objects for each .nd2 file found.
        
    Raises:
        FileNotFoundError: If the path doesn't exist.
        ValueError: If no .nd2 files are found in the path.
    """
    # Use provided path or fall back to config
    if path is None:
        path = config.get('raw_nd2_path')
    
    if not path:
        raise ValueError("No path provided and no raw_nd2_path configured")
    
    # Convert to Path object for easier handling
    nd2_path = Path(path)
    
    # Check if path exists
    if not nd2_path.exists():
        raise FileNotFoundError(f"Path does not exist: {nd2_path}")
    
    # Find all .nd2 files in the directory
    nd2_files = list(nd2_path.glob("*.nd2"))
    
    if not nd2_files:
        raise ValueError(f"No .nd2 files found in: {nd2_path}")
    
    # Create ND2Reader objects for each file
    nd2_objects = []
    for nd2_file in nd2_files:
        try:
            nd2_reader = ND2Reader(str(nd2_file))
            nd2_objects.append(nd2_reader)
            print(f"Loaded: {nd2_file.name}")
        except Exception as e:
            print(f"Warning: Could not load {nd2_file.name}: {e}")
    
    if not nd2_objects:
        raise ValueError("No valid .nd2 files could be loaded")
    
    print(f"Successfully loaded {len(nd2_objects)} ND2 files")
    return nd2_objects
