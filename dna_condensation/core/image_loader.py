from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Imports
import os
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from nd2reader import ND2Reader
from pathlib import Path
from dna_condensation.pipeline.config import config

# Define global cache directory outside of Git tracking
BBBC022_CACHE_DIR = project_root / ".bbbc022_cache"

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


def load_bbbc022_images(
    count: int = 10,
    channels: Optional[List[str]] = None,
    roles: Optional[List[str]] = None,
    wells: Optional[List[str]] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    max_plates: int = 2,
    use_cache: bool = True,
    nuclei_only: bool = True
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Load BBBC022 microscopy images using imageProcessingUtils with intelligent caching.
    
    This function leverages the imageProcessingUtils package to download and load
    BBBC022 images with specified filtering criteria. Downloads are cached outside
    of Git tracking for efficiency across multiple validation runs.
    
    Args:
        count: Number of images to sample
        channels: List of channels to include (e.g., ['OrigHoechst'] for DNA, ['OrigTubulin'] for tubulin)
                 If None, defaults to ['OrigHoechst'] for DNA analysis
        roles: List of well roles to include ('compound' for treatments, 'mock' for controls)
               Note: Current imageProcessingUtils implementation doesn't filter by roles directly.
               Use BBBC022 metadata CSV to map wells to roles after loading.
        wells: List of specific wells to include (e.g., ['A01', 'A02'])
               If None, samples from all available wells
        seed: Random seed for reproducible results
        output_dir: Directory to cache downloaded images (optional, will use global cache if None)
        max_plates: Maximum number of plates to sample from
        use_cache: If True, use persistent cache directory to avoid re-downloading zip files
        nuclei_only: If True, focus on nuclei segmentation for DNA condensation analysis
        
    Returns:
        Tuple of (images, metadata)
        - images: List of numpy arrays (H, W) for single channel or (H, W, C) for multiple channels
        - metadata: List of dictionaries containing image metadata including well information
        
    Raises:
        ImportError: If imageProcessingUtils is not properly installed
        ValueError: If no images could be loaded with the specified criteria
        
    Examples:
        # Load control (mock) wells with caching
        control_images, control_metadata = load_bbbc022_images(
            count=20, 
            channels=['OrigHoechst'], 
            wells=['A13', 'A14', 'B13', 'B14']  # Known mock wells
        )
        
        # Load treatment wells with DNA condensation compounds
        treatment_images, treatment_metadata = load_bbbc022_images(
            count=20,
            channels=['OrigHoechst'],
            wells=['E03', 'M07', 'M21']  # Wells with staurosporine/camptothecin
        )
        
    References:
        - Broad Bioimage Benchmark Collection: https://bbbc.broadinstitute.org/BBBC022
        - Original paper: Gustafsdottir et al. (2013) PLoS ONE
        - Metadata: https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_image.csv
    """
    
    try:
        from imageProcessingUtils.sample_data import fetch_bbbc022_samples
    except ImportError as e:
        raise ImportError(
            "imageProcessingUtils package is required for BBBC022 functionality. "
            "Install with: pip install git+https://github.com/Zbedd/imageProcessingUtils.git"
        ) from e
    
    # Set default parameters optimized for DNA condensation analysis
    if channels is None:
        channels = ['OrigHoechst']  # DNA channel for condensation analysis
    
    # Setup intelligent caching
    if use_cache and output_dir is None:
        # Use global cache directory outside Git tracking
        cache_dir = BBBC022_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(cache_dir)
        print(f"Using persistent cache: {cache_dir}")
        
        # Check if required zip files already exist in cache
        cache_raw_dir = cache_dir / "raw"
        existing_zips = []
        if cache_raw_dir.exists():
            existing_zips = list(cache_raw_dir.glob("BBBC022_v1_images_*.zip"))
        
        cache_available = len(existing_zips) >= 2
        if cache_available:
            print(f"✓ Found {len(existing_zips)} cached zip files - using cached data")
            print(f"  Cached files: {[z.name for z in existing_zips[:2]]}")
        else:
            print(f"Cache has {len(existing_zips)} zip files - will download missing files")
            
    elif output_dir:
        print(f"Using specified output directory: {output_dir}")
        cache_available = False
    else:
        cache_available = False
    
    print(f"Loading {count} BBBC022 images...")
    print(f"Channels: {channels}")
    print(f"Roles: {roles if roles else 'All available'}")
    print(f"Wells: {wells if wells else 'All available'}")
    
    try:
        if use_cache and cache_available:
            # Check if processed cache is available (faster than re-processing)
            processed_dir = cache_dir / "processed"
            metadata_file = cache_dir / "metadata.json"
            
            if processed_dir.exists() and metadata_file.exists():
                # Load from processed cache
                import json
                with open(metadata_file, 'r') as f:
                    cached_metadata = json.load(f)
                
                if len(cached_metadata) >= count:
                    print(f"✓ Loading from processed cache ({len(cached_metadata)} images available)")
                    
                    # Load the requested number of images
                    selected_metadata = cached_metadata[:count] if count <= len(cached_metadata) else cached_metadata
                    
                    cached_images = []
                    for meta in selected_metadata:
                        processed_path = meta.get('processed_path')
                        if processed_path and Path(processed_path).exists():
                            img = np.load(processed_path)
                            cached_images.append(img)
                        else:
                            # Fallback to original processing if processed file missing
                            print(f"Warning: Missing processed file {processed_path}, falling back to full processing")
                            break
                    
                    if len(cached_images) == len(selected_metadata):
                        # Successfully loaded all from cache
                        print(f"✓ Loaded {len(cached_images)} images from cache in <1 second")
                        images = cached_images
                        metadata = selected_metadata
                    else:
                        # Fall through to full processing
                        print("Cache incomplete, falling back to full processing...")
                        images, metadata = fetch_bbbc022_samples(
                            count=count,
                            channels=channels,
                            wells=wells,
                            nuclei_only=nuclei_only,
                            seed=seed,
                            output_dir=output_dir,
                            max_plates=max_plates
                        )
                else:
                    print(f"Cache has {len(cached_metadata)} images, need {count} - processing more...")
                    images, metadata = fetch_bbbc022_samples(
                        count=count,
                        channels=channels,
                        wells=wells,
                        nuclei_only=nuclei_only,
                        seed=seed,
                        output_dir=output_dir,
                        max_plates=max_plates
                    )
            else:
                print("No processed cache found, using zip files...")
                images, metadata = fetch_bbbc022_samples(
                    count=count,
                    channels=channels,
                    wells=wells,
                    nuclei_only=nuclei_only,
                    seed=seed,
                    output_dir=output_dir,
                    max_plates=max_plates
                )
        else:
            # Download fresh data or use specified directory
            if cache_available:
                print("Cache ignored - downloading fresh data...")
            else:
                print("Downloading BBBC022 data...")
            images, metadata = fetch_bbbc022_samples(
                count=count,
                channels=channels,
                wells=wells,
                nuclei_only=nuclei_only,
                seed=seed,
                output_dir=output_dir,
                max_plates=max_plates
            )
        
        if not images:
            raise ValueError(
                f"No images could be loaded with the specified criteria. "
                f"Try increasing count or relaxing filters."
            )
        
        # Ensure images are in the expected format for the pipeline
        processed_images = []
        for img in images:
            if img.dtype != np.float32 and img.dtype != np.float64:
                # Convert to float for processing pipeline compatibility
                img = img.astype(np.float32)
            processed_images.append(img)
        
        print(f"✓ Successfully loaded {len(processed_images)} BBBC022 images")
        print(f"  Image shape: {processed_images[0].shape}")
        print(f"  Data type: {processed_images[0].dtype}")
        
        # Count wells in final dataset for information
        if metadata:
            well_counts = {}
            for meta in metadata:
                well = meta.get('well', 'Unknown')
                well_counts[well] = well_counts.get(well, 0) + 1
            print(f"  Well distribution: {dict(sorted(well_counts.items())[:5])}{'...' if len(well_counts) > 5 else ''}")
        
        return processed_images, metadata
        
    except Exception as e:
        raise ValueError(f"Failed to load BBBC022 images: {e}") from e
