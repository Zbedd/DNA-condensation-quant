#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.preprocessor import batch_collapse_z_axis
from dna_condensation.core.segmentation import bulk_segment_images

def test_20_percent_filter():
    print("=== TESTING 20% SIZE FILTER ===")
    
    # Initialize config
    config = Config()
    
    # Load a small set of test images
    nd2_folder_path = config.get("raw_nd2_path")
    nd2_objects = get_nd2_objects(nd2_folder_path)
    print(f"Loaded {len(nd2_objects)} ND2 files")
    
    # Take first 2 files for quick test
    test_objects = nd2_objects[:2]
    print(f"Testing with {len(test_objects)} files")
    
    # Collapse z-axis
    print("Collapsing z-axis...")
    collapsed_images = batch_collapse_z_axis(test_objects, method='mean')
    
    # Get config parameters (no defaults)
    channel_index = config.get("segmentation_channel_index")
    segmentation_method = config.get("segmentation_method")
    size_filter_config = config.get("size_filtering")
    
    print(f"Channel index: {channel_index}")
    print(f"Segmentation method: {segmentation_method}")
    print(f"Size filter config: {size_filter_config}")
    
    # Test segmentation with 20% filter
    filter_enabled = size_filter_config.get('enabled')
    filter_percentage = size_filter_config.get('min_size_percentage')
    
    print(f"Size filtering: {'enabled' if filter_enabled else 'disabled'}")
    print(f"Filter percentage: {filter_percentage}%")
    
    # Run segmentation
    masks = bulk_segment_images(
        collapsed_images, 
        channel_index=channel_index, 
        method=segmentation_method,
        size_filter_config=size_filter_config
    )
    
    print(f"Segmentation completed successfully!")
    print(f"Generated {len(masks)} masks")
    
    print("✅ 20% filter test completed successfully!")

if __name__ == "__main__":
    test_20_percent_filter()
