#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.z_stack_handling import batch_collapse_z_axis
from dna_condensation.core.preprocessor import bulk_preprocess_images

def test_preprocessing():
    print("=== TESTING PREPROCESSING FUNCTIONS ===")
    
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
    
    # Get config parameters
    channel_index = config.get("segmentation_channel_index")
    
    print(f"Channel index: {channel_index}")
    print(f"Image shapes: {[img.shape if img is not None else None for img in collapsed_images]}")
    
    # Test individual preprocessing methods
    print("\n--- Testing Background Correction ---")
    bg_corrected = bulk_preprocess_images(
        collapsed_images, 
        channel_index=channel_index,
        methods=['background_correction'],
        bg_ball_radius=50
    )
    
    print("\n--- Testing Intensity Normalization ---")
    normalized = bulk_preprocess_images(
        collapsed_images, 
        channel_index=channel_index,
        methods=['intensity_normalization'],
        norm_method='percentile',
        norm_percentile_range=(1, 99)
    )
    
    print("\n--- Testing Deconvolution ---")
    deconvolved = bulk_preprocess_images(
        collapsed_images, 
        channel_index=channel_index,
        methods=['deconvolution'],
        deconv_sigma=1.0,
        deconv_iterations=5  # Reduced for faster testing
    )
    
    print("\n--- Testing Combined Preprocessing ---")
    full_pipeline = bulk_preprocess_images(
        collapsed_images, 
        channel_index=channel_index,
        methods=['background_correction', 'intensity_normalization'],
        bg_ball_radius=50,
        norm_method='percentile'
    )
    
    print("✅ All preprocessing tests completed successfully!")
    
    # Print some statistics
    for i, (orig, bg, norm, deconv, full) in enumerate(zip(collapsed_images, bg_corrected, normalized, deconvolved, full_pipeline)):
        if orig is not None:
            if orig.ndim == 3:
                orig_ch = orig[:, :, channel_index]
                bg_ch = bg[:, :, channel_index]
                norm_ch = norm[:, :, channel_index]
                deconv_ch = deconv[:, :, channel_index]
                full_ch = full[:, :, channel_index]
            else:
                orig_ch = orig
                bg_ch = bg
                norm_ch = norm
                deconv_ch = deconv
                full_ch = full
                
            print(f"\nImage {i+1} statistics:")
            print(f"  Original: mean={orig_ch.mean():.1f}, std={orig_ch.std():.1f}, range=[{orig_ch.min()}-{orig_ch.max()}]")
            print(f"  Background corrected: mean={bg_ch.mean():.1f}, std={bg_ch.std():.1f}, range=[{bg_ch.min()}-{bg_ch.max()}]")
            print(f"  Normalized: mean={norm_ch.mean():.1f}, std={norm_ch.std():.1f}, range=[{norm_ch.min()}-{norm_ch.max()}]")
            print(f"  Deconvolved: mean={deconv_ch.mean():.1f}, std={deconv_ch.std():.1f}, range=[{deconv_ch.min()}-{deconv_ch.max()}]")
            print(f"  Full pipeline: mean={full_ch.mean():.1f}, std={full_ch.std():.1f}, range=[{full_ch.min()}-{full_ch.max()}]")

if __name__ == "__main__":
    test_preprocessing()
