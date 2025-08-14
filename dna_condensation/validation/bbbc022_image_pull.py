#!/usr/bin/env python3
"""
BBBC022 Image Pull and Processing

This module handles fetching, processing, and formatting BBBC022 images
for use in the DNA condensation analysis pipeline. It provides the same
output format as ND2 processing to ensure seamless integration.
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dna_condensation.core.image_loader import load_bbbc022_images
from dna_condensation.core.segmentation import bulk_segment_images
from dna_condensation.core.preprocessor import bulk_preprocess_images


"""
NOTE: Grouping utilities retained below for reference, but commented out to avoid accidental use.
Grouping is now performed in the unified batch processor via config-provided well mapping.

def identify_bbbc022_groups(...):
    pass
"""


"""
def add_experimental_metadata(...):
    pass
"""


def process_bbbc022_images(bbbc022_config: Dict[str, Any], 
                          segmentation_config: Dict[str, Any],
                          preprocessing_config: Dict[str, Any],
                          output_dir: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[Dict]]:
    """
    Process BBBC022 images through the complete pipeline to match ND2 output format.
    
    Parameters:
    -----------
    bbbc022_config : Dict
        BBBC022-specific configuration
    segmentation_config : Dict  
        Segmentation configuration
    preprocessing_config : Dict
        Preprocessing configuration
    output_dir : Path
        Output directory for BBBC022 data
        
    Returns:
    --------
    Tuple[List[np.ndarray], List[np.ndarray], List[str], List[Dict]]
        (final_images, masks, image_names, metadata) - same format as ND2 pipeline
    """
    print("="*60)
    print("PROCESSING BBBC022 IMAGES")
    print("="*60)
    
    # Step 1: Load BBBC022 images
    print("\n1. Loading BBBC022 images...")
    images, metadata = load_bbbc022_images(
        count=bbbc022_config.get('count', 20),
        channels=bbbc022_config.get('channels', ['OrigHoechst']),
        seed=bbbc022_config.get('seed', 42),
        output_dir=str(output_dir / "bbbc022_data")
    )
    
    print(f"✓ Loaded {len(images)} images")
    
    # Step 2-3: Grouping is handled in batch_processor; here we only ensure image_name exists
    print("\n2. Preparing metadata (grouping handled by batch_processor)...")
    enhanced_metadata = []
    for i, m in enumerate(metadata):
        mm = m.copy()
        if 'image_name' not in mm:
            mm['image_name'] = mm.get('filename', f'bbbc022_image_{i+1}')
        enhanced_metadata.append(mm)
    image_names = [m['image_name'] for m in enhanced_metadata]
    
    # Step 3: Convert images to uint8 if needed (BBBC022 images come as float32)
    print("\n3. Converting image data types...")
    converted_images = []
    for i, img in enumerate(images):
        if img.dtype != np.uint8:
            if img.dtype == np.float32:
                # Check the range of float32 values first
                img_min, img_max = img.min(), img.max()
                print(f"Image {i+1}: float32 range [{img_min:.6f}, {img_max:.6f}]")
                
                if img_max <= 1.0:
                    # Values are in [0, 1] range - scale to [0, 255]
                    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
                else:
                    # Values might already be in [0, 255] range or larger
                    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
                    
                print(f"  Converted to uint8 range [{img_uint8.min()}, {img_uint8.max()}]")
            else:
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            converted_images.append(img_uint8)
        else:
            converted_images.append(img)
    
    # Debug: Check first converted image
    if converted_images:
        first_img = converted_images[0]
        print(f"First converted image: shape={first_img.shape}, dtype={first_img.dtype}")
        print(f"  Value range: [{first_img.min()}, {first_img.max()}]")
        print(f"  Mean: {first_img.mean():.3f}, Std: {first_img.std():.3f}")
    
    # Step 5: Apply global preprocessing (same as ND2 pipeline)
    print("\n5. Applying global preprocessing...")
    methods = []
    if preprocessing_config.get("deconvolution"):
        methods.append("deconvolution")
    if preprocessing_config.get("background_correction"):
        methods.append("background_correction") 
    if preprocessing_config.get("intensity_normalization"):
        methods.append("intensity_normalization")
    
    if methods:
        print(f"Applying global preprocessing: {' → '.join(methods)}")
        global_preprocessed = bulk_preprocess_images(
            converted_images,
            channel_index=segmentation_config.get("segmentation_channel_index", 0),
            background_correction=preprocessing_config.get("background_correction", False),
            deconvolution=preprocessing_config.get("deconvolution", False),
            intensity_normalization=preprocessing_config.get("intensity_normalization", False),
            bg_ball_radius=preprocessing_config.get("bg_ball_radius", 50),
            deconv_iterations=preprocessing_config.get("deconv_iterations", 10),
            norm_method=preprocessing_config.get("norm_method", "percentile")
        )
    else:
        global_preprocessed = converted_images
    
    # Step 6: Segmentation
    print("\n6. Segmenting nuclei...")
    masks = bulk_segment_images(
        global_preprocessed,
        method=segmentation_config.get("segmentation_method", "otsu"),
        channel_index=segmentation_config.get("segmentation_channel_index", 0)
    )
    
    # Step 7: Per-nucleus preprocessing (if enabled, same as ND2 pipeline)
    final_images = global_preprocessed
    if preprocessing_config.get("per_nucleus_normalization"):
        print("\n7. Applying per-nucleus normalization...")
        from dna_condensation.core.preprocessor import per_nucleus_intensity_normalization
        
        per_nucleus_preprocessed = []
        for image, mask in zip(global_preprocessed, masks):
            if mask is not None:
                norm_image, _ = per_nucleus_intensity_normalization(image, mask)
                per_nucleus_preprocessed.append(norm_image)
            else:
                per_nucleus_preprocessed.append(image)
        
        final_images = per_nucleus_preprocessed
    
    print(f"\n✓ BBBC022 processing complete!")
    print(f"✓ Final output: {len(final_images)} images, {len(masks)} masks")
    
    return final_images, masks, image_names, enhanced_metadata


if __name__ == "__main__":
    # Test the BBBC022 processor
    from dna_condensation.pipeline.config import Config
    
    config = Config()
    bbbc022_config = config.get("bbbc022_settings", {})
    
    output_dir = Path("validation_output_test")
    output_dir.mkdir(exist_ok=True)
    
    try:
        images, masks, names, metadata = process_bbbc022_images(
            bbbc022_config,
            {"segmentation_method": "otsu", "segmentation_channel_index": 0},
            {"background_correction": True, "intensity_normalization": True},
            output_dir
        )
        
        print(f"\nTest completed successfully!")
        print(f"Processed {len(images)} images with {len([m for m in masks if m is not None])} successful segmentations")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
