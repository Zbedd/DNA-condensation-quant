import sys
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.preprocessor import batch_collapse_z_axis
from dna_condensation.core.segmentation import bulk_segment_images

def test_batch_processor():
    """Test the batch processor with current config settings."""
    print("=== TESTING UPDATED BATCH PROCESSOR ===\n")
    
    # Load config
    config = Config()
    nd2_folder_path = config.get("raw_nd2_path")
    channel_index = config.get("segmentation_channel_index", 0)
    segmentation_method = config.get("segmentation_method", "yolo")
    
    print(f"ND2 folder: {nd2_folder_path}")
    print(f"Segmentation method: {segmentation_method}")
    print(f"Channel index: {channel_index}")
    
    # Get first 2 ND2 files for testing
    nd2_objects = get_nd2_objects(nd2_folder_path)
    test_objects = nd2_objects[:2]
    
    print(f"\nProcessing {len(test_objects)} files:")
    for i, obj in enumerate(test_objects):
        print(f"  {i+1}. {obj.filename}")
    
    # Collapse z-axis
    print(f"\nCollapsing z-axis for {len(test_objects)} ND2 files")
    collapsed_images = batch_collapse_z_axis(test_objects, method='mean', verbose=False)
    
    # Segment images using config method
    print(f'Segmenting collapsed images with {segmentation_method.upper()} model (using channel {channel_index})')
    masks = bulk_segment_images(collapsed_images, channel_index=channel_index, method=segmentation_method)
    
    # Prepare image for visualization - extract the same channel used for segmentation
    first_image = collapsed_images[0]
    if first_image.ndim == 3:
        # Multi-channel image - extract the channel used for segmentation
        display_image = first_image[:, :, channel_index]
        print(f"Extracted channel {channel_index} for visualization: {display_image.shape}")
    else:
        # Single-channel image
        display_image = first_image
        print(f"Using single-channel image for visualization: {display_image.shape}")
    
    # Show results
    import numpy as np
    for i, mask in enumerate(masks):
        if mask is not None:
            num_objects = len(np.unique(mask)) - 1
            print(f"Image {i+1}: {num_objects} objects detected")
        else:
            print(f"Image {i+1}: Segmentation failed")
    
    print(f"\nBatch processing complete with {segmentation_method.upper()} method!")
    return collapsed_images, masks

if __name__ == "__main__":
    images, masks = test_batch_processor()
