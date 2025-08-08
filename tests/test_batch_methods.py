import numpy as np
import sys
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.z_stack_handling import batch_collapse_z_axis
from dna_condensation.core.segmentation import bulk_segment_images

def test_batch_with_methods():
    """Test batch processing with different segmentation methods."""
    print("=== TESTING BATCH PROCESSING WITH DIFFERENT METHODS ===\n")
    
    # Load config
    config = Config()
    nd2_folder_path = config.get("raw_nd2_path")
    channel_index = config.get("segmentation_channel_index", 0)
    method = config.get("segmentation_method", "yolo")
    
    print(f"Config method: {method}")
    print(f"Config channel: {channel_index}")
    
    # Get first 2 ND2 files for testing
    nd2_objects = get_nd2_objects(nd2_folder_path)
    test_objects = nd2_objects[:2]  # Test with just first 2
    
    print(f"\nTesting with {len(test_objects)} files:")
    for i, obj in enumerate(test_objects):
        print(f"  {i+1}. {obj.filename}")
    
    # Collapse z-axis
    collapsed_images = batch_collapse_z_axis(test_objects, method='mean', verbose=False)
    print(f"\nCollapsed {len([img for img in collapsed_images if img is not None])}/{len(collapsed_images)} images")
    
    # Test YOLO method
    print(f"\n--- Testing YOLO Method ---")
    yolo_masks = bulk_segment_images(collapsed_images, channel_index=channel_index, method='yolo', verbose=True, return_labels=True)
    
    # Test Watershed method
    print(f"\n--- Testing Watershed Method ---")
    watershed_masks = bulk_segment_images(collapsed_images, channel_index=channel_index, method='watershed', verbose=True, return_labels=True)
    
    # Compare results
    print(f"\n--- Comparison Results ---")
    for i, (yolo_mask, watershed_mask) in enumerate(zip(yolo_masks, watershed_masks)):
        if yolo_mask is not None and watershed_mask is not None:
            yolo_objects = len(np.unique(yolo_mask)) - 1
            watershed_objects = len(np.unique(watershed_mask)) - 1
            print(f"Image {i+1}: YOLO={yolo_objects}, Watershed={watershed_objects}")
        else:
            print(f"Image {i+1}: Failed")
    
    return yolo_masks, watershed_masks

if __name__ == "__main__":
    yolo_results, watershed_results = test_batch_with_methods()
