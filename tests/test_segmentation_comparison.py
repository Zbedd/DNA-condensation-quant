import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.z_stack_handling import collapse_z_axis
from dna_condensation.core.segmentation import segment_image_yolo, segment_image_watershed, segment_image

def compare_segmentation_methods():
    """Compare YOLO and watershed segmentation methods."""
    print("=== COMPARING YOLO vs WATERSHED SEGMENTATION ===\n")
    
    # Load config and test image
    config = Config()
    nd2_folder_path = config.get("raw_nd2_path")
    channel_index = config.get("segmentation_channel_index", 0)
    
    # Get first ND2 file
    nd2_objects = get_nd2_objects(nd2_folder_path)
    first_nd2 = nd2_objects[0]
    print(f"Testing with: {first_nd2.filename}")
    
    # Preprocess image
    collapsed_image = collapse_z_axis(first_nd2, method='mean', verbose=False)
    print(f"Image shape: {collapsed_image.shape}")
    
    if collapsed_image.ndim == 3:
        test_channel = collapsed_image[:, :, channel_index]
        test_image = np.stack([test_channel, test_channel, test_channel], axis=2)  # Make it 3-channel for testing
    else:
        test_image = collapsed_image
    
    print(f"Test image shape: {test_image.shape}")
    
    # Test both methods
    print(f"\n--- Testing YOLO Segmentation ---")
    try:
        yolo_mask = segment_image(test_image, channel_index=channel_index, method='yolo', return_labels=True)
        yolo_objects = len(np.unique(yolo_mask)) - 1  # Subtract background
        print(f"YOLO detected {yolo_objects} objects")
        print(f"YOLO mask shape: {yolo_mask.shape}, dtype: {yolo_mask.dtype}")
    except Exception as e:
        print(f"YOLO segmentation failed: {e}")
        yolo_mask = None
        yolo_objects = 0
    
    print(f"\n--- Testing Watershed Segmentation ---")
    try:
        watershed_mask = segment_image(test_image, channel_index=channel_index, method='watershed', return_labels=True)
        watershed_objects = len(np.unique(watershed_mask)) - 1  # Subtract background
        print(f"Watershed detected {watershed_objects} objects")
        print(f"Watershed mask shape: {watershed_mask.shape}, dtype: {watershed_mask.dtype}")
    except Exception as e:
        print(f"Watershed segmentation failed: {e}")
        watershed_mask = None
        watershed_objects = 0
    
    # Create comparison visualization
    if yolo_mask is not None and watershed_mask is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(test_image[:, :, channel_index], cmap='gray')
        axes[0, 0].set_title(f'Original Image (Channel {channel_index})')
        axes[0, 0].axis('off')
        
        # YOLO results
        axes[0, 1].imshow(yolo_mask, cmap='tab10')
        axes[0, 1].set_title(f'YOLO Segmentation ({yolo_objects} objects)')
        axes[0, 1].axis('off')
        
        # YOLO binary
        yolo_binary = segment_image(test_image, channel_index=channel_index, method='yolo', return_labels=False)
        axes[0, 2].imshow(yolo_binary, cmap='Reds', alpha=0.8)
        axes[0, 2].set_title('YOLO Binary Mask')
        axes[0, 2].axis('off')
        
        # Watershed results
        axes[1, 0].imshow(test_image[:, :, channel_index], cmap='gray')
        axes[1, 0].set_title(f'Original Image (Channel {channel_index})')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(watershed_mask, cmap='tab10')
        axes[1, 1].set_title(f'Watershed Segmentation ({watershed_objects} objects)')
        axes[1, 1].axis('off')
        
        # Watershed binary
        watershed_binary = segment_image(test_image, channel_index=channel_index, method='watershed', return_labels=False)
        axes[1, 2].imshow(watershed_binary, cmap='Blues', alpha=0.8)
        axes[1, 2].set_title('Watershed Binary Mask')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('segmentation_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved comparison image: segmentation_comparison.png")
        print(f"YOLO vs Watershed: {yolo_objects} vs {watershed_objects} objects detected")
    
    return yolo_mask, watershed_mask

if __name__ == "__main__":
    yolo_result, watershed_result = compare_segmentation_methods()
