import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.preprocessor import collapse_z_axis
from dna_condensation.core.segmentation import segment_image

def test_improved_watershed():
    """Test the improved watershed segmentation."""
    print("=== TESTING IMPROVED WATERSHED SEGMENTATION ===\n")
    
    # Load config and test image
    config = Config()
    nd2_folder_path = config.get("raw_nd2_path")
    channel_index = config.get("segmentation_channel_index", 1)
    
    # Get first ND2 file
    nd2_objects = get_nd2_objects(nd2_folder_path)
    first_nd2 = nd2_objects[0]
    print(f"Testing with: {first_nd2.filename}")
    
    # Preprocess image
    collapsed_image = collapse_z_axis(first_nd2, method='mean', verbose=False)
    print(f"Image shape: {collapsed_image.shape}")
    print(f"Using channel {channel_index} for segmentation")
    
    # Test improved watershed method
    print(f"\n--- Testing Improved Watershed Segmentation ---")
    try:
        # Test with labeled output
        watershed_labels = segment_image(collapsed_image, channel_index=channel_index, 
                                       method='watershed', return_labels=True)
        watershed_binary = segment_image(collapsed_image, channel_index=channel_index, 
                                       method='watershed', return_labels=False)
        
        num_objects = len(np.unique(watershed_labels)) - 1  # Subtract background
        print(f"Watershed detected {num_objects} objects")
        print(f"Label mask shape: {watershed_labels.shape}, dtype: {watershed_labels.dtype}")
        print(f"Binary mask shape: {watershed_binary.shape}, dtype: {watershed_binary.dtype}")
        
        # Create detailed visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract the segmentation channel for display
        if collapsed_image.ndim == 3:
            display_channel = collapsed_image[:, :, channel_index]
        else:
            display_channel = collapsed_image
            
        # Row 1: Original and results
        axes[0, 0].imshow(display_channel, cmap='gray')
        axes[0, 0].set_title(f'Original Image (Channel {channel_index})')
        axes[0, 0].axis('off')
        
        # Labeled watershed result
        im1 = axes[0, 1].imshow(watershed_labels, cmap='tab20')
        axes[0, 1].set_title(f'Watershed Labels ({num_objects} nuclei)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Binary mask
        axes[0, 2].imshow(watershed_binary, cmap='Reds', alpha=0.8)
        axes[0, 2].set_title('Binary Mask')
        axes[0, 2].axis('off')
        
        # Row 2: Overlay and analysis
        # Overlay on original
        overlay = np.stack([display_channel, display_channel, display_channel], axis=2)
        # Add colored boundaries
        boundaries = watershed_labels > 0
        overlay[boundaries] = [255, 100, 100]  # Red boundaries
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Overlay (Red = Detected Nuclei)')
        axes[1, 0].axis('off')
        
        # Show individual nuclei (first 20)
        unique_labels = np.unique(watershed_labels)[1:]  # Skip background
        sample_labels = unique_labels[:20] if len(unique_labels) > 20 else unique_labels
        
        sample_mask = np.zeros_like(watershed_labels)
        for label_id in sample_labels:
            sample_mask[watershed_labels == label_id] = label_id
            
        im2 = axes[1, 1].imshow(sample_mask, cmap='tab20')
        axes[1, 1].set_title(f'Sample Nuclei (first {len(sample_labels)})')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Size distribution
        sizes = []
        for label_id in unique_labels:
            size = np.sum(watershed_labels == label_id)
            sizes.append(size)
            
        axes[1, 2].hist(sizes, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 2].set_xlabel('Nucleus Size (pixels)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title(f'Size Distribution\nMean: {np.mean(sizes):.0f} pixels')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('improved_watershed_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved detailed analysis: improved_watershed_analysis.png")
        print(f"Nucleus size statistics:")
        print(f"  Mean: {np.mean(sizes):.1f} pixels")
        print(f"  Std: {np.std(sizes):.1f} pixels")
        print(f"  Min: {np.min(sizes)} pixels")
        print(f"  Max: {np.max(sizes)} pixels")
        
        return watershed_labels, watershed_binary
        
    except Exception as e:
        print(f"Improved watershed segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    labels, binary = test_improved_watershed()
