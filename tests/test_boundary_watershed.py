import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.preprocessor import collapse_z_axis
from dna_condensation.core.segmentation import segment_image

def test_boundary_respecting_watershed():
    """Test the boundary-respecting watershed segmentation."""
    print("=== TESTING BOUNDARY-RESPECTING WATERSHED ===\n")
    
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
    
    # Test boundary-respecting watershed
    print(f"\n--- Testing Boundary-Respecting Watershed ---")
    try:
        watershed_labels = segment_image(collapsed_image, channel_index=channel_index, 
                                       method='watershed', return_labels=True)
        watershed_binary = segment_image(collapsed_image, channel_index=channel_index, 
                                       method='watershed', return_labels=False)
        
        num_objects = len(np.unique(watershed_labels)) - 1
        print(f"Watershed detected {num_objects} objects")
        
        # Analyze boundary quality
        if collapsed_image.ndim == 3:
            display_channel = collapsed_image[:, :, channel_index]
        else:
            display_channel = collapsed_image
        
        # Create boundary visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # Original image
        axes[0, 0].imshow(display_channel, cmap='gray')
        axes[0, 0].set_title(f'Original (Channel {channel_index})')
        axes[0, 0].axis('off')
        
        # Watershed labels with boundaries
        # Create boundary overlay
        from scipy.ndimage import binary_erosion
        boundaries = np.zeros_like(watershed_labels, dtype=bool)
        
        for label_id in range(1, watershed_labels.max() + 1):
            label_mask = watershed_labels == label_id
            eroded = binary_erosion(label_mask)
            boundary = label_mask & ~eroded
            boundaries |= boundary
        
        overlay = np.stack([display_channel, display_channel, display_channel], axis=2)
        overlay[boundaries] = [255, 0, 0]  # Red boundaries
        
        axes[0, 1].imshow(overlay)
        axes[0, 1].set_title(f'Boundaries ({num_objects} nuclei)')
        axes[0, 1].axis('off')
        
        # Individual labels
        im = axes[1, 0].imshow(watershed_labels, cmap='tab20')
        axes[1, 0].set_title('Individual Labels')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Analyze separation - check for merged objects
        # Look at nucleus sizes and shapes
        sizes = []
        circularities = []
        
        for label_id in range(1, watershed_labels.max() + 1):
            mask = watershed_labels == label_id
            size = np.sum(mask)
            sizes.append(size)
            
            # Calculate circularity (4π*area / perimeter²)
            import cv2
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    circularity = 4 * np.pi * size / (perimeter ** 2)
                    circularities.append(circularity)
        
        axes[1, 1].scatter(sizes, circularities, alpha=0.7, s=50)
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Good circularity')
        axes[1, 1].set_xlabel('Nucleus Size (pixels)')
        axes[1, 1].set_ylabel('Circularity (1.0 = perfect circle)')
        axes[1, 1].set_title('Size vs Shape Analysis')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('boundary_respecting_watershed.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved analysis: boundary_respecting_watershed.png")
        print(f"Size statistics:")
        print(f"  Mean: {np.mean(sizes):.1f} pixels")
        print(f"  Std: {np.std(sizes):.1f} pixels")
        print(f"  Range: {np.min(sizes)} - {np.max(sizes)} pixels")
        
        if circularities:
            print(f"Shape statistics:")
            print(f"  Mean circularity: {np.mean(circularities):.3f}")
            print(f"  Good shapes (>0.5): {sum(1 for c in circularities if c > 0.5)}/{len(circularities)}")
        
        # Check for potential merging by looking for very large objects
        large_objects = [s for s in sizes if s > np.mean(sizes) + 2*np.std(sizes)]
        if large_objects:
            print(f"Warning: {len(large_objects)} potentially merged objects detected")
            print(f"  Large sizes: {large_objects}")
        else:
            print("✓ No obviously merged objects detected")
            
        return watershed_labels, watershed_binary
        
    except Exception as e:
        print(f"Boundary-respecting watershed failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    labels, binary = test_boundary_respecting_watershed()
