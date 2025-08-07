import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.preprocessor import collapse_z_axis
from dna_condensation.core.segmentation import segment_image

def test_refined_otsu():
    """Test the refined Otsu method against the threshold debug baseline."""
    print("=== TESTING REFINED OTSU METHOD ===\n")
    
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
    
    if collapsed_image.ndim == 3:
        display_channel = collapsed_image[:, :, channel_index]
    else:
        display_channel = collapsed_image
    
    print(f"Image shape: {display_channel.shape}")
    print(f"Image range: {display_channel.min()} - {display_channel.max()}")
    
    # Reproduce the threshold debug baseline
    print(f"\n--- Threshold Debug Baseline ---")
    from skimage import filters
    
    otsu_thresh = filters.threshold_otsu(display_channel)
    moderate_thresh = otsu_thresh * 1.2
    debug_binary = display_channel > moderate_thresh
    debug_pixels = np.sum(debug_binary)
    
    print(f"Otsu threshold: {otsu_thresh}")
    print(f"1.2x Otsu threshold: {moderate_thresh}")
    print(f"Debug binary pixels: {debug_pixels} ({100*debug_pixels/debug_binary.size:.1f}%)")
    
    # Test refined Otsu method
    print(f"\n--- Refined Otsu Method ---")
    try:
        otsu_labels = segment_image(collapsed_image, channel_index=channel_index, 
                                   method='otsu', return_labels=True)
        otsu_binary = segment_image(collapsed_image, channel_index=channel_index, 
                                   method='otsu', return_labels=False)
        
        num_objects = len(np.unique(otsu_labels)) - 1
        method_pixels = np.sum(otsu_binary > 0)
        
        print(f"Otsu method detected: {num_objects} objects")
        print(f"Method binary pixels: {method_pixels} ({100*method_pixels/otsu_binary.size:.1f}%)")
        
        # Calculate pixel agreement
        if debug_pixels > 0 and method_pixels > 0:
            overlap = np.sum((debug_binary > 0) & (otsu_binary > 0))
            agreement = overlap / max(debug_pixels, method_pixels)
            print(f"Pixel overlap: {overlap} ({100*agreement:.1f}% agreement)")
        
        # Visualize comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Original and binary masks
        axes[0, 0].imshow(display_channel, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(debug_binary, cmap='gray')
        axes[0, 1].set_title(f'Debug Binary\\n{debug_pixels} pixels')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(otsu_binary, cmap='gray')
        axes[0, 2].set_title(f'Method Binary\\n{method_pixels} pixels')
        axes[0, 2].axis('off')
        
        # Row 2: Labels and comparison
        axes[1, 0].imshow(otsu_labels, cmap='tab20')
        axes[1, 0].set_title(f'Method Labels\\n{num_objects} objects')
        axes[1, 0].axis('off')
        
        # Difference visualization
        difference = np.zeros_like(display_channel)
        difference[debug_binary & ~(otsu_binary > 0)] = 1  # In debug but not method (red)
        difference[~debug_binary & (otsu_binary > 0)] = 2  # In method but not debug (blue)
        difference[debug_binary & (otsu_binary > 0)] = 3   # In both (green)
        
        colors = ['black', 'red', 'blue', 'green']
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        axes[1, 1].imshow(difference, cmap=cmap, vmin=0, vmax=3)
        axes[1, 1].set_title('Comparison\\nRed=Debug only, Blue=Method only, Green=Both')
        axes[1, 1].axis('off')
        
        # Overlay on original
        overlay = np.stack([display_channel, display_channel, display_channel], axis=2)
        # Add boundaries for method objects
        from scipy.ndimage import binary_erosion
        
        method_boundaries = np.zeros_like(otsu_labels, dtype=bool)
        for label_id in range(1, otsu_labels.max() + 1):
            label_mask = otsu_labels == label_id
            eroded = binary_erosion(label_mask)
            boundary = label_mask & ~eroded
            method_boundaries |= boundary
        
        overlay[method_boundaries] = [0, 255, 0]  # Green boundaries
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Method Boundaries')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('refined_otsu_test.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\\nSaved comparison: refined_otsu_test.png")
        
        # Analysis
        if method_pixels > debug_pixels * 0.8 and method_pixels < debug_pixels * 1.2:
            print("✓ Pixel count matches debug baseline well")
        elif method_pixels < debug_pixels * 0.5:
            print("⚠ Method is too conservative (too few pixels)")
        elif method_pixels > debug_pixels * 2:
            print("⚠ Method is too permissive (too many pixels)")
        else:
            print("~ Method has moderate difference from baseline")
            
        return otsu_labels, otsu_binary, debug_binary
        
    except Exception as e:
        print(f"Refined Otsu method failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    labels, method_binary, debug_binary = test_refined_otsu()
