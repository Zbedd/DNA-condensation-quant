import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import filters
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.z_stack_handling import collapse_z_axis

def debug_watershed_thresholds():
    """Debug watershed thresholds to see what went wrong."""
    print("=== DEBUGGING WATERSHED THRESHOLDS ===\n")
    
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
        image = collapsed_image[:, :, channel_index]
    else:
        image = collapsed_image
    
    print(f"Image shape: {image.shape}")
    print(f"Image range: {image.min()} - {image.max()}")
    print(f"Image dtype: {image.dtype}")
    
    # Test different threshold methods
    print("\n--- Threshold Analysis ---")
    
    # Otsu threshold
    otsu_thresh = filters.threshold_otsu(image)
    print(f"Otsu threshold: {otsu_thresh}")
    
    # Conservative (2x Otsu)
    conservative_thresh = 2 * otsu_thresh
    print(f"Conservative threshold (2x): {conservative_thresh}")
    
    # Check how many pixels pass each threshold
    otsu_pixels = np.sum(image > otsu_thresh)
    conservative_pixels = np.sum(image > conservative_thresh)
    
    print(f"Pixels above Otsu: {otsu_pixels} ({100*otsu_pixels/image.size:.1f}%)")
    print(f"Pixels above Conservative: {conservative_pixels} ({100*conservative_pixels/image.size:.1f}%)")
    
    # Test with different multipliers
    print(f"\n--- Testing Different Multipliers ---")
    for mult in [0.8, 1.0, 1.2, 1.5, 2.0]:
        thresh = mult * otsu_thresh
        pixels = np.sum(image > thresh)
        percentage = 100 * pixels / image.size
        print(f"  {mult}x Otsu ({thresh:.1f}): {pixels} pixels ({percentage:.1f}%)")
    
    # Visualize thresholds
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Histogram with thresholds
    axes[0, 1].hist(image.ravel(), bins=100, alpha=0.7, color='gray')
    axes[0, 1].axvline(otsu_thresh, color='red', linestyle='--', label=f'Otsu: {otsu_thresh:.1f}')
    axes[0, 1].axvline(conservative_thresh, color='orange', linestyle='--', label=f'2x Otsu: {conservative_thresh:.1f}')
    axes[0, 1].set_xlabel('Pixel Intensity')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Intensity Histogram')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Otsu binary
    otsu_binary = image > otsu_thresh
    axes[0, 2].imshow(otsu_binary, cmap='gray')
    axes[0, 2].set_title(f'Otsu Binary ({otsu_pixels} pixels)')
    axes[0, 2].axis('off')
    
    # Conservative binary
    conservative_binary = image > conservative_thresh
    axes[1, 0].imshow(conservative_binary, cmap='gray')
    axes[1, 0].set_title(f'Conservative Binary ({conservative_pixels} pixels)')
    axes[1, 0].axis('off')
    
    # Try 1.2x Otsu as middle ground
    middle_thresh = 1.2 * otsu_thresh
    middle_binary = image > middle_thresh
    middle_pixels = np.sum(middle_binary)
    axes[1, 1].imshow(middle_binary, cmap='gray')
    axes[1, 1].set_title(f'1.2x Otsu Binary ({middle_pixels} pixels)')
    axes[1, 1].axis('off')
    
    # Try adaptive threshold comparison
    try:
        from skimage.filters import threshold_local
        local_thresh = threshold_local(image, block_size=101, offset=0.01)
        local_binary = image > local_thresh
        local_pixels = np.sum(local_binary)
        axes[1, 2].imshow(local_binary, cmap='gray')
        axes[1, 2].set_title(f'Local Adaptive ({local_pixels} pixels)')
        axes[1, 2].axis('off')
    except:
        axes[1, 2].text(0.5, 0.5, 'Local threshold failed', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('threshold_debug.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved threshold analysis: threshold_debug.png")
    
    # Recommend new threshold
    if conservative_pixels < 100:  # Too few pixels
        if middle_pixels > 1000:
            print(f"\nRecommendation: Use 1.2x Otsu threshold ({middle_thresh:.1f})")
        else:
            print(f"\nRecommendation: Use standard Otsu threshold ({otsu_thresh:.1f})")
    else:
        print(f"\nRecommendation: Conservative threshold seems reasonable")

if __name__ == "__main__":
    debug_watershed_thresholds()
