import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.preprocessor import collapse_z_axis
from dna_condensation.core.segmentation import segment_image

def compare_segmentation_methods():
    """Compare YOLO and watershed segmentation side by side."""
    print("=== COMPARING SEGMENTATION METHODS ===\n")
    
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
    
    # Extract display channel
    if collapsed_image.ndim == 3:
        display_channel = collapsed_image[:, :, channel_index]
    else:
        display_channel = collapsed_image
    
    # Test both methods
    methods = ['yolo', 'watershed']
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} ---")
        try:
            labels = segment_image(collapsed_image, channel_index=channel_index, 
                                 method=method, return_labels=True)
            binary = segment_image(collapsed_image, channel_index=channel_index, 
                                 method=method, return_labels=False)
            
            num_objects = len(np.unique(labels)) - 1
            print(f"{method.upper()} detected {num_objects} objects")
            
            results[method] = {
                'labels': labels,
                'binary': binary,
                'count': num_objects
            }
            
        except Exception as e:
            print(f"{method.upper()} failed: {e}")
            results[method] = {
                'labels': np.zeros_like(display_channel, dtype=np.uint16),
                'binary': np.zeros_like(display_channel, dtype=np.uint8),
                'count': 0
            }
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image (top row)
    axes[0, 0].imshow(display_channel, cmap='gray')
    axes[0, 0].set_title(f'Original (Channel {channel_index})')
    axes[0, 0].axis('off')
    
    # YOLO results
    if results['yolo']['count'] > 0:
        im_yolo = axes[0, 1].imshow(results['yolo']['labels'], cmap='tab20', vmin=0, vmax=20)
        axes[0, 1].set_title(f'YOLO Labels ({results["yolo"]["count"]} nuclei)')
        axes[0, 1].axis('off')
        
        # Create YOLO boundary overlay
        from scipy.ndimage import binary_erosion
        yolo_boundaries = np.zeros_like(results['yolo']['labels'], dtype=bool)
        for label_id in range(1, results['yolo']['labels'].max() + 1):
            label_mask = results['yolo']['labels'] == label_id
            eroded = binary_erosion(label_mask)
            boundary = label_mask & ~eroded
            yolo_boundaries |= boundary
        
        yolo_overlay = np.stack([display_channel, display_channel, display_channel], axis=2)
        yolo_overlay[yolo_boundaries] = [255, 0, 0]  # Red boundaries
        axes[1, 1].imshow(yolo_overlay)
        axes[1, 1].set_title('YOLO Boundaries')
        axes[1, 1].axis('off')
    else:
        axes[0, 1].imshow(display_channel, cmap='gray')
        axes[0, 1].set_title('YOLO Failed')
        axes[0, 1].axis('off')
        axes[1, 1].imshow(display_channel, cmap='gray')
        axes[1, 1].set_title('YOLO Failed')
        axes[1, 1].axis('off')
    
    # Watershed results
    if results['watershed']['count'] > 0:
        im_watershed = axes[0, 2].imshow(results['watershed']['labels'], cmap='tab20', vmin=0, vmax=20)
        axes[0, 2].set_title(f'Watershed Labels ({results["watershed"]["count"]} nuclei)')
        axes[0, 2].axis('off')
        
        # Create watershed boundary overlay
        watershed_boundaries = np.zeros_like(results['watershed']['labels'], dtype=bool)
        for label_id in range(1, results['watershed']['labels'].max() + 1):
            label_mask = results['watershed']['labels'] == label_id
            eroded = binary_erosion(label_mask)
            boundary = label_mask & ~eroded
            watershed_boundaries |= boundary
        
        watershed_overlay = np.stack([display_channel, display_channel, display_channel], axis=2)
        watershed_overlay[watershed_boundaries] = [0, 255, 0]  # Green boundaries
        axes[1, 2].imshow(watershed_overlay)
        axes[1, 2].set_title('Watershed Boundaries')
        axes[1, 2].axis('off')
    else:
        axes[0, 2].imshow(display_channel, cmap='gray')
        axes[0, 2].set_title('Watershed Failed')
        axes[0, 2].axis('off')
        axes[1, 2].imshow(display_channel, cmap='gray')
        axes[1, 2].set_title('Watershed Failed')
        axes[1, 2].axis('off')
    
    # Combined overlay (bottom left)
    if results['yolo']['count'] > 0 and results['watershed']['count'] > 0:
        combined_overlay = np.stack([display_channel, display_channel, display_channel], axis=2)
        combined_overlay[yolo_boundaries] = [255, 0, 0]  # Red for YOLO
        combined_overlay[watershed_boundaries] = [0, 255, 0]  # Green for watershed
        # Purple where they overlap
        overlap = yolo_boundaries & watershed_boundaries
        combined_overlay[overlap] = [255, 0, 255]  # Purple for overlap
        axes[1, 0].imshow(combined_overlay)
        axes[1, 0].set_title('Combined (Red=YOLO, Green=Watershed)')
        axes[1, 0].axis('off')
    else:
        axes[1, 0].imshow(display_channel, cmap='gray')
        axes[1, 0].set_title('Comparison Not Available')
        axes[1, 0].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comparison: segmentation_comparison.png")
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"YOLO:      {results['yolo']['count']:3d} objects")
    print(f"Watershed: {results['watershed']['count']:3d} objects")
    
    if results['yolo']['count'] > 0 and results['watershed']['count'] > 0:
        # Calculate size statistics for each method
        for method in ['yolo', 'watershed']:
            labels = results[method]['labels']
            sizes = []
            for label_id in range(1, labels.max() + 1):
                size = np.sum(labels == label_id)
                sizes.append(size)
            
            if sizes:
                print(f"\n{method.upper()} size statistics:")
                print(f"  Mean size: {np.mean(sizes):.1f} pixels")
                print(f"  Size range: {np.min(sizes)} - {np.max(sizes)} pixels")
    
    return results

if __name__ == "__main__":
    results = compare_segmentation_methods()
