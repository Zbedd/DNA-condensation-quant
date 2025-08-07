import numpy as np
import sys
import matplotlib.pyplot as plt
import time
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.preprocessor import collapse_z_axis
from dna_condensation.core.segmentation import segment_image

def compare_all_three_methods():
    """Compare YOLO, Watershed, and Otsu segmentation methods."""
    print("=== COMPARING ALL THREE SEGMENTATION METHODS ===\n")
    
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
    
    # Test all three methods
    methods = ['yolo', 'otsu', 'watershed']
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} ---")
        start_time = time.time()
        
        try:
            labels = segment_image(collapsed_image, channel_index=channel_index, 
                                 method=method, return_labels=True)
            binary = segment_image(collapsed_image, channel_index=channel_index, 
                                 method=method, return_labels=False)
            
            elapsed = time.time() - start_time
            num_objects = len(np.unique(labels)) - 1
            print(f"{method.upper()} detected {num_objects} objects ({elapsed:.1f}s)")
            
            # Calculate size statistics
            sizes = []
            for label_id in range(1, labels.max() + 1):
                size = np.sum(labels == label_id)
                sizes.append(size)
            
            results[method] = {
                'labels': labels,
                'binary': binary,
                'count': num_objects,
                'time': elapsed,
                'sizes': sizes,
                'mean_size': np.mean(sizes) if sizes else 0
            }
            
        except Exception as e:
            print(f"{method.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {
                'labels': np.zeros_like(display_channel, dtype=np.uint16),
                'binary': np.zeros_like(display_channel, dtype=np.uint8),
                'count': 0,
                'time': 0,
                'sizes': [],
                'mean_size': 0
            }
    
    # Create comparison visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Row 1: Original and labeled results
    axes[0, 0].imshow(display_channel, cmap='gray')
    axes[0, 0].set_title(f'Original (Channel {channel_index})')
    axes[0, 0].axis('off')
    
    # YOLO labels
    if results['yolo']['count'] > 0:
        im_yolo = axes[0, 1].imshow(results['yolo']['labels'], cmap='tab20', vmin=0, vmax=20)
        axes[0, 1].set_title(f'YOLO Labels\\n{results["yolo"]["count"]} objects ({results["yolo"]["time"]:.1f}s)')
    else:
        axes[0, 1].imshow(display_channel, cmap='gray')
        axes[0, 1].set_title('YOLO Failed')
    axes[0, 1].axis('off')
    
    # Otsu labels
    if results['otsu']['count'] > 0:
        im_otsu = axes[0, 2].imshow(results['otsu']['labels'], cmap='tab20', vmin=0, vmax=20)
        axes[0, 2].set_title(f'Otsu Labels\\n{results["otsu"]["count"]} objects ({results["otsu"]["time"]:.1f}s)')
    else:
        axes[0, 2].imshow(display_channel, cmap='gray')
        axes[0, 2].set_title('Otsu Failed')
    axes[0, 2].axis('off')
    
    # Row 2: Boundary overlays
    from scipy.ndimage import binary_erosion
    
    # YOLO boundaries
    if results['yolo']['count'] > 0:
        yolo_boundaries = np.zeros_like(results['yolo']['labels'], dtype=bool)
        for label_id in range(1, min(results['yolo']['labels'].max() + 1, 100)):  # Limit to avoid memory issues
            label_mask = results['yolo']['labels'] == label_id
            eroded = binary_erosion(label_mask)
            boundary = label_mask & ~eroded
            yolo_boundaries |= boundary
        
        yolo_overlay = np.stack([display_channel, display_channel, display_channel], axis=2)
        yolo_overlay[yolo_boundaries] = [255, 0, 0]  # Red boundaries
        axes[1, 0].imshow(yolo_overlay)
        axes[1, 0].set_title('YOLO Boundaries')
    else:
        axes[1, 0].imshow(display_channel, cmap='gray')
        axes[1, 0].set_title('YOLO Failed')
    axes[1, 0].axis('off')
    
    # Otsu boundaries
    if results['otsu']['count'] > 0:
        otsu_boundaries = np.zeros_like(results['otsu']['labels'], dtype=bool)
        for label_id in range(1, results['otsu']['labels'].max() + 1):
            label_mask = results['otsu']['labels'] == label_id
            eroded = binary_erosion(label_mask)
            boundary = label_mask & ~eroded
            otsu_boundaries |= boundary
        
        otsu_overlay = np.stack([display_channel, display_channel, display_channel], axis=2)
        otsu_overlay[otsu_boundaries] = [0, 255, 0]  # Green boundaries
        axes[1, 1].imshow(otsu_overlay)
        axes[1, 1].set_title('Otsu Boundaries')
    else:
        axes[1, 1].imshow(display_channel, cmap='gray')
        axes[1, 1].set_title('Otsu Failed')
    axes[1, 1].axis('off')
    
    # Watershed boundaries
    if results['watershed']['count'] > 0:
        watershed_boundaries = np.zeros_like(results['watershed']['labels'], dtype=bool)
        for label_id in range(1, results['watershed']['labels'].max() + 1):
            label_mask = results['watershed']['labels'] == label_id
            eroded = binary_erosion(label_mask)
            boundary = label_mask & ~eroded
            watershed_boundaries |= boundary
        
        watershed_overlay = np.stack([display_channel, display_channel, display_channel], axis=2)
        watershed_overlay[watershed_boundaries] = [0, 0, 255]  # Blue boundaries
        axes[1, 2].imshow(watershed_overlay)
        axes[1, 2].set_title('Watershed Boundaries')
    else:
        axes[1, 2].imshow(display_channel, cmap='gray')
        axes[1, 2].set_title('Watershed Failed')
    axes[1, 2].axis('off')
    
    # Row 3: Statistics and comparison
    # Object count comparison
    methods_list = ['YOLO', 'Otsu', 'Watershed']
    counts = [results['yolo']['count'], results['otsu']['count'], results['watershed']['count']]
    times = [results['yolo']['time'], results['otsu']['time'], results['watershed']['time']]
    
    axes[2, 0].bar(methods_list, counts, color=['red', 'green', 'blue'], alpha=0.7)
    axes[2, 0].set_ylabel('Number of Objects')
    axes[2, 0].set_title('Object Counts')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Processing time comparison
    axes[2, 1].bar(methods_list, times, color=['red', 'green', 'blue'], alpha=0.7)
    axes[2, 1].set_ylabel('Processing Time (seconds)')
    axes[2, 1].set_title('Processing Times')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Size distribution comparison
    all_sizes = []
    all_labels = []
    for method in ['yolo', 'otsu', 'watershed']:
        if results[method]['sizes']:
            # Limit to reasonable range for visualization
            limited_sizes = [s for s in results[method]['sizes'] if s < 2000]
            all_sizes.extend(limited_sizes)
            all_labels.extend([method.upper()] * len(limited_sizes))
    
    if all_sizes:
        # Create violin plot equivalent with histograms
        unique_methods = list(set(all_labels))
        for i, method in enumerate(unique_methods):
            method_sizes = [s for s, l in zip(all_sizes, all_labels) if l == method]
            if method_sizes:
                axes[2, 2].hist(method_sizes, bins=20, alpha=0.5, label=method, density=True)
        
        axes[2, 2].set_xlabel('Object Size (pixels)')
        axes[2, 2].set_ylabel('Density')
        axes[2, 2].set_title('Size Distributions')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
    else:
        axes[2, 2].text(0.5, 0.5, 'No size data available', ha='center', va='center', transform=axes[2, 2].transAxes)
    
    # Watershed labels in second row
    if results['watershed']['count'] > 0:
        # Move watershed to row 1, col 3 (overwriting one position)
        axes[0, 2].clear()
        im_watershed = axes[0, 2].imshow(results['watershed']['labels'], cmap='tab20', vmin=0, vmax=20)
        axes[0, 2].set_title(f'Watershed Labels\\n{results["watershed"]["count"]} objects ({results["watershed"]["time"]:.1f}s)')
        axes[0, 2].axis('off')
        
        # Show Otsu in row 1, col 2 and watershed in a different position
        axes[0, 1].clear()
        if results['otsu']['count'] > 0:
            im_otsu = axes[0, 1].imshow(results['otsu']['labels'], cmap='tab20', vmin=0, vmax=20)
            axes[0, 1].set_title(f'Otsu Labels\\n{results["otsu"]["count"]} objects ({results["otsu"]["time"]:.1f}s)')
        else:
            axes[0, 1].imshow(display_channel, cmap='gray')
            axes[0, 1].set_title('Otsu Failed')
        axes[0, 1].axis('off')
        
        # Put YOLO back in row 1, col 0 - wait, that's original. Let me reorganize:
        # Row 1: Original, Otsu, Watershed
        # Keep as is, this is fine
    
    plt.tight_layout()
    plt.savefig('three_method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved three-method comparison: three_method_comparison.png")
    
    # Summary statistics
    print(f"\n=== SUMMARY COMPARISON ===")
    print(f"{'Method':<10} {'Objects':<8} {'Time':<6} {'Mean Size':<10}")
    print("-" * 40)
    for method in ['yolo', 'otsu', 'watershed']:
        result = results[method]
        print(f"{method.upper():<10} {result['count']:<8} {result['time']:.1f}s{'':<3} {result['mean_size']:.1f}")
    
    print(f"\n--- Method Relationships ---")
    if results['otsu']['count'] > 0 and results['watershed']['count'] > 0:
        ratio = results['watershed']['count'] / results['otsu']['count']
        print(f"Watershed/Otsu ratio: {ratio:.2f}")
        if ratio < 0.5:
            print("→ Watershed is much more conservative than Otsu")
        elif ratio < 0.8:
            print("→ Watershed is moderately more conservative than Otsu")
        elif ratio > 1.2:
            print("→ Watershed detects more objects than Otsu")
        else:
            print("→ Watershed and Otsu are well-matched")
    
    return results

if __name__ == "__main__":
    results = compare_all_three_methods()
