import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.z_stack_handling import collapse_z_axis
from dna_condensation.core.segmentation import segment_image, filter_labels_by_size

def test_size_filtering():
    """Test the size filtering functionality."""
    print("=== TESTING SIZE FILTERING FUNCTIONALITY ===\n")
    
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
    
    # Test with Otsu method (fast and good object counts)
    print(f"\n--- Testing Size Filtering with Otsu Method ---")
    
    # Get original segmentation
    original_labels = segment_image(collapsed_image, channel_index=channel_index, 
                                   method='otsu', return_labels=True)
    original_count = len(np.unique(original_labels)) - 1
    
    print(f"Original segmentation: {original_count} objects")
    
    # Calculate original size statistics
    original_sizes = []
    for label_id in range(1, original_labels.max() + 1):
        size = np.sum(original_labels == label_id)
        original_sizes.append(size)
    
    if original_sizes:
        original_median = np.median(original_sizes)
        print(f"Original size statistics:")
        print(f"  Median: {original_median:.1f} pixels")
        print(f"  Range: {np.min(original_sizes)} - {np.max(original_sizes)} pixels")
        print(f"  Mean: {np.mean(original_sizes):.1f} ± {np.std(original_sizes):.1f}")
    
    # Test different filtering percentages
    filter_percentages = [5, 10, 20, 30]
    results = {}
    
    for percentage in filter_percentages:
        print(f"\n--- Testing {percentage}% size filter ---")
        filtered_labels = filter_labels_by_size(original_labels, min_size_percentage=percentage, verbose=True)
        filtered_count = len(np.unique(filtered_labels)) - 1
        results[percentage] = {
            'labels': filtered_labels,
            'count': filtered_count,
            'removal_rate': 100 * (original_count - filtered_count) / original_count if original_count > 0 else 0
        }
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image and segmentation
    axes[0, 0].imshow(display_channel, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(original_labels, cmap='tab20')
    axes[0, 1].set_title(f'Original Labels\\n{original_count} objects')
    axes[0, 1].axis('off')
    
    # Show 10% filtering result
    if 10 in results:
        axes[0, 2].imshow(results[10]['labels'], cmap='tab20')
        axes[0, 2].set_title(f'10% Size Filter\\n{results[10]["count"]} objects')
        axes[0, 2].axis('off')
    
    # Show 20% filtering result
    if 20 in results:
        axes[1, 0].imshow(results[20]['labels'], cmap='tab20')
        axes[1, 0].set_title(f'20% Size Filter\\n{results[20]["count"]} objects')
        axes[1, 0].axis('off')
    
    # Size distribution histogram
    if original_sizes:
        axes[1, 1].hist(original_sizes, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].axvline(original_median, color='red', linestyle='--', linewidth=2, label=f'Median: {original_median:.0f}')
        
        # Show filter thresholds
        for percentage in [10, 20]:
            threshold = original_median * (percentage / 100.0)
            axes[1, 1].axvline(threshold, color='orange', linestyle=':', alpha=0.7, 
                              label=f'{percentage}% threshold: {threshold:.0f}')
        
        axes[1, 1].set_xlabel('Object Size (pixels)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Size Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Filtering effectiveness plot
    percentages_list = list(results.keys())
    counts = [results[p]['count'] for p in percentages_list]
    removal_rates = [results[p]['removal_rate'] for p in percentages_list]
    
    ax_twin = axes[1, 2].twinx()
    
    line1 = axes[1, 2].plot(percentages_list, counts, 'bo-', linewidth=2, markersize=8, label='Objects Remaining')
    axes[1, 2].set_xlabel('Filter Percentage (%)')
    axes[1, 2].set_ylabel('Objects Remaining', color='blue')
    axes[1, 2].tick_params(axis='y', labelcolor='blue')
    
    line2 = ax_twin.plot(percentages_list, removal_rates, 'ro-', linewidth=2, markersize=8, label='% Removed')
    ax_twin.set_ylabel('Objects Removed (%)', color='red')
    ax_twin.tick_params(axis='y', labelcolor='red')
    
    axes[1, 2].set_title('Filter Effectiveness')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add legend for both lines
    lines1, labels1 = axes[1, 2].get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    axes[1, 2].legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    plt.savefig('size_filtering_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved size filtering test: size_filtering_test.png")
    
    # Summary
    print(f"\n=== SIZE FILTERING SUMMARY ===")
    print(f"Original objects: {original_count}")
    for percentage in filter_percentages:
        result = results[percentage]
        print(f"{percentage:2d}% filter: {result['count']:3d} objects ({result['removal_rate']:5.1f}% removed)")
    
    # Recommendations
    print(f"\n--- Recommendations ---")
    if original_count > 50:
        print("• Consider 10-20% filtering to remove small debris")
    elif original_count > 20:
        print("• Consider 5-10% filtering for conservative cleanup")
    else:
        print("• Use minimal filtering (5%) to preserve objects")
    
    return results

if __name__ == "__main__":
    results = test_size_filtering()
