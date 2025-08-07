import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.preprocessor import collapse_z_axis
from dna_condensation.core.segmentation import bulk_segment_images

def test_batch_with_size_filtering():
    """Test batch processing with different size filtering configurations."""
    print("=== TESTING BATCH PROCESSING WITH SIZE FILTERING ===\n")
    
    # Load config and test images
    config = Config()
    nd2_folder_path = config.get("raw_nd2_path")
    channel_index = config.get("segmentation_channel_index", 1)
    
    # Get first 3 ND2 files for quick testing
    nd2_objects = get_nd2_objects(nd2_folder_path)
    test_files = nd2_objects[:3]
    
    print(f"Testing with {len(test_files)} files:")
    for i, nd2 in enumerate(test_files):
        print(f"  {i+1}. {nd2.filename.split('/')[-1]}")
    
    # Preprocess images
    print(f"\nCollapsing z-axis...")
    collapsed_images = []
    for nd2 in test_files:
        img = None
        try:
            from dna_condensation.core.preprocessor import collapse_z_axis
            img = collapse_z_axis(nd2, method='mean', verbose=False)
            collapsed_images.append(img)
        except Exception as e:
            print(f"Failed to process {nd2.filename}: {e}")
            collapsed_images.append(None)
    
    # Test different filtering configurations
    filter_configs = [
        {"enabled": False, "min_size_percentage": 0},      # No filtering
        {"enabled": True, "min_size_percentage": 5},       # Very permissive
        {"enabled": True, "min_size_percentage": 10},      # Default
        {"enabled": True, "min_size_percentage": 20},      # More aggressive
    ]
    
    results = {}
    
    for i, filter_config in enumerate(filter_configs):
        filter_name = f"No filter" if not filter_config["enabled"] else f"{filter_config['min_size_percentage']}% filter"
        print(f"\n--- Testing {filter_name} ---")
        
        try:
            masks = bulk_segment_images(
                collapsed_images, 
                channel_index=channel_index, 
                method='otsu', 
                return_labels=True,
                size_filter_config=filter_config,
                verbose=True
            )
            
            # Calculate statistics
            object_counts = []
            for mask in masks:
                if mask is not None:
                    count = len(np.unique(mask)) - 1
                    object_counts.append(count)
                else:
                    object_counts.append(0)
            
            results[filter_name] = {
                'config': filter_config,
                'masks': masks,
                'counts': object_counts,
                'total_objects': sum(object_counts),
                'mean_objects': np.mean(object_counts) if object_counts else 0
            }
            
            print(f"Results: {object_counts} objects per image (total: {sum(object_counts)}, mean: {np.mean(object_counts):.1f})")
            
        except Exception as e:
            print(f"Failed with {filter_name}: {e}")
            results[filter_name] = {'error': str(e)}
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Display results for first image
    first_image = collapsed_images[0]
    if first_image is not None and first_image.ndim == 3:
        display_channel = first_image[:, :, channel_index]
    else:
        display_channel = first_image
    
    # Original image
    axes[0, 0].imshow(display_channel, cmap='gray')
    axes[0, 0].set_title('Original Image\\nChannel 1 (DNA)')
    axes[0, 0].axis('off')
    
    # Show segmentation results with different filters
    filter_names = list(results.keys())
    for i, filter_name in enumerate(filter_names[:3]):
        if 'masks' in results[filter_name] and results[filter_name]['masks'][0] is not None:
            mask = results[filter_name]['masks'][0]
            count = results[filter_name]['counts'][0]
            
            axes[0, i+1].imshow(mask, cmap='tab20')
            axes[0, i+1].set_title(f'{filter_name}\\n{count} objects')
            axes[0, i+1].axis('off')
        else:
            axes[0, i+1].text(0.5, 0.5, f'{filter_name}\\nFailed', ha='center', va='center', 
                             transform=axes[0, i+1].transAxes)
            axes[0, i+1].axis('off')
    
    # Bottom row: Statistics
    
    # Object counts per image
    x_pos = np.arange(len(test_files))
    width = 0.2
    
    for i, filter_name in enumerate(filter_names):
        if 'counts' in results[filter_name]:
            counts = results[filter_name]['counts']
            axes[1, 0].bar(x_pos + i*width, counts, width, label=filter_name, alpha=0.8)
    
    axes[1, 0].set_xlabel('Image Index')
    axes[1, 0].set_ylabel('Object Count')
    axes[1, 0].set_title('Object Counts by Filter')
    axes[1, 0].set_xticks(x_pos + width*1.5)
    axes[1, 0].set_xticklabels([f'Image {i+1}' for i in range(len(test_files))])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total objects comparison
    filter_names_clean = [name for name in filter_names if 'counts' in results[name]]
    total_objects = [results[name]['total_objects'] for name in filter_names_clean]
    
    axes[1, 1].bar(filter_names_clean, total_objects, alpha=0.8, color=['blue', 'green', 'orange', 'red'])
    axes[1, 1].set_ylabel('Total Objects (All Images)')
    axes[1, 1].set_title('Total Object Counts')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Filtering effectiveness
    if len(filter_names_clean) > 1:
        baseline_count = results[filter_names_clean[0]]['total_objects']  # No filter
        removal_percentages = []
        filter_percentages = []
        
        for name in filter_names_clean[1:]:  # Skip "No filter"
            filtered_count = results[name]['total_objects']
            removal_pct = 100 * (baseline_count - filtered_count) / baseline_count if baseline_count > 0 else 0
            removal_percentages.append(removal_pct)
            
            # Extract filter percentage from name
            if '%' in name:
                filter_pct = int(name.split('%')[0])
                filter_percentages.append(filter_pct)
        
        if filter_percentages and removal_percentages:
            axes[1, 2].plot(filter_percentages, removal_percentages, 'bo-', linewidth=2, markersize=8)
            axes[1, 2].set_xlabel('Filter Threshold (%)')
            axes[1, 2].set_ylabel('Objects Removed (%)')
            axes[1, 2].set_title('Filter Effectiveness')
            axes[1, 2].grid(True, alpha=0.3)
    
    # Summary table
    axes[1, 3].axis('off')
    table_data = []
    for name in filter_names_clean:
        result = results[name]
        table_data.append([
            name.replace(' filter', ''),
            f"{result['total_objects']}",
            f"{result['mean_objects']:.1f}"
        ])
    
    table = axes[1, 3].table(cellText=table_data,
                            colLabels=['Filter', 'Total', 'Mean/Image'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 3].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig('batch_size_filtering_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comparison: batch_size_filtering_comparison.png")
    
    # Print summary
    print(f"\n=== BATCH SIZE FILTERING SUMMARY ===")
    for filter_name in filter_names_clean:
        result = results[filter_name]
        config_desc = "disabled" if not result['config']['enabled'] else f"{result['config']['min_size_percentage']}% threshold"
        print(f"{filter_name:15}: {result['total_objects']:3d} total objects ({result['mean_objects']:5.1f} per image) - {config_desc}")
    
    return results

if __name__ == "__main__":
    results = test_batch_with_size_filtering()
