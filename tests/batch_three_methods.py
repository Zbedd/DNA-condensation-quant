import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import os
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.z_stack_handling import collapse_z_axis
from dna_condensation.core.segmentation import segment_image

def batch_test_three_methods(max_files=5):
    """Test all three methods across multiple files."""
    print("=== BATCH TEST: THREE SEGMENTATION METHODS ===\n")
    
    # Load config
    config = Config()
    nd2_folder_path = config.get("raw_nd2_path")
    channel_index = config.get("segmentation_channel_index", 1)
    
    # Get ND2 files
    nd2_objects = get_nd2_objects(nd2_folder_path)
    test_files = nd2_objects[:max_files]
    
    print(f"Testing {len(test_files)} files with all three methods:")
    for nd2 in test_files:
        print(f"  - {os.path.basename(nd2.filename)}")
    
    all_results = []
    
    for i, nd2 in enumerate(test_files):
        print(f"\n--- File {i+1}/{len(test_files)}: {os.path.basename(nd2.filename)} ---")
        
        try:
            # Preprocess
            collapsed_image = collapse_z_axis(nd2, method='mean', verbose=False)
            
            file_results = {
                'filename': os.path.basename(nd2.filename),
                'image_shape': collapsed_image.shape
            }
            
            # Test all three methods
            for method in ['yolo', 'otsu', 'watershed']:
                print(f"  {method.upper()}...", end=' ')
                start_time = time.time()
                
                try:
                    labels = segment_image(collapsed_image, channel_index=channel_index, 
                                         method=method, return_labels=True)
                    num_objects = len(np.unique(labels)) - 1
                    elapsed = time.time() - start_time
                    
                    # Calculate size statistics
                    sizes = []
                    for label_id in range(1, labels.max() + 1):
                        size = np.sum(labels == label_id)
                        sizes.append(size)
                    
                    file_results[method] = {
                        'count': num_objects,
                        'time': elapsed,
                        'success': True,
                        'mean_size': np.mean(sizes) if sizes else 0,
                        'size_range': (np.min(sizes), np.max(sizes)) if sizes else (0, 0)
                    }
                    
                    print(f"{num_objects} objects ({elapsed:.1f}s)")
                    
                except Exception as e:
                    elapsed = time.time() - start_time
                    file_results[method] = {
                        'count': 0,
                        'time': elapsed,
                        'success': False,
                        'error': str(e),
                        'mean_size': 0,
                        'size_range': (0, 0)
                    }
                    print(f"FAILED ({elapsed:.1f}s)")
            
            all_results.append(file_results)
            
        except Exception as e:
            print(f"Failed to process file: {e}")
            continue
    
    # Analyze results
    print(f"\n=== BATCH ANALYSIS ===")
    
    # Summary table
    print(f"\\n{'Filename':<25} {'YOLO':<6} {'Otsu':<6} {'Watershed':<10} {'W/O Ratio':<8}")
    print("-" * 70)
    
    ratios = []
    yolo_counts = []
    otsu_counts = []
    watershed_counts = []
    
    for result in all_results:
        filename = result['filename'][:24]
        yolo_count = result['yolo']['count'] if result['yolo']['success'] else 0
        otsu_count = result['otsu']['count'] if result['otsu']['success'] else 0
        watershed_count = result['watershed']['count'] if result['watershed']['success'] else 0
        
        yolo_counts.append(yolo_count)
        otsu_counts.append(otsu_count)
        watershed_counts.append(watershed_count)
        
        # Calculate watershed/otsu ratio
        if otsu_count > 0:
            ratio = watershed_count / otsu_count
            ratios.append(ratio)
            ratio_str = f"{ratio:.2f}"
        else:
            ratio_str = "N/A"
        
        print(f"{filename:<25} {yolo_count:<6} {otsu_count:<6} {watershed_count:<10} {ratio_str:<8}")
    
    # Statistics
    print(f"\\n=== STATISTICS ===")
    
    valid_yolo = [c for c in yolo_counts if c > 0]
    valid_otsu = [c for c in otsu_counts if c > 0]
    valid_watershed = [c for c in watershed_counts if c > 0]
    
    if valid_yolo:
        print(f"YOLO:      {np.mean(valid_yolo):.1f} ± {np.std(valid_yolo):.1f} objects (range: {np.min(valid_yolo)}-{np.max(valid_yolo)})")
    if valid_otsu:
        print(f"Otsu:      {np.mean(valid_otsu):.1f} ± {np.std(valid_otsu):.1f} objects (range: {np.min(valid_otsu)}-{np.max(valid_otsu)})")
    if valid_watershed:
        print(f"Watershed: {np.mean(valid_watershed):.1f} ± {np.std(valid_watershed):.1f} objects (range: {np.min(valid_watershed)}-{np.max(valid_watershed)})")
    
    if ratios:
        print(f"\\nWatershed/Otsu ratio: {np.mean(ratios):.2f} ± {np.std(ratios):.2f} (range: {np.min(ratios):.2f}-{np.max(ratios):.2f})")
        
        if np.mean(ratios) > 1.2:
            print("→ Watershed consistently detects more objects than Otsu (good separation)")
        elif np.mean(ratios) < 0.8:
            print("→ Watershed is more conservative than Otsu (may need adjustment)")
        else:
            print("→ Watershed and Otsu are well-balanced")
    
    # Create visualization
    if len(all_results) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        filenames = [result['filename'][:15] for result in all_results]
        x = np.arange(len(filenames))
        
        # Object counts comparison
        width = 0.25
        axes[0, 0].bar(x - width, yolo_counts, width, label='YOLO', alpha=0.7, color='red')
        axes[0, 0].bar(x, otsu_counts, width, label='Otsu', alpha=0.7, color='green')
        axes[0, 0].bar(x + width, watershed_counts, width, label='Watershed', alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Files')
        axes[0, 0].set_ylabel('Number of Objects')
        axes[0, 0].set_title('Object Counts by Method')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(filenames, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Processing times
        yolo_times = [result['yolo']['time'] for result in all_results]
        otsu_times = [result['otsu']['time'] for result in all_results]
        watershed_times = [result['watershed']['time'] for result in all_results]
        
        axes[0, 1].bar(x - width, yolo_times, width, label='YOLO', alpha=0.7, color='red')
        axes[0, 1].bar(x, otsu_times, width, label='Otsu', alpha=0.7, color='green')
        axes[0, 1].bar(x + width, watershed_times, width, label='Watershed', alpha=0.7, color='blue')
        axes[0, 1].set_xlabel('Files')
        axes[0, 1].set_ylabel('Processing Time (seconds)')
        axes[0, 1].set_title('Processing Times by Method')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(filenames, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Watershed/Otsu ratio plot
        if ratios and len(ratios) == len(filenames):
            axes[1, 0].plot(x, ratios, 'bo-', linewidth=2, markersize=8)
            axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal counts')
            axes[1, 0].set_xlabel('Files')
            axes[1, 0].set_ylabel('Watershed/Otsu Ratio')
            axes[1, 0].set_title('Watershed vs Otsu Consistency')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(filenames, rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Method comparison scatter plot
        if valid_otsu and valid_watershed and len(valid_otsu) == len(valid_watershed):
            axes[1, 1].scatter(valid_otsu, valid_watershed, alpha=0.7, s=100)
            # Add diagonal line
            max_val = max(max(valid_otsu), max(valid_watershed))
            axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Equal counts')
            axes[1, 1].set_xlabel('Otsu Object Count')
            axes[1, 1].set_ylabel('Watershed Object Count')
            axes[1, 1].set_title('Otsu vs Watershed Correlation')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('batch_three_methods.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\\nSaved batch analysis: batch_three_methods.png")
    
    return all_results

if __name__ == "__main__":
    results = batch_test_three_methods(max_files=5)
