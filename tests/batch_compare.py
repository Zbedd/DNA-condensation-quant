import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import os
sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.preprocessor import collapse_z_axis
from dna_condensation.core.segmentation import segment_image

def batch_compare_methods(max_files=5):
    """Compare segmentation methods across multiple files."""
    print("=== BATCH COMPARISON OF SEGMENTATION METHODS ===\n")
    
    # Load config
    config = Config()
    nd2_folder_path = config.get("raw_nd2_path")
    channel_index = config.get("segmentation_channel_index", 1)
    
    # Get ND2 files
    nd2_objects = get_nd2_objects(nd2_folder_path)
    print(f"Found {len(nd2_objects)} ND2 files")
    
    # Limit the number of files for testing
    test_files = nd2_objects[:max_files]
    print(f"Testing first {len(test_files)} files:")
    for nd2 in test_files:
        print(f"  - {os.path.basename(nd2.filename)}")
    
    results = []
    
    for i, nd2 in enumerate(test_files):
        print(f"\n--- Processing {i+1}/{len(test_files)}: {os.path.basename(nd2.filename)} ---")
        
        try:
            # Preprocess
            collapsed_image = collapse_z_axis(nd2, method='mean', verbose=False)
            print(f"Image shape: {collapsed_image.shape}")
            
            # Test both methods
            file_results = {
                'filename': os.path.basename(nd2.filename),
                'image_shape': collapsed_image.shape
            }
            
            for method in ['yolo', 'watershed']:
                print(f"Testing {method}...", end=' ')
                start_time = time.time()
                
                try:
                    labels = segment_image(collapsed_image, channel_index=channel_index, 
                                         method=method, return_labels=True)
                    num_objects = len(np.unique(labels)) - 1
                    
                    # Calculate size statistics
                    sizes = []
                    for label_id in range(1, labels.max() + 1):
                        size = np.sum(labels == label_id)
                        sizes.append(size)
                    
                    elapsed = time.time() - start_time
                    
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
                    print(f"FAILED ({elapsed:.1f}s): {e}")
            
            results.append(file_results)
            
        except Exception as e:
            print(f"Failed to process file: {e}")
            continue
    
    # Create summary visualization
    print(f"\n=== SUMMARY RESULTS ===")
    
    # Collect statistics
    yolo_counts = []
    watershed_counts = []
    yolo_times = []
    watershed_times = []
    filenames = []
    
    success_count = {'yolo': 0, 'watershed': 0}
    
    for result in results:
        filenames.append(result['filename'][:20])  # Truncate long names
        
        for method in ['yolo', 'watershed']:
            if result[method]['success']:
                success_count[method] += 1
                if method == 'yolo':
                    yolo_counts.append(result[method]['count'])
                    yolo_times.append(result[method]['time'])
                else:
                    watershed_counts.append(result[method]['count'])
                    watershed_times.append(result[method]['time'])
            else:
                if method == 'yolo':
                    yolo_counts.append(0)
                    yolo_times.append(result[method]['time'])
                else:
                    watershed_counts.append(0)
                    watershed_times.append(result[method]['time'])
    
    # Print summary table
    print(f"\n{'Filename':<20} {'YOLO':<8} {'Watershed':<8} {'YOLO Time':<10} {'Watershed Time':<10}")
    print("-" * 70)
    
    for i, result in enumerate(results):
        yolo_str = f"{result['yolo']['count']}" if result['yolo']['success'] else "FAIL"
        watershed_str = f"{result['watershed']['count']}" if result['watershed']['success'] else "FAIL"
        
        print(f"{result['filename'][:20]:<20} {yolo_str:<8} {watershed_str:<8} "
              f"{result['yolo']['time']:.1f}s{'':<6} {result['watershed']['time']:.1f}s")
    
    # Statistics
    print(f"\n=== STATISTICS ===")
    print(f"Files processed: {len(results)}")
    print(f"YOLO success rate: {success_count['yolo']}/{len(results)} ({100*success_count['yolo']/len(results):.1f}%)")
    print(f"Watershed success rate: {success_count['watershed']}/{len(results)} ({100*success_count['watershed']/len(results):.1f}%)")
    
    if yolo_counts and any(c > 0 for c in yolo_counts):
        valid_yolo = [c for c in yolo_counts if c > 0]
        print(f"\nYOLO object counts: {np.mean(valid_yolo):.1f} ± {np.std(valid_yolo):.1f} (range: {np.min(valid_yolo)}-{np.max(valid_yolo)})")
    
    if watershed_counts and any(c > 0 for c in watershed_counts):
        valid_watershed = [c for c in watershed_counts if c > 0]
        print(f"Watershed object counts: {np.mean(valid_watershed):.1f} ± {np.std(valid_watershed):.1f} (range: {np.min(valid_watershed)}-{np.max(valid_watershed)})")
    
    if yolo_times:
        print(f"\nYOLO average time: {np.mean(yolo_times):.1f}s")
    if watershed_times:
        print(f"Watershed average time: {np.mean(watershed_times):.1f}s")
    
    # Create comparison plot
    if len(results) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Object counts comparison
        x = np.arange(len(filenames))
        width = 0.35
        
        ax1.bar(x - width/2, yolo_counts, width, label='YOLO', alpha=0.7, color='blue')
        ax1.bar(x + width/2, watershed_counts, width, label='Watershed', alpha=0.7, color='green')
        ax1.set_xlabel('Files')
        ax1.set_ylabel('Number of Objects')
        ax1.set_title('Object Counts by Method')
        ax1.set_xticks(x)
        ax1.set_xticklabels(filenames, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Processing time comparison
        ax2.bar(x - width/2, yolo_times, width, label='YOLO', alpha=0.7, color='blue')
        ax2.bar(x + width/2, watershed_times, width, label='Watershed', alpha=0.7, color='green')
        ax2.set_xlabel('Files')
        ax2.set_ylabel('Processing Time (seconds)')
        ax2.set_title('Processing Time by Method')
        ax2.set_xticks(x)
        ax2.set_xticklabels(filenames, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('batch_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved batch comparison: batch_comparison.png")
    
    return results

if __name__ == "__main__":
    results = batch_compare_methods(max_files=5)
