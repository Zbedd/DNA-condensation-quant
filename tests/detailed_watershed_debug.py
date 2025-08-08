import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import filters, feature, morphology, segmentation
from scipy import ndimage
import cv2

sys.path.append('c:/VScode/DNA-condensation-quant')

from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.z_stack_handling import collapse_z_axis

def detailed_watershed_debug():
    """Step by step debug of watershed segmentation."""
    print("=== DETAILED WATERSHED DEBUG ===\n")
    
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
        processed_image = collapsed_image[:, :, channel_index]
    else:
        processed_image = collapsed_image
    
    print(f"Image shape: {processed_image.shape}")
    print(f"Image range: {processed_image.min()} - {processed_image.max()}")
    
    # Step 1: Thresholding
    print("\n--- Step 1: Thresholding ---")
    otsu_threshold = filters.threshold_otsu(processed_image)
    threshold = 2.0 * otsu_threshold  # Conservative threshold
    binary_mask = processed_image > threshold
    
    print(f"Otsu threshold: {otsu_threshold}")
    print(f"Used threshold: {threshold}")
    print(f"Binary pixels: {np.sum(binary_mask)} ({100*np.sum(binary_mask)/binary_mask.size:.1f}%)")
    
    if np.sum(binary_mask) == 0:
        print("ERROR: No pixels above threshold!")
        return
    
    # Step 2: Morphological cleaning
    print("\n--- Step 2: Morphological Cleaning ---")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    print(f"Cleaned pixels: {np.sum(cleaned)} ({100*np.sum(cleaned)/cleaned.size:.1f}%)")
    
    if np.sum(cleaned) == 0:
        print("ERROR: No pixels after morphological cleaning!")
        return
    
    # Step 3: Distance transform
    print("\n--- Step 3: Distance Transform ---")
    distance = ndimage.distance_transform_edt(cleaned)
    print(f"Distance range: {distance.min()} - {distance.max()}")
    
    if distance.max() == 0:
        print("ERROR: No distance transform computed!")
        return
    
    # Step 4: Find local maxima (seeds)
    print("\n--- Step 4: Find Seeds ---")
    
    # Try different peak finding methods
    methods_to_try = [
        ("peak_local_maxima_0.7", lambda: feature.peak_local_maxima(distance, min_distance=20, threshold_abs=0.7*distance.max())),
        ("peak_local_maxima_0.5", lambda: feature.peak_local_maxima(distance, min_distance=20, threshold_abs=0.5*distance.max())),
        ("peak_local_maxima_0.3", lambda: feature.peak_local_maxima(distance, min_distance=20, threshold_abs=0.3*distance.max())),
        ("h_maxima_0.7", lambda: morphology.h_maxima(distance, 0.7*distance.max())),
        ("h_maxima_0.5", lambda: morphology.h_maxima(distance, 0.5*distance.max())),
        ("h_maxima_0.3", lambda: morphology.h_maxima(distance, 0.3*distance.max())),
    ]
    
    seeds_results = {}
    for name, method in methods_to_try:
        try:
            if "peak_local_maxima" in name:
                peaks = method()
                if len(peaks[0]) > 0:
                    seeds_mask = np.zeros_like(distance, dtype=bool)
                    seeds_mask[peaks] = True
                    num_seeds = len(peaks[0])
                else:
                    seeds_mask = np.zeros_like(distance, dtype=bool)
                    num_seeds = 0
            else:  # h_maxima
                seeds_mask = method()
                num_seeds = np.sum(seeds_mask)
            
            seeds_results[name] = (seeds_mask, num_seeds)
            print(f"  {name}: {num_seeds} seeds")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            seeds_results[name] = (np.zeros_like(distance, dtype=bool), 0)
    
    # Choose best seed method
    best_method = None
    best_count = 0
    for name, (seeds_mask, count) in seeds_results.items():
        if 5 <= count <= 100:  # Reasonable range
            if count > best_count:
                best_count = count
                best_method = name
    
    if best_method is None:
        # Fallback to method with any seeds
        for name, (seeds_mask, count) in seeds_results.items():
            if count > 0:
                best_method = name
                best_count = count
                break
    
    if best_method is None:
        print("ERROR: No seeds found with any method!")
        return
    
    print(f"\nUsing {best_method} with {best_count} seeds")
    seeds_mask, _ = seeds_results[best_method]
    
    # Step 5: Label seeds
    print("\n--- Step 5: Label Seeds ---")
    markers, num_markers = ndimage.label(seeds_mask)
    print(f"Labeled markers: {num_markers}")
    
    if num_markers == 0:
        print("ERROR: No markers created!")
        return
    
    # Step 6: Watershed
    print("\n--- Step 6: Watershed ---")
    # Create watershed input
    watershed_input = -distance  # Negative for watershed
    watershed_input[~cleaned.astype(bool)] = 0  # Mask background
    
    # Apply watershed
    labels = segmentation.watershed(watershed_input, markers, mask=cleaned.astype(bool))
    num_objects = len(np.unique(labels)) - 1  # Subtract background
    print(f"Watershed objects: {num_objects}")
    
    # Visualize all steps
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Row 1: Original processing
    axes[0, 0].imshow(processed_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(binary_mask, cmap='gray')
    axes[0, 1].set_title(f'Binary Mask\n{np.sum(binary_mask)} pixels')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cleaned, cmap='gray')
    axes[0, 2].set_title(f'Cleaned Mask\n{np.sum(cleaned)} pixels')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(distance, cmap='viridis')
    axes[0, 3].set_title(f'Distance Transform\nMax: {distance.max():.1f}')
    axes[0, 3].axis('off')
    
    # Row 2: Seeds comparison
    axes[1, 0].imshow(processed_image, cmap='gray', alpha=0.7)
    if best_count > 0:
        y_coords, x_coords = np.where(seeds_mask)
        axes[1, 0].scatter(x_coords, y_coords, c='red', s=20, alpha=0.8)
    axes[1, 0].set_title(f'Seeds ({best_method})\n{best_count} seeds')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(markers, cmap='tab20')
    axes[1, 1].set_title(f'Markers\n{num_markers} regions')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(labels, cmap='tab20')
    axes[1, 2].set_title(f'Watershed Labels\n{num_objects} objects')
    axes[1, 2].axis('off')
    
    # Overlay result on original
    if num_objects > 0:
        overlay = np.stack([processed_image]*3, axis=2)
        # Add colored boundaries
        from scipy.ndimage import binary_erosion
        boundaries = np.zeros_like(labels, dtype=bool)
        for label_id in range(1, labels.max() + 1):
            label_mask = labels == label_id
            eroded = binary_erosion(label_mask)
            boundary = label_mask & ~eroded
            boundaries |= boundary
        overlay[boundaries] = [255, 0, 0]
        axes[1, 3].imshow(overlay)
    else:
        axes[1, 3].imshow(processed_image, cmap='gray')
    axes[1, 3].set_title('Final Result')
    axes[1, 3].axis('off')
    
    # Row 3: Method comparison
    methods_plot = list(seeds_results.keys())[:4]
    for i, method in enumerate(methods_plot):
        seeds_mask_plot, count = seeds_results[method]
        axes[2, i].imshow(processed_image, cmap='gray', alpha=0.7)
        if count > 0:
            y_coords, x_coords = np.where(seeds_mask_plot)
            axes[2, i].scatter(x_coords, y_coords, c='red', s=10, alpha=0.8)
        axes[2, i].set_title(f'{method}\n{count} seeds')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('watershed_detailed_debug.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved detailed debug: watershed_detailed_debug.png")
    
    # Suggest fixes
    if num_objects == 0:
        print("\n=== SUGGESTED FIXES ===")
        if best_count == 0:
            print("1. Lower the peak threshold (try 0.3 instead of 0.7)")
            print("2. Reduce minimum distance between peaks")
            print("3. Try different peak finding method")
        else:
            print("1. Check watershed input preparation")
            print("2. Verify mask is being applied correctly")
    else:
        print(f"\n✓ SUCCESS: Found {num_objects} objects")
        
    return labels

if __name__ == "__main__":
    result = detailed_watershed_debug()
