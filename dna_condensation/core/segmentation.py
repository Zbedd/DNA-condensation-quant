from imageProcessingUtils.yolo.segmentation import segmentation_pipeline_yolo
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.filters import gaussian
import cv2

# Check for cupy availability (for potential future GPU optimizations)
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

def segment_image_yolo(image: np.ndarray, channel_index: int, use_gpu: bool = True, return_labels: bool = False) -> np.ndarray:
    """
    Segment the input image using a pre-trained YOLO model.
    
    Args:
        image: Input image as a numpy array (uint8 format expected)
               Can be 2D grayscale or 3D multi-channel
        channel_index: Index of the channel to use for segmentation (0-based)
                      For 2D images, this parameter is ignored
                      For 3D images, specifies which channel to extract (e.g., 0=R, 1=G, 2=B for RGB)
        use_gpu: Whether to use GPU acceleration if available (passed to YOLO)
        return_labels: If True, return labeled mask with object IDs; if False, return binary mask
    
    Returns:
        numpy.ndarray: Segmented image mask 
                      - If return_labels=False: binary mask (uint8, 0/255)
                      - If return_labels=True: labeled mask (uint16 with nucleus IDs, 0=background)
    
    Note:
        The underlying segmentation_pipeline_yolo returns:
        Tuple of (instance_labels, binary_mask):
        - instance_labels: 2D uint16 array with unique IDs for each nucleus (0=background)
        - binary_mask: 2D boolean array marking all detected nuclear regions
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a numpy array of dtype uint8")
    
    # Extract specific channel for segmentation (YOLO segmentation expects 2D)
    if image.ndim == 3:
        # Multi-channel image - extract specified channel
        if channel_index < 0 or channel_index >= image.shape[2]:
            raise ValueError(f"Channel index {channel_index} is out of range for image with {image.shape[2]} channels")
        processed_image = image[:, :, channel_index]
    elif image.ndim == 2:
        # Already single channel - channel_index is ignored
        processed_image = image
        if channel_index != 0:
            print(f"Warning: channel_index={channel_index} specified for 2D image, using single channel")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    # Run segmentation pipeline with processed image
    if use_gpu and HAS_CUPY:
        print("GPU acceleration available - YOLO pipeline will use GPU if configured")
    elif use_gpu and not HAS_CUPY:
        print("GPU requested but CuPy not available, using CPU")
    
    # YOLO segmentation returns (instance_labels, binary_mask)
    # instance_labels: uint16 with nucleus IDs (0=background, 1,2,3...=nuclei)
    # binary_mask: bool indicating presence of any nucleus
    instance_labels, binary_mask = segmentation_pipeline_yolo(processed_image)
    
    if return_labels:
        # Return the labeled mask with nucleus IDs
        return instance_labels
    else:
        # Convert binary mask to uint8 (0/255 format)
        return (binary_mask.astype(np.uint8) * 255)

def segment_image_watershed(image: np.ndarray, channel_index: int, use_gpu: bool = True, return_labels: bool = False) -> np.ndarray:
    """
    Segment the input image using watershed segmentation.
    
    Args:
        image: Input image as a numpy array (uint8 format expected)
               Can be 2D grayscale or 3D multi-channel
        channel_index: Index of the channel to use for segmentation (0-based)
                      For 2D images, this parameter is ignored
                      For 3D images, specifies which channel to extract
        use_gpu: Whether to use GPU acceleration if available (currently not used for watershed)
        return_labels: If True, return labeled mask with object IDs; if False, return binary mask
    
    Returns:
        numpy.ndarray: Segmented image mask 
                      - If return_labels=False: binary mask (uint8, 0/255)
                      - If return_labels=True: labeled mask (uint16 with nucleus IDs, 0=background)
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a numpy array of dtype uint8")
    
    # Extract specific channel for segmentation
    if image.ndim == 3:
        # Multi-channel image - extract specified channel
        if channel_index < 0 or channel_index >= image.shape[2]:
            raise ValueError(f"Channel index {channel_index} is out of range for image with {image.shape[2]} channels")
        processed_image = image[:, :, channel_index]
    elif image.ndim == 2:
        # Already single channel - channel_index is ignored
        processed_image = image
        if channel_index != 0:
            print(f"Warning: channel_index={channel_index} specified for 2D image, using single channel")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    # Watershed segmentation pipeline
    try:
        # 1. Apply moderate Gaussian blur to reduce noise but preserve edges
        blurred = gaussian(processed_image.astype(np.float32), sigma=1.5)
        blurred = (blurred * 255).astype(np.uint8)
        
        # 2. Use more conservative thresholding to respect nuclear boundaries
        # Start with Otsu but make it more restrictive to avoid over-expansion
        otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        
        # Use same threshold as Otsu method for consistency
        conservative_thresh = int(otsu_thresh * 1.2)  # Same as Otsu method
        _, binary_mask = cv2.threshold(blurred, conservative_thresh, 255, cv2.THRESH_BINARY)
        
        # 3. Light morphological cleanup - minimal to preserve boundaries
        # Only remove very small noise, avoid closing gaps between nuclei
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
        
        # 4. Remove very small objects (noise) but keep size filter minimal
        from scipy.ndimage import label as scipy_label
        labeled_mask, num_features = scipy_label(binary_mask)
        
        # Use same minimum size as Otsu method for consistency
        min_nucleus_size = 200  # Same as Otsu method
        for i in range(1, num_features + 1):
            if np.sum(labeled_mask == i) < min_nucleus_size:
                binary_mask[labeled_mask == i] = 0
        
        # 5. Calculate distance transform on the conservative mask
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # 6. Find seeds more conservatively to avoid over-segmentation
        # Use h_maxima which is more robust than manual peak finding
        from skimage.morphology import h_maxima
        
        # Use h_maxima with lower threshold to be more similar to Otsu results
        # This finds more peaks, getting closer to Otsu object counts
        local_max = h_maxima(distance, 0.2 * distance.max())
        
        # 7. Label the seeds
        seeds, num_seeds = ndimage.label(local_max)
        
        # 8. Apply watershed with the original (less smoothed) distance transform
        # This helps maintain sharp boundaries
        labels = watershed(-distance, seeds, mask=binary_mask > 0)
        
        # 9. Minimal post-processing - avoid filling gaps between nuclei
        # Only smooth individual nucleus boundaries without merging
        final_labels = labels.astype(np.uint16)
        
        # Light smoothing within each nucleus without expanding boundaries
        for label_id in range(1, labels.max() + 1):
            label_mask = (labels == label_id).astype(np.uint8) * 255
            # Very light smoothing - just fill small holes within nuclei
            smoothed_mask = cv2.morphologyEx(label_mask, cv2.MORPH_CLOSE, 
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            # Update only pixels that were already part of this nucleus
            final_labels[smoothed_mask > label_mask] = 0  # Remove expansions
            
        # Convert to proper output format
        instance_labels = final_labels.astype(np.uint16)
        binary_output = (final_labels > 0).astype(bool)
        
        if return_labels:
            return instance_labels
        else:
            # Convert binary mask to uint8 (0/255 format)
            return (binary_output.astype(np.uint8) * 255)
            
    except Exception as e:
        print(f"Watershed segmentation failed: {e}")
        # Return empty mask on failure
        empty_shape = processed_image.shape
        if return_labels:
            return np.zeros(empty_shape, dtype=np.uint16)
        else:
            return np.zeros(empty_shape, dtype=np.uint8)

def segment_image_otsu(image: np.ndarray, channel_index: int, use_gpu: bool = True, return_labels: bool = False) -> np.ndarray:
    """
    Segment the input image using 1.2x Otsu thresholding with connected components labeling.
    This method provides a baseline that's more sensitive than watershed but less than YOLO.
    
    Args:
        image: Input image as a numpy array (uint8 format expected)
               Can be 2D grayscale or 3D multi-channel
        channel_index: Index of the channel to use for segmentation (0-based)
                      For 2D images, this parameter is ignored
                      For 3D images, specifies which channel to extract
        use_gpu: Whether to use GPU acceleration if available (for future CuPy implementation)
        return_labels: If True, return labeled mask with object IDs; if False, return binary mask
    
    Returns:
        numpy.ndarray: Segmented image mask 
                      - If return_labels=False: binary mask (uint8, 0/255)
                      - If return_labels=True: labeled mask (uint16 with nucleus IDs, 0=background)
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a numpy array of dtype uint8")
    
    # Extract specific channel for segmentation
    if image.ndim == 3:
        if channel_index < 0 or channel_index >= image.shape[2]:
            raise ValueError(f"Channel index {channel_index} is out of range for image with {image.shape[2]} channels")
        processed_image = image[:, :, channel_index]
    elif image.ndim == 2:
        processed_image = image
        if channel_index != 0:
            print(f"Warning: channel_index={channel_index} specified for 2D image, using single channel")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    try:
        # Use the same approach as the successful threshold debug
        # No Gaussian blur - work directly with the original image
        
        # Apply 1.2x Otsu thresholding using scikit-image (same as debug)
        from skimage import filters
        otsu_thresh = filters.threshold_otsu(processed_image)
        moderate_thresh = otsu_thresh * 1.2  # 1.2x Otsu for moderate sensitivity
        binary_mask = processed_image > moderate_thresh
        
        # Convert to uint8 for morphological operations
        binary_mask = binary_mask.astype(np.uint8) * 255
        
        # Very minimal morphological cleanup - just remove tiny noise
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_tiny)
        
        # Convert back to boolean for labeling
        binary_mask = binary_mask > 0
        
        # Remove very small objects (noise) with minimal filtering
        from scipy.ndimage import label as scipy_label
        labeled_mask, num_features = scipy_label(binary_mask)
        
        # Very permissive minimum size filter to match debug results
        min_nucleus_size = 50  # Much smaller to preserve more objects
        for i in range(1, num_features + 1):
            if np.sum(labeled_mask == i) < min_nucleus_size:
                binary_mask[labeled_mask == i] = 0
        
        # Final connected components labeling
        final_labels, num_labels = scipy_label(binary_mask)
        
        # Convert to proper output format
        if return_labels:
            return final_labels.astype(np.uint16)
        else:
            # Convert binary mask to uint8 (0/255 format)
            return (final_labels > 0).astype(np.uint8) * 255
            
    except Exception as e:
        print(f"Otsu segmentation failed: {e}")
        # Return empty mask on failure
        empty_shape = processed_image.shape
        if return_labels:
            return np.zeros(empty_shape, dtype=np.uint16)
        else:
            return np.zeros(empty_shape, dtype=np.uint8)

def segment_image(image: np.ndarray, channel_index: int, method: str = 'yolo', use_gpu: bool = True, return_labels: bool = False) -> np.ndarray:
    """
    Segment the input image using the specified method.
    
    Args:
        image: Input image as a numpy array (uint8 format expected)
        channel_index: Index of the channel to use for segmentation (0-based)
        method: Segmentation method ('yolo', 'watershed', or 'otsu')
        use_gpu: Whether to use GPU acceleration if available
        return_labels: If True, return labeled mask with object IDs; if False, return binary mask
    
    Returns:
        numpy.ndarray: Segmented image mask
    """
    if method.lower() == 'yolo':
        return segment_image_yolo(image, channel_index, use_gpu, return_labels)
    elif method.lower() == 'watershed':
        return segment_image_watershed(image, channel_index, use_gpu, return_labels)
    elif method.lower() == 'otsu':
        return segment_image_otsu(image, channel_index, use_gpu, return_labels)
    else:
        raise ValueError(f"Unknown segmentation method: {method}. Choose 'yolo', 'watershed', or 'otsu'")

def filter_labels_by_size(labels: np.ndarray, min_size_percentage: float = 10.0, verbose: bool = False) -> np.ndarray:
    """
    Filter labeled mask by removing objects smaller than a percentage of the median size.
    
    Args:
        labels: Labeled mask with object IDs (background = 0)
        min_size_percentage: Minimum size as percentage of median size (e.g., 10.0 for 10%)
        verbose: Whether to print filtering statistics
    
    Returns:
        numpy.ndarray: Filtered labeled mask with small objects removed and labels renumbered
    """
    if not isinstance(labels, np.ndarray):
        raise ValueError("Input labels must be a numpy array")
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        if verbose:
            print("No objects found in labels, returning unchanged")
        return labels.copy()
    
    # Calculate sizes for each object
    sizes = []
    for label_id in unique_labels:
        size = np.sum(labels == label_id)
        sizes.append(size)
    
    sizes = np.array(sizes)
    median_size = np.median(sizes)
    min_size_threshold = median_size * (min_size_percentage / 100.0)
    
    if verbose:
        print(f"Size filtering statistics:")
        print(f"  Objects before filtering: {len(sizes)}")
        print(f"  Median size: {median_size:.1f} pixels")
        print(f"  Minimum size threshold ({min_size_percentage}%): {min_size_threshold:.1f} pixels")
    
    # Create filtered mask
    filtered_labels = np.zeros_like(labels)
    new_label_id = 1
    objects_removed = 0
    
    for i, label_id in enumerate(unique_labels):
        if sizes[i] >= min_size_threshold:
            # Keep this object with new sequential label
            filtered_labels[labels == label_id] = new_label_id
            new_label_id += 1
        else:
            # Remove this object
            objects_removed += 1
    
    objects_remaining = new_label_id - 1
    
    if verbose:
        print(f"  Objects after filtering: {objects_remaining}")
        print(f"  Objects removed: {objects_removed}")
        print(f"  Removal percentage: {100*objects_removed/len(sizes):.1f}%")
    
    return filtered_labels

def bulk_segment_images(images: list, channel_index: int, method: str = 'yolo', use_gpu: bool = True, verbose: bool = True, return_labels: bool = False, size_filter_config: dict = None) -> list:
    """
    Segment a list of images using the specified method.
    
    Args:
        images: List of input images as numpy arrays (uint8 format expected)
        channel_index: Index of the channel to use for segmentation (0-based)
                      For 2D images, this parameter is ignored
                      For 3D images, specifies which channel to extract (e.g., 0=R, 1=G, 2=B for RGB)
        method: Segmentation method ('yolo', 'watershed', or 'otsu')
        use_gpu: Whether to use GPU acceleration if available
        verbose: Whether to print progress information
        return_labels: If True, return labeled masks with nucleus IDs; if False, return binary masks
        size_filter_config: Dictionary with size filtering parameters:
                           {'enabled': bool, 'min_size_percentage': float}
                           If None, no size filtering is applied
    
    Returns:
        list: List of segmented image masks 
              - If return_labels=False: binary masks (uint8, 0/255)
              - If return_labels=True: labeled masks (uint16 with nucleus IDs, 0=background)
    
    Note:
        For YOLO method: The underlying segmentation_pipeline_yolo returns for each image:
        Tuple of (instance_labels, binary_mask):
        - instance_labels: 2D uint16 array with unique IDs for each nucleus (0=background)
        - binary_mask: 2D boolean array marking all detected nuclear regions
        
        For watershed method: Uses distance transform and watershed algorithm for segmentation.
    """
    segmented_masks = []
    
    # Parse size filtering configuration
    apply_size_filter = False
    min_size_percentage = 10.0
    
    if size_filter_config is not None:
        apply_size_filter = size_filter_config.get('enabled', False)
        min_size_percentage = size_filter_config.get('min_size_percentage', 10.0)
    
    if verbose:
        gpu_status = "GPU" if (use_gpu and HAS_CUPY and method.lower() == 'yolo') else "CPU"
        mask_type = "labeled" if return_labels else "binary"
        filter_status = f"with {min_size_percentage}% size filter" if apply_size_filter else "no filtering"
        print(f"Segmenting {len(images)} images using {method.upper()} on {gpu_status} ({mask_type} masks, channel {channel_index}, {filter_status})")
    
    for i, img in enumerate(images):
        if img is None:
            print(f"Skipping None image at index {i}")
            segmented_masks.append(None)
            continue
            
        try:
            # Get the segmentation result
            if return_labels or apply_size_filter:
                # Always get labels first if we need filtering or want labels output
                mask = segment_image(img, channel_index=channel_index, method=method, use_gpu=use_gpu, return_labels=True)
                
                # Apply size filtering if enabled
                if apply_size_filter:
                    original_count = len(np.unique(mask)) - 1
                    mask = filter_labels_by_size(mask, min_size_percentage=min_size_percentage, verbose=False)
                    filtered_count = len(np.unique(mask)) - 1
                    
                    if verbose and i < 3:  # Show filtering stats for first few images
                        print(f"  Image {i+1}: {original_count} → {filtered_count} objects after {min_size_percentage}% size filter")
                
                # Convert to binary if that's what was requested
                if not return_labels:
                    mask = (mask > 0).astype(np.uint8) * 255
                    
            else:
                # Get binary mask directly if no filtering needed
                mask = segment_image(img, channel_index=channel_index, method=method, use_gpu=use_gpu, return_labels=False)
            
            segmented_masks.append(mask)
            
            if verbose and (i + 1) % 5 == 0:  # Progress update every 5 images
                print(f"Segmented {i + 1}/{len(images)} images")
                
        except Exception as e:
            print(f"Error segmenting image {i + 1}: {e}")
            segmented_masks.append(None)
    
    if verbose:
        successful = sum(1 for mask in segmented_masks if mask is not None)
        print(f"Successfully segmented {successful}/{len(images)} images using {method.upper()}")
        
        if apply_size_filter and successful > 0:
            print(f"Size filtering applied: minimum {min_size_percentage}% of median object size")
    
    return segmented_masks