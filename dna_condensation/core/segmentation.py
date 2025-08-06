from imageProcessingUtils.yolo.segmentation import segmentation_pipeline_yolo
import numpy as np

# Check for cupy availability (for potential future GPU optimizations)
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

def segment_image(image: np.ndarray, channel_index: int, use_gpu: bool = True, return_labels: bool = False) -> np.ndarray:
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

def bulk_segment_images(images: list, channel_index: int, use_gpu: bool = True, verbose: bool = True, return_labels: bool = False) -> list:
    """
    Segment a list of images using a pre-trained YOLO model.
    
    Args:
        images: List of input images as numpy arrays (uint8 format expected)
        channel_index: Index of the channel to use for segmentation (0-based)
                      For 2D images, this parameter is ignored
                      For 3D images, specifies which channel to extract (e.g., 0=R, 1=G, 2=B for RGB)
        use_gpu: Whether to use GPU acceleration if available
        verbose: Whether to print progress information
        return_labels: If True, return labeled masks with nucleus IDs; if False, return binary masks
    
    Returns:
        list: List of segmented image masks 
              - If return_labels=False: binary masks (uint8, 0/255)
              - If return_labels=True: labeled masks (uint16 with nucleus IDs, 0=background)
    
    Note:
        The underlying segmentation_pipeline_yolo returns for each image:
        Tuple of (instance_labels, binary_mask):
        - instance_labels: 2D uint16 array with unique IDs for each nucleus (0=background)
        - binary_mask: 2D boolean array marking all detected nuclear regions
    """
    segmented_masks = []
    
    if verbose:
        gpu_status = "GPU" if (use_gpu and HAS_CUPY) else "CPU"
        mask_type = "labeled" if return_labels else "binary"
        print(f"Segmenting {len(images)} images using {gpu_status} ({mask_type} masks, channel {channel_index})")
    
    for i, img in enumerate(images):
        if img is None:
            print(f"Skipping None image at index {i}")
            segmented_masks.append(None)
            continue
            
        try:
            mask = segment_image(img, channel_index=channel_index, use_gpu=use_gpu, return_labels=return_labels)
            segmented_masks.append(mask)
            
            if verbose and (i + 1) % 5 == 0:  # Progress update every 5 images
                print(f"Segmented {i + 1}/{len(images)} images")
                
        except Exception as e:
            print(f"Error segmenting image {i + 1}: {e}")
            segmented_masks.append(None)
    
    if verbose:
        successful = sum(1 for mask in segmented_masks if mask is not None)
        print(f"Successfully segmented {successful}/{len(images)} images")
    
    return segmented_masks