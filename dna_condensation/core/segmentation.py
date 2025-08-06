from imageProcessingUtils.yolo.segmentation import segmentation_pipeline_yolo
import numpy as np

# Check for cupy availability (for potential future GPU optimizations)
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

def segment_image(image: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """
    Segment the input image using a pre-trained YOLO model.
    
    Args:
        image: Input image as a numpy array (uint8 format expected)
        use_gpu: Whether to use GPU acceleration if available (passed to YOLO)
    
    Returns:
        numpy.ndarray: Segmented image mask (binary uint8 format)
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a numpy array of dtype uint8")
    
    # Run segmentation pipeline with original numpy array
    # Let the YOLO pipeline handle GPU acceleration internally
    if use_gpu and HAS_CUPY:
        print("GPU acceleration available - YOLO pipeline will use GPU if configured")
    elif use_gpu and not HAS_CUPY:
        print("GPU requested but CuPy not available, using CPU")
    
    # Always pass numpy array to segmentation pipeline
    segmented_mask = segmentation_pipeline_yolo(image)

    return segmented_mask

def bulk_segment_images(images: list, use_gpu: bool = True, verbose: bool = True) -> list:
    """
    Segment a list of images using a pre-trained YOLO model.
    
    Args:
        images: List of input images as numpy arrays (uint8 format expected)
        use_gpu: Whether to use GPU acceleration if available
        verbose: Whether to print progress information
    
    Returns:
        list: List of segmented image masks (binary uint8 format)
    """
    segmented_masks = []
    
    if verbose:
        gpu_status = "GPU" if (use_gpu and HAS_CUPY) else "CPU"
        print(f"Segmenting {len(images)} images using {gpu_status}")
    
    for i, img in enumerate(images):
        if img is None:
            print(f"Skipping None image at index {i}")
            segmented_masks.append(None)
            continue
            
        try:
            mask = segment_image(img, use_gpu=use_gpu)
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