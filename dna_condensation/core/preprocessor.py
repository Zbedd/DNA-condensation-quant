import numpy as np
from scipy import ndimage
from skimage.restoration import rolling_ball
from skimage.morphology import disk
import cv2
from typing import List, Tuple, Optional, Union, Dict
import os

# ==== PREPROCESSING FUNCTIONS FOR HOMOGENEITY ANALYSIS ====

"""
RECOMMENDED PREPROCESSING ORDER FOR HOMOGENEITY ANALYSIS

The order of preprocessing steps significantly affects the quality of homogeneity measurements.
Follow this recommended sequence for optimal results:

STANDARD WORKFLOW (for comparing images across experiments):
1. deconvolution() [OPTIONAL] - Apply first to sharpen features before other corrections
2. background_correction() [REQUIRED] - Remove spatial illumination artifacts 
3. intensity_normalization() [RECOMMENDED] - Standardize intensity ranges across images

PER-NUCLEUS WORKFLOW (for within-nucleus homogeneity analysis):
1. deconvolution() [OPTIONAL] - Apply first if needed
2. background_correction() [REQUIRED] - Critical for accurate within-nucleus measurements
3. segmentation() [REQUIRED] - Generate nucleus labels (not in this module)
4. per_nucleus_intensity_normalization() [RECOMMENDED] - Use INSTEAD of global intensity_normalization

WHY THIS ORDER MATTERS:

→ Deconvolution first: Works best on raw data before other intensity modifications
→ Background correction second: Removes spatial artifacts that would bias all downstream analysis
→ Intensity normalization last: Standardizes already-corrected images for comparison

AVOID THESE COMMON MISTAKES:
✗ Background correction after intensity normalization (loses spatial context)
✗ Deconvolution after background correction (operates on modified intensities)
✗ Using both global intensity_normalization AND per_nucleus_normalization (redundant)

EXAMPLE USAGE:

# Standard workflow for multi-image comparison:
preprocessed = bulk_preprocess_images(
    images, channel_index=0,
    methods=['background_correction', 'intensity_normalization'],
    bg_ball_radius=50, norm_method='percentile'
)

# Per-nucleus workflow for homogeneity analysis:
bg_corrected = bulk_preprocess_images(
    images, channel_index=0, 
    methods=['background_correction'],  # Skip global normalization!
    bg_ball_radius=50
)
# ... perform segmentation to get labels ...
normalized_image, stats = per_nucleus_intensity_normalization(
    bg_corrected[0], labels, target_mean=1.0
)

PARAMETER RECOMMENDATIONS:
- bg_ball_radius: Set to ~1.5x average nucleus diameter (typically 30-100 pixels)
- norm_method: Use 'percentile' (robust) or 'target_mean' (preserves relative intensities)
- deconv_iterations: Start with 5-10 iterations, increase if needed
- target_mean: Use 1.0 for per-nucleus normalization (makes CV calculations intuitive)

"""

def deconvolution(image: np.ndarray, channel_index: int, sigma: float = 1.0, iterations: int = 10) -> np.ndarray:
    """
    Apply Richardson-Lucy deconvolution to sharpen features and reduce out-of-focus haze.
    This is a simplified deconvolution using scipy - for better results consider using
    specialized tools like Huygens or DeconvLab2.
    
    PROCESSING ORDER: Apply FIRST in preprocessing pipeline (before background correction)
    
    Args:
        image: Input image as numpy array (uint8 format expected)
        channel_index: Index of the channel to process (0-based)
        sigma: Standard deviation for the PSF Gaussian kernel
        iterations: Number of Richardson-Lucy iterations
    
    Returns:
        numpy.ndarray: Deconvolved image in uint8 format
        
    Recommended Usage:
        Apply first to raw images before other preprocessing steps.
        Deconvolution works best on unmodified intensity data.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a numpy array of dtype uint8")
    
    # Extract specific channel for processing
    if image.ndim == 3:
        if channel_index < 0 or channel_index >= image.shape[2]:
            raise ValueError(f"Channel index {channel_index} is out of range for image with {image.shape[2]} channels")
        processed_image = image[:, :, channel_index].astype(np.float64)
    elif image.ndim == 2:
        processed_image = image.astype(np.float64)
        if channel_index != 0:
            print(f"Warning: channel_index={channel_index} specified for 2D image, using single channel")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    try:
        # Normalize to 0-1 range
        processed_image = processed_image / 255.0
        
        # Create a simple Gaussian PSF (Point Spread Function)
        psf_size = int(6 * sigma + 1)
        if psf_size % 2 == 0:
            psf_size += 1
        
        y, x = np.ogrid[-psf_size//2:psf_size//2+1, -psf_size//2:psf_size//2+1]
        psf = np.exp(-(x*x + y*y) / (2.0 * sigma**2))
        psf = psf / psf.sum()
        
        # Richardson-Lucy deconvolution
        from scipy.signal import convolve2d
        
        # Initialize estimate
        estimate = processed_image.copy()
        
        for i in range(iterations):
            # Forward convolution: simulate how the true image would look with PSF blur
            convolved = convolve2d(estimate, psf, mode='same', boundary='symm')
            
            # Avoid division by zero in ratio calculation
            convolved = np.maximum(convolved, 1e-10)
            
            # Calculate correction ratio (observed / simulated intensities)
            ratio = processed_image / convolved
            
            # Backward convolution with flipped PSF to propagate corrections
            psf_flipped = np.flip(psf)
            correction = convolve2d(ratio, psf_flipped, mode='same', boundary='symm')
            
            # Apply multiplicative correction to refine estimate
            estimate = estimate * correction
            
            # Ensure non-negative values (physical constraint)
            estimate = np.maximum(estimate, 0)
        
        # Convert back to uint8
        result = np.clip(estimate * 255, 0, 255).astype(np.uint8)
        
        # Restore original shape if multi-channel
        if image.ndim == 3:
            output = image.copy()
            output[:, :, channel_index] = result
            return output
        else:
            return result
            
    except Exception as e:
        print(f"Deconvolution failed: {e}")
        return image.copy()

def background_correction(image: np.ndarray, channel_index: int, ball_radius: int = 50) -> np.ndarray:
    """
    Apply rolling ball background subtraction to remove uneven illumination.
    
    PROCESSING ORDER: Apply SECOND in preprocessing pipeline (after deconvolution, before normalization)
    IMPORTANCE: CRITICAL for homogeneity analysis - spatial artifacts will appear as false heterogeneity
    
    Args:
        image: Input image as numpy array (uint8 format expected)
        channel_index: Index of the channel to process (0-based)
        ball_radius: Radius of the rolling ball for background estimation
                    Recommendation: ~1.5x average nucleus diameter (typically 30-100 pixels)
    
    Returns:
        numpy.ndarray: Background-corrected image in uint8 format
        
    Recommended Usage:
        Essential step before any intensity-based analysis. Set ball_radius to be larger
        than your largest nucleus to avoid removing actual biological signal.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a numpy array of dtype uint8")
    
    # Extract specific channel for processing
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
        # Apply rolling ball background subtraction
        # Note: rolling_ball expects float input, returns float output
        processed_float = processed_image.astype(np.float64)
        
        # Create rolling ball structuring element
        ball = disk(ball_radius)
        
        # Estimate background using rolling ball
        background = rolling_ball(processed_float, radius=ball_radius)
        
        # Subtract background
        corrected = processed_float - background
        
        # Ensure non-negative values and convert back to uint8
        corrected = np.maximum(corrected, 0)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        # Restore original shape if multi-channel
        if image.ndim == 3:
            output = image.copy()
            output[:, :, channel_index] = corrected
            return output
        else:
            return corrected
            
    except Exception as e:
        print(f"Background correction failed: {e}")
        return image.copy()

def intensity_normalization(image: np.ndarray, channel_index: int, method: str = 'percentile', 
                                           percentile_range: tuple = (1, 99), target_mean: float = 128.0) -> np.ndarray:
    """
    Apply intensity normalization to correct for staining/loading variations.
    
    PROCESSING ORDER: Apply LAST in standard preprocessing pipeline (after background correction)
    ALTERNATIVE: Use per_nucleus_intensity_normalization() instead for homogeneity analysis
    
    Args:
        image: Input image as numpy array (uint8 format expected)
        channel_index: Index of the channel to process (0-based)
        method: Normalization method ('percentile', 'zscore', 'minmax', 'target_mean')
        percentile_range: For percentile method, the (low, high) percentiles to clip
        target_mean: For target_mean method, the desired mean intensity
    
    Returns:
        numpy.ndarray: Intensity-normalized image in uint8 format
        
    Recommended Usage:
        Use for comparing images across experiments. For homogeneity analysis within nuclei,
        prefer per_nucleus_intensity_normalization() after segmentation instead.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Input image must be a numpy array of dtype uint8")
    
    # Extract specific channel for processing
    if image.ndim == 3:
        if channel_index < 0 or channel_index >= image.shape[2]:
            raise ValueError(f"Channel index {channel_index} is out of range for image with {image.shape[2]} channels")
        processed_image = image[:, :, channel_index].astype(np.float64)
    elif image.ndim == 2:
        processed_image = image.astype(np.float64)
        if channel_index != 0:
            print(f"Warning: channel_index={channel_index} specified for 2D image, using single channel")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    try:
        if method == 'percentile':
            # Percentile-based normalization (robust to outliers)
            low_val = np.percentile(processed_image, percentile_range[0])
            high_val = np.percentile(processed_image, percentile_range[1])
            
            # Clip and rescale
            normalized = np.clip(processed_image, low_val, high_val)
            normalized = (normalized - low_val) / (high_val - low_val) * 255
            
        elif method == 'zscore':
            # Z-score normalization
            mean_val = np.mean(processed_image)
            std_val = np.std(processed_image)
            
            if std_val > 0:
                normalized = (processed_image - mean_val) / std_val
                # Rescale to 0-255 range (keep 3 standard deviations)
                normalized = np.clip(normalized, -3, 3)
                normalized = (normalized + 3) / 6 * 255
            else:
                normalized = processed_image
                
        elif method == 'minmax':
            # Min-max normalization
            min_val = np.min(processed_image)
            max_val = np.max(processed_image)
            
            if max_val > min_val:
                normalized = (processed_image - min_val) / (max_val - min_val) * 255
            else:
                normalized = processed_image
                
        elif method == 'target_mean':
            # Normalize to achieve target mean while preserving relative intensities
            current_mean = np.mean(processed_image)
            
            if current_mean > 0:
                scale_factor = target_mean / current_mean
                normalized = processed_image * scale_factor
                normalized = np.clip(normalized, 0, 255)
            else:
                normalized = processed_image
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Convert back to uint8
        result = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # Restore original shape if multi-channel
        if image.ndim == 3:
            output = image.copy()
            output[:, :, channel_index] = result
            return output
        else:
            return result
            
    except Exception as e:
        print(f"Intensity normalization failed: {e}")
        return image.copy()

def bulk_preprocess_images(images: list, channel_index: int, methods: list = ['background_correction'], 
                          verbose: bool = True, **kwargs) -> list:
    """
    Apply preprocessing to a list of images using specified methods.
    
    RECOMMENDED METHOD ORDER: ['deconvolution', 'background_correction', 'intensity_normalization']
    FOR HOMOGENEITY ANALYSIS: ['background_correction'] only, then use per_nucleus_intensity_normalization()
    
    Args:
        images: List of input images as numpy arrays (uint8 format expected)
        channel_index: Index of the channel to process (0-based)
        methods: List of preprocessing methods to apply in order
                Available: 'deconvolution', 'background_correction', 'intensity_normalization'
                Recommended order: Apply deconvolution first, background_correction second, 
                intensity_normalization last
        verbose: Whether to print progress information
        **kwargs: Additional parameters for specific preprocessing methods:
                 - deconv_sigma: Standard deviation for deconvolution PSF (default: 1.0)
                 - deconv_iterations: Iterations for deconvolution (default: 10)
                 - bg_ball_radius: Rolling ball radius for background correction (default: 50)
                 - norm_method: Normalization method (default: 'percentile')
                 - norm_percentile_range: Percentile range for normalization (default: (1, 99))
                 - norm_target_mean: Target mean for target_mean normalization (default: 128.0)
    
    Returns:
        list: List of preprocessed images
        
    Example Usage:
        # Standard workflow
        preprocessed = bulk_preprocess_images(
            images, channel_index=0,
            methods=['background_correction', 'intensity_normalization']
        )
        
        # For homogeneity analysis (skip global normalization)
        bg_corrected = bulk_preprocess_images(
            images, channel_index=0,
            methods=['background_correction']
        )
    """
    if not methods:
        if verbose:
            print("No preprocessing methods specified, returning original images")
        return [img.copy() if img is not None else None for img in images]
    
    # Extract kwargs for each method
    deconv_sigma = kwargs.get('deconv_sigma', 1.0)
    deconv_iterations = kwargs.get('deconv_iterations', 10)
    bg_ball_radius = kwargs.get('bg_ball_radius', 50)
    norm_method = kwargs.get('norm_method', 'percentile')
    norm_percentile_range = kwargs.get('norm_percentile_range', (1, 99))
    norm_target_mean = kwargs.get('norm_target_mean', 128.0)
    
    if verbose:
        methods_str = " → ".join(methods)
        print(f"Preprocessing {len(images)} images: {methods_str} (channel {channel_index})")
    
    processed_images = []
    
    for i, img in enumerate(images):
        if img is None:
            print(f"Skipping None image at index {i}")
            processed_images.append(None)
            continue
            
        try:
            current_img = img.copy()
            
            # Apply each preprocessing method in sequence
            for method in methods:
                if method == 'deconvolution':
                    current_img = deconvolution(
                        current_img, channel_index, sigma=deconv_sigma, iterations=deconv_iterations
                    )
                elif method == 'background_correction':
                    current_img = background_correction(
                        current_img, channel_index, ball_radius=bg_ball_radius
                    )
                elif method == 'intensity_normalization':
                    current_img = intensity_normalization(
                        current_img, channel_index, method=norm_method, 
                        percentile_range=norm_percentile_range, target_mean=norm_target_mean
                    )
                else:
                    print(f"Warning: Unknown preprocessing method '{method}', skipping")
            
            processed_images.append(current_img)
            
            if verbose and (i + 1) % 5 == 0:  # Progress update every 5 images
                print(f"Preprocessed {i + 1}/{len(images)} images")
                
        except Exception as e:
            print(f"Error preprocessing image {i + 1}: {e}")
            processed_images.append(img.copy() if img is not None else None)
    
    if verbose:
        successful = sum(1 for img in processed_images if img is not None)
        print(f"Successfully preprocessed {successful}/{len(images)} images")
    
    return processed_images


def per_nucleus_intensity_normalization(image: np.ndarray, 
                                      labels: np.ndarray, 
                                      target_mean: float = 1.0,
                                      noise_floor: float = 0.01,
                                      min_nucleus_size: int = 50,
                                      verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Apply per-nucleus intensity normalization to make intensity-based features
    comparable across nuclei by focusing on distribution patterns rather than absolute brightness.
    
    PROCESSING ORDER: Apply AFTER background_correction() and segmentation, INSTEAD OF intensity_normalization()
    PURPOSE: Optimized for homogeneity analysis within individual nuclei
    
    This function normalizes each segmented nucleus individually by rescaling intensities
    within each nucleus to have a consistent mean value. This is particularly useful for
    homogeneity analysis where you want to compare coefficient of variation (CV) and 
    high-intensity fractions across different nuclei.
    
    Args:
        image: Input image as numpy array (background-corrected intensities recommended)
        labels: Integer label mask of same shape where each pixel is assigned a nucleus ID (0 = background)
        target_mean: Target mean intensity value for each nucleus after normalization (default: 1.0)
                    Recommendation: Use 1.0 for intuitive CV calculations
        noise_floor: Minimum mean intensity threshold - nuclei below this are skipped (default: 0.01)
        min_nucleus_size: Minimum number of pixels required for a nucleus to be processed (default: 50)
        verbose: Whether to print processing information
    
    Returns:
        Tuple containing:
        - normalized_image: numpy.ndarray with per-nucleus normalized intensities (float dtype)
        - stats: dict with normalization statistics per nucleus
                {'nucleus_id': {'original_mean': float, 'scale_factor': float, 'pixel_count': int}}
    
    Recommended Workflow:
        1. Apply background_correction() to raw image
        2. Perform segmentation to generate labels
        3. Apply this function instead of global intensity_normalization()
        4. Proceed with homogeneity measurements (CV, texture analysis, etc.)
    
    Notes:
        - Input image should be background-corrected first for best results
        - Resulting normalized image will have dtype float64 with mean ≈ target_mean per nucleus
        - Use this AFTER standard preprocessing but INSTEAD of global intensity_normalization
        - With target_mean=1.0, CV = std_dev (simplifies downstream calculations)
    """
    if not isinstance(image, np.ndarray) or not isinstance(labels, np.ndarray):
        raise ValueError("Both image and labels must be numpy arrays")
    
    if image.shape != labels.shape:
        raise ValueError(f"Image shape {image.shape} must match labels shape {labels.shape}")
    
    # Convert image to float for processing
    if image.dtype == np.uint8:
        img_float = image.astype(np.float64) / 255.0
    elif image.dtype == np.uint16:
        img_float = image.astype(np.float64) / 65535.0
    else:
        img_float = image.astype(np.float64)
    
    # Initialize output array
    normalized_image = np.zeros_like(img_float, dtype=np.float64)
    
    # Dictionary to store statistics
    stats = {}
    
    # Get unique nucleus labels (excluding background = 0)
    nucleus_ids = np.unique(labels)
    nucleus_ids = nucleus_ids[nucleus_ids > 0]
    
    if verbose:
        print(f"Processing {len(nucleus_ids)} nuclei for per-nucleus normalization")
    
    processed_count = 0
    skipped_count = 0
    
    for nucleus_id in nucleus_ids:
        # Get pixel coordinates for this nucleus
        mask = labels == nucleus_id
        nucleus_pixels = img_float[mask]
        
        # Quality control checks
        pixel_count = len(nucleus_pixels)
        if pixel_count < min_nucleus_size:
            if verbose:
                print(f"Skipping nucleus {nucleus_id}: too small ({pixel_count} pixels)")
            skipped_count += 1
            continue
        
        # Calculate mean intensity
        original_mean = np.mean(nucleus_pixels)
        
        if original_mean <= noise_floor:
            if verbose:
                print(f"Skipping nucleus {nucleus_id}: below noise floor ({original_mean:.4f})")
            skipped_count += 1
            continue
        
        # Calculate normalization factor
        scale_factor = target_mean / original_mean
        
        # Apply normalization
        normalized_pixels = nucleus_pixels * scale_factor
        
        # Write back to output image
        normalized_image[mask] = normalized_pixels
        
        # Store statistics
        stats[int(nucleus_id)] = {
            'original_mean': float(original_mean),
            'scale_factor': float(scale_factor),
            'pixel_count': int(pixel_count),
            'normalized_mean': float(np.mean(normalized_pixels))
        }
        
        processed_count += 1
        
        if verbose and processed_count % 100 == 0:
            print(f"Processed {processed_count} nuclei...")
    
    if verbose:
        print(f"✓ Successfully normalized {processed_count} nuclei")
        print(f"✗ Skipped {skipped_count} nuclei (size/noise filters)")
        
        if processed_count > 0:
            scale_factors = [s['scale_factor'] for s in stats.values()]
            print(f"Scale factor range: {np.min(scale_factors):.3f} - {np.max(scale_factors):.3f}")
            print(f"Mean scale factor: {np.mean(scale_factors):.3f} ± {np.std(scale_factors):.3f}")
    
    return normalized_image, stats