# Preprocessing Functions Implementation Summary

## Overview
Added three preprocessing functions to `dna_condensation/core/segmentation.py` following the same bulk architecture as the segmentation functions.

## Functions Implemented

### 1. Deconvolution
**Function**: `preprocess_image_deconvolution()`
- **Purpose**: Sharpen features and reduce out-of-focus haze
- **Method**: Richardson-Lucy deconvolution with Gaussian PSF
- **Parameters**:
  - `sigma`: Standard deviation for PSF (default: 1.0)
  - `iterations`: Number of RL iterations (default: 10)
- **Note**: For better results, consider specialized tools like Huygens or DeconvLab2

### 2. Background Correction
**Function**: `preprocess_image_background_correction()`
- **Purpose**: Remove uneven illumination using rolling ball subtraction
- **Method**: Rolling ball background subtraction from scikit-image
- **Parameters**:
  - `ball_radius`: Radius of rolling ball (default: 50)
- **Benefits**: Corrects for uneven lighting and field illumination

### 3. Intensity Normalization
**Function**: `preprocess_image_intensity_normalization()`
- **Purpose**: Normalize intensities to correct for staining/loading variations
- **Methods Available**:
  - `'percentile'`: Robust normalization using percentile clipping (default)
  - `'zscore'`: Z-score normalization
  - `'minmax'`: Min-max scaling
  - `'target_mean'`: Scale to achieve target mean intensity
- **Parameters**:
  - `method`: Normalization method
  - `percentile_range`: (low, high) percentiles for clipping (default: (1, 99))
  - `target_mean`: Target mean for target_mean method (default: 128.0)

### 4. Bulk Processing
**Function**: `bulk_preprocess_images()`
- **Purpose**: Apply preprocessing pipeline to multiple images
- **Features**:
  - Process multiple images efficiently
  - Chain multiple preprocessing methods
  - Progress reporting
  - Error handling
  - Configurable parameters for each method

## Usage Examples

### Single Method
```python
# Background correction only
processed = bulk_preprocess_images(
    images, 
    channel_index=1,
    methods=['background_correction'],
    bg_ball_radius=50
)
```

### Combined Pipeline
```python
# Full preprocessing pipeline
processed = bulk_preprocess_images(
    images, 
    channel_index=1,
    methods=['background_correction', 'deconvolution', 'intensity_normalization'],
    bg_ball_radius=50,
    deconv_sigma=1.0,
    deconv_iterations=10,
    norm_method='percentile',
    norm_percentile_range=(1, 99)
)
```

## Integration with Existing Workflow
- Same architecture as `bulk_segment_images()`
- Compatible with existing image loading and z-stack handling
- Works with same image formats (uint8 numpy arrays)
- Supports both 2D and 3D multi-channel images
- Channel-specific processing using `channel_index`

## Testing Results
✅ All preprocessing functions tested successfully:
- Background correction: Reduces mean intensity and dynamic range
- Intensity normalization: Stretches contrast to full 0-255 range
- Deconvolution: Enhances sharpness while preserving structure
- Combined pipeline: Sequential application works correctly

## Performance Notes
- Background correction: Fast, minimal computational overhead
- Intensity normalization: Very fast, simple statistical operations
- Deconvolution: More computationally intensive, scales with iterations
- Memory efficient: Processes one image at a time
- Progress reporting for large batches

## Future Enhancements
- GPU acceleration for deconvolution using CuPy
- Integration with specialized deconvolution libraries
- Additional normalization methods (histogram matching, etc.)
- Automatic parameter optimization based on image characteristics
