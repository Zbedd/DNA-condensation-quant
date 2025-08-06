import numpy as np
from typing import Union, Optional
from nd2reader import ND2Reader


def collapse_z_axis(nd2_image: Union[ND2Reader, np.ndarray], 
                    method: str = 'mean',
                    channel: Optional[int] = None,
                    timepoint: Optional[int] = None,
                    verbose: bool = False) -> np.ndarray:
    """
    Collapse an ND2 hyperstack image along the z-axis to create a 2D projection.
    
    Args:
        nd2_image: ND2Reader object or numpy array with z-dimension
        method: Projection method ('max', 'mean', 'sum', 'median')
        channel: Specific channel to process (None = all channels)
        timepoint: Specific timepoint to process (None = first timepoint)
        verbose: Whether to print detailed debug information
    
    Returns:
        numpy.ndarray: 2D or 3D array (depending on channels) with z-axis collapsed
        
    Raises:
        ValueError: If method is not supported or no z-dimension found
        IndexError: If specified channel/timepoint doesn't exist
    """
    # Supported projection methods
    projection_methods = {
        'max': np.max,
        'mean': np.mean,
        'sum': np.sum,
        'median': np.median
    }
    
    if method not in projection_methods:
        raise ValueError(f"Unsupported method '{method}'. Choose from: {list(projection_methods.keys())}")
    
    # Convert ND2Reader to numpy array if needed
    if isinstance(nd2_image, ND2Reader):
        if verbose:
            print(f"Converting ND2 hyperstack to numpy array...")
        try:
            # Get raw array without automatic squeezing
            image_data = np.array(nd2_image)
            if verbose:
                print(f"Initial converted shape: {image_data.shape}")
                print(f"Number of dimensions: {image_data.ndim}")
                print(f"Array dtype: {image_data.dtype}")
            
            # Check ND2 metadata to understand original structure
            if hasattr(nd2_image, 'sizes') and verbose:
                print(f"Original ND2 sizes: {nd2_image.sizes}")
                
            # If we have singleton dimensions that were squeezed, we might need to handle them
            expected_dims = len(getattr(nd2_image, 'sizes', {}))
            actual_dims = image_data.ndim
            if expected_dims != actual_dims and verbose:
                print(f"WARNING: Expected {expected_dims} dimensions but got {actual_dims}")
                print("This might indicate singleton dimensions were automatically squeezed")
                
        except Exception as e:
            print(f"Error converting ND2 to array: {e}")
            raise
    else:
        image_data = nd2_image
    
    # Validate input array
    if not isinstance(image_data, np.ndarray):
        raise ValueError("Input must be ND2Reader object or numpy array")
    
    # Find z-axis dimension
    z_axis = _find_z_axis(image_data, nd2_image if isinstance(nd2_image, ND2Reader) else None, verbose=verbose)
    
    if z_axis is None:
        raise ValueError("No z-dimension found in image data")
    
    # Additional validation before applying projection
    if verbose:
        print(f"About to collapse axis {z_axis} from shape {image_data.shape}")
    if z_axis >= image_data.ndim:
        raise ValueError(f"Z-axis index {z_axis} is out of bounds for array with {image_data.ndim} dimensions")
    
    # Apply projection method along z-axis
    projection_func = projection_methods[method]
    try:
        collapsed_image = projection_func(image_data, axis=z_axis)
        if verbose:
            print(f"Collapsed to shape: {collapsed_image.shape}")
    except Exception as e:
        print(f"Error during {method} projection along axis {z_axis}: {e}")
        print(f"Array shape: {image_data.shape}")
        print(f"Array dtype: {image_data.dtype}")
        raise
    
    # Convert to uint8 if necessary
    if collapsed_image.dtype != np.uint8:
        collapsed_image = _normalize_to_uint8(collapsed_image)
    
    return collapsed_image


def _find_z_axis(image_data: np.ndarray, nd2_reader: Optional[ND2Reader] = None, verbose: bool = False) -> Optional[int]:
    """
    Find which axis corresponds to the z-dimension by inspecting ND2 metadata.
    Handles dimensions of size 1 that might be squeezed out.
    
    Args:
        image_data: Image array
        nd2_reader: Optional ND2Reader for metadata
        verbose: Whether to print detailed debug information
    
    Returns:
        int: Axis index for z-dimension, or None if not found
    """
    if verbose:
        print(f"=== Z-AXIS DETECTION DEBUG ===")
        print(f"Image data shape: {image_data.shape}")
        print(f"Image data ndim: {image_data.ndim}")
    
    if nd2_reader:
        # Check for axes attribute
        if hasattr(nd2_reader, 'axes'):
            axes = nd2_reader.axes
            if verbose:
                print(f"ND2 axes order: {axes}")
            
            # Check for sizes to understand dimension mapping
            if hasattr(nd2_reader, 'sizes'):
                sizes = nd2_reader.sizes
                if verbose:
                    print(f"ND2 sizes: {sizes}")
                
                # Filter out dimensions of size 1 (they might be squeezed)
                non_singleton_dims = [(dim, size) for dim, size in sizes.items() if size > 1]
                if verbose:
                    print(f"Non-singleton dimensions: {non_singleton_dims}")
                
                # Find z in non-singleton dimensions
                if 'z' in [dim for dim, size in non_singleton_dims] and 'z' in sizes and sizes['z'] > 1:
                    # Map z position considering only non-singleton dimensions
                    z_size = sizes['z']
                    if verbose:
                        print(f"Looking for z-dimension with size {z_size}")
                    
                    # Find which axis in the current array has the z-size
                    for i, dim_size in enumerate(image_data.shape):
                        if dim_size == z_size:
                            if verbose:
                                print(f"Found z-dimension at axis {i} with size {dim_size}")
                            return i
                    
                    if verbose:
                        print(f"Could not find axis with z-size {z_size} in shape {image_data.shape}")
            
            # Fallback: try original axes approach but with validation
            if 'z' in axes:
                z_position = axes.index('z')
                if verbose:
                    print(f"Z-axis found at metadata position: {z_position}")
                
                # But adjust for potentially squeezed dimensions
                if z_position < image_data.ndim:
                    if verbose:
                        print(f"Z-axis position {z_position} is within array bounds")
                    return z_position
                else:
                    if verbose:
                        print(f"WARNING: Z-axis metadata position {z_position} exceeds array dimensions ({image_data.ndim})")
                        print("Attempting to find z-axis by size matching...")
        else:
            if verbose:
                print("ND2 object lacks 'axes' attribute")
            
        # Size-based detection as backup
        if hasattr(nd2_reader, 'sizes'):
            sizes = nd2_reader.sizes
            if verbose:
                print(f"ND2 sizes: {sizes}")
            
            if 'z' in sizes and sizes['z'] > 1:
                z_size = sizes['z']
                if verbose:
                    print(f"Looking for z-dimension with size {z_size}")
                
                # Find which axis matches the z-size
                for i, dim_size in enumerate(image_data.shape):
                    if dim_size == z_size:
                        if verbose:
                            print(f"Found matching z-dimension at axis {i}")
                        return i
    
    # Final fallback: look for dimension that could be z-stack
    if verbose:
        print("Using fallback z-axis detection...")
    if image_data.ndim >= 3:
        # Look for a dimension that could be z (typically > 1 but reasonable for z-stack)
        for i, dim_size in enumerate(image_data.shape):
            if 2 <= dim_size <= 100:  # Reasonable z-stack range
                if verbose:
                    print(f"Fallback: assuming axis {i} (size {dim_size}) is z-axis")
                return i
    
    if verbose:
        print("No z-dimension detected")
        print("=== END Z-AXIS DEBUG ===")
    return None

def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to uint8 format.
    
    Args:
        image: Input image array
    
    Returns:
        numpy.ndarray: Image normalized to uint8 (0-255)
    """
    # Handle different input types
    if image.dtype == np.uint8:
        return image
    
    # Normalize to 0-1 range
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
    else:
        normalized = np.zeros_like(image, dtype=np.float64)
    
    # Scale to 0-255 and convert to uint8
    return (normalized * 255).astype(np.uint8)


def batch_collapse_z_axis(nd2_readers: list, 
                         method: str = 'max',
                         channel: Optional[int] = None,
                         verbose: bool = True) -> list:
    """
    Collapse z-axis for multiple ND2 files.
    
    Args:
        nd2_readers: List of ND2Reader objects
        method: Projection method ('max', 'mean', 'sum', 'median')
        channel: Specific channel to process (None = all channels)
        verbose: Whether to print detailed debug information
    
    Returns:
        list: List of collapsed images (numpy arrays)
    """
    collapsed_images = []
    
    for i, nd2_reader in enumerate(nd2_readers):
        if verbose:
            print(f"\n=== Processing file {i+1}/{len(nd2_readers)} ===")
            print(f"File: {getattr(nd2_reader, 'filename', 'Unknown')}")
            
        try:
            collapsed = collapse_z_axis(nd2_reader, method=method, channel=channel, verbose=verbose)
            collapsed_images.append(collapsed)
            
            if verbose:
                print(f"✓ Successfully processed file {i+1}/{len(nd2_readers)}")
            else:
                print(f"Processed {i+1}/{len(nd2_readers)}: {getattr(nd2_reader, 'filename', 'Unknown')}")
                
        except Exception as e:
            print(f"✗ Error processing file {i+1}: {e}")
            collapsed_images.append(None)
    
    return collapsed_images