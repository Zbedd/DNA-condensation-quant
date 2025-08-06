import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Union


def plot(image: np.ndarray, 
         title: Optional[str] = None,
         figsize: Tuple[int, int] = (10, 8),
         cmap: str = 'gray',
         show_colorbar: bool = True,
         vmin: Optional[float] = None,
         vmax: Optional[float] = None) -> None:
    """
    Plot a uint8 image using matplotlib.
    
    Args:
        image: Input image as numpy array (uint8 format expected)
        title: Optional title for the plot
        figsize: Figure size as (width, height) tuple
        cmap: Colormap to use for display ('gray', 'viridis', 'plasma', etc.)
        show_colorbar: Whether to show colorbar
        vmin: Minimum value for colormap scaling (None = auto)
        vmax: Maximum value for colormap scaling (None = auto)
    
    Raises:
        ValueError: If image is not a valid numpy array
        TypeError: If image dtype is not uint8
    """
    # Validate input
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if image.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {image.dtype}")
    
    # Handle different image dimensions
    if image.ndim == 2:
        # Grayscale image
        display_image = image
    elif image.ndim == 3:
        if image.shape[2] == 1:
            # Single channel image
            display_image = image.squeeze()
        elif image.shape[2] == 3:
            # RGB image
            display_image = image
            cmap = None  # Don't use colormap for RGB
        elif image.shape[2] == 4:
            # RGBA image
            display_image = image
            cmap = None  # Don't use colormap for RGBA
        else:
            raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Display the image
    im = plt.imshow(display_image, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add title if provided
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar for grayscale images
    if show_colorbar and cmap is not None:
        plt.colorbar(im, label='Intensity')
    
    # Remove axes for cleaner display
    plt.axis('off')
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Display the plot
    plt.show()


def plot_multiple(images: list, 
                  titles: Optional[list] = None,
                  figsize: Tuple[int, int] = (15, 10),
                  cmap: str = 'gray',
                  rows: Optional[int] = None,
                  cols: Optional[int] = None) -> None:
    """
    Plot multiple uint8 images in a grid layout.
    
    Args:
        images: List of numpy arrays (uint8 format expected)
        titles: Optional list of titles for each image
        figsize: Figure size as (width, height) tuple
        cmap: Colormap to use for display
        rows: Number of rows in grid (auto-calculated if None)
        cols: Number of columns in grid (auto-calculated if None)
    """
    n_images = len(images)
    
    if n_images == 0:
        raise ValueError("No images provided")
    
    # Auto-calculate grid dimensions if not provided
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    elif rows is None:
        rows = int(np.ceil(n_images / cols))
    elif cols is None:
        cols = int(np.ceil(n_images / rows))
    
    # Create subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single subplot case
    if n_images == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each image
    for i, image in enumerate(images):
        if i >= len(axes):
            break
            
        # Validate image
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            axes[i].text(0.5, 0.5, f'Invalid image {i}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            continue
        
        # Display image
        display_image = image.squeeze() if image.ndim == 3 and image.shape[2] == 1 else image
        use_cmap = None if (image.ndim == 3 and image.shape[2] in [3, 4]) else cmap
        
        axes[i].imshow(display_image, cmap=use_cmap)
        axes[i].axis('off')
        
        # Add title if provided
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=12)
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()