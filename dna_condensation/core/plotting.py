import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Union


def plot_image(image: np.ndarray, 
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


def plot_image_mask(image: np.ndarray, 
                   mask: np.ndarray,
                   title: Optional[str] = None,
                   figsize: Tuple[int, int] = (15, 6),
                   alpha: float = 0.5) -> None:
    """
    Plot an image alongside its segmentation mask for visualization.
    
    Args:
        image: Original image as numpy array (uint8 format expected)
        mask: Segmentation mask as numpy array (binary or labeled)
        title: Optional main title for the figure
        figsize: Figure size as (width, height) tuple
        alpha: Transparency for overlay (0.0 = transparent, 1.0 = opaque)
    
    Raises:
        ValueError: If image and mask shapes don't match
        TypeError: If image dtype is not uint8
    """
    # Validate inputs
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    if not isinstance(mask, np.ndarray):
        raise ValueError("Mask must be a numpy array")
    
    if image.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {image.dtype}")
    
    # Handle different image dimensions for comparison
    img_shape = image.shape[:2]  # Get height, width only
    mask_shape = mask.shape[:2] if mask.ndim >= 2 else mask.shape
    
    if img_shape != mask_shape:
        raise ValueError(f"Image shape {img_shape} doesn't match mask shape {mask_shape}")
    
    # Prepare image for display
    if image.ndim == 3 and image.shape[2] == 1:
        display_image = image.squeeze()
    else:
        display_image = image
    
    # Prepare mask for display
    display_mask = mask.squeeze() if mask.ndim > 2 else mask
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Original image
    if display_image.ndim == 3 and display_image.shape[2] in [3, 4]:
        # RGB/RGBA image
        axes[0].imshow(display_image)
    else:
        # Grayscale image
        axes[0].imshow(display_image, cmap='gray')
    
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot 2: Segmentation mask
    # Use different colors for different labels if it's a labeled mask
    if display_mask.max() > 1:
        # Multi-label mask
        im_mask = axes[1].imshow(display_mask, cmap='tab10')
        plt.colorbar(im_mask, ax=axes[1], label='Object ID')
    else:
        # Binary mask
        axes[1].imshow(display_mask, cmap='Reds', alpha=0.8)
    
    axes[1].set_title('Segmentation Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Plot 3: Overlay
    if display_image.ndim == 3 and display_image.shape[2] in [3, 4]:
        # RGB/RGBA image
        axes[2].imshow(display_image)
    else:
        # Grayscale image
        axes[2].imshow(display_image, cmap='gray')
    
    # Overlay mask with transparency
    if display_mask.max() > 1:
        # Multi-label mask - use different colors
        axes[2].imshow(display_mask, cmap='tab10', alpha=alpha)
    else:
        # Binary mask - use red overlay
        axes[2].imshow(display_mask, cmap='Reds', alpha=alpha)
    
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add main title if provided
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Display the plot
    plt.show()


def plot_preprocessing_comparison(original_image: np.ndarray,
                                global_preprocessed: np.ndarray,
                                per_nucleus_preprocessed: np.ndarray,
                                labels: np.ndarray,
                                title: str = "Preprocessing Comparison",
                                figsize: Tuple[int, int] = (18, 12),
                                save_path: Optional[str] = None) -> None:
    """
    Compare original, globally preprocessed, and per-nucleus preprocessed images.
    
    Args:
        original_image: Original image after z-collapse
        global_preprocessed: Image after global preprocessing (background correction + normalization)
        per_nucleus_preprocessed: Image after per-nucleus normalization
        labels: Segmentation labels for overlay
        title: Title for the entire figure
        figsize: Figure size as (width, height) tuple
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Ensure all images are 2D for display
    orig_2d = original_image.squeeze() if original_image.ndim > 2 else original_image
    global_2d = global_preprocessed.squeeze() if global_preprocessed.ndim > 2 else global_preprocessed
    per_nucleus_2d = per_nucleus_preprocessed.squeeze() if per_nucleus_preprocessed.ndim > 2 else per_nucleus_preprocessed
    
    # Top row: Raw images
    axes[0, 0].imshow(orig_2d, cmap='gray')
    axes[0, 0].set_title('Original (Z-collapsed)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(global_2d, cmap='gray')
    axes[0, 1].set_title('Global Preprocessing\n(Background + Intensity Norm)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(per_nucleus_2d, cmap='gray')
    axes[0, 2].set_title('Per-Nucleus Normalization\n(Mean = 1.0 per nucleus)')
    axes[0, 2].axis('off')
    
    # Bottom row: With segmentation overlays
    # Original with overlay
    overlay_orig = np.stack([orig_2d, orig_2d, orig_2d], axis=2)
    boundaries = (labels > 0).astype(bool)
    if overlay_orig.dtype == np.uint8:
        overlay_orig[boundaries] = [255, 100, 100]  # Red boundaries
    else:
        max_val = overlay_orig.max()
        overlay_orig[boundaries] = [max_val, max_val*0.4, max_val*0.4]
    
    axes[1, 0].imshow(overlay_orig)
    axes[1, 0].set_title(f'Original + Segmentation\n({len(np.unique(labels))-1} nuclei)')
    axes[1, 0].axis('off')
    
    # Global preprocessed with overlay
    overlay_global = np.stack([global_2d, global_2d, global_2d], axis=2)
    if overlay_global.dtype == np.uint8:
        overlay_global[boundaries] = [255, 100, 100]
    else:
        max_val = overlay_global.max()
        overlay_global[boundaries] = [max_val, max_val*0.4, max_val*0.4]
    
    axes[1, 1].imshow(overlay_global)
    axes[1, 1].set_title('Global Preprocessing + Segmentation')
    axes[1, 1].axis('off')
    
    # Per-nucleus with intensity statistics
    overlay_per_nucleus = np.stack([per_nucleus_2d, per_nucleus_2d, per_nucleus_2d], axis=2)
    if overlay_per_nucleus.dtype == np.uint8:
        overlay_per_nucleus[boundaries] = [255, 100, 100]
    else:
        max_val = overlay_per_nucleus.max()
        overlay_per_nucleus[boundaries] = [max_val, max_val*0.4, max_val*0.4]
    
    axes[1, 2].imshow(overlay_per_nucleus)
    
    # Calculate some statistics for per-nucleus image
    nucleus_ids = np.unique(labels)[1:]  # Skip background
    cvs = []
    for nid in nucleus_ids[:10]:  # Sample first 10 nuclei for stats
        nucleus_pixels = per_nucleus_2d[labels == nid]
        if len(nucleus_pixels) > 0:
            cv = np.std(nucleus_pixels) / np.mean(nucleus_pixels) if np.mean(nucleus_pixels) > 0 else 0
            cvs.append(cv)
    
    mean_cv = np.mean(cvs) if cvs else 0
    axes[1, 2].set_title(f'Per-Nucleus + Segmentation\n(Sample CV: {mean_cv:.3f})')
    axes[1, 2].axis('off')
    
    plt.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved preprocessing comparison to: {save_path}")
    
    plt.show()
    