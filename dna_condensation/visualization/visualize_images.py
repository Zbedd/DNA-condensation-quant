#!/usr/bin/env python3
"""
Image Visualization Script for DNA Condensation Analysis

This script provides configurable visualization of images at different processing stages:
- Raw (unprocessed) images
- Preprocessed images (background correction, intensity normalization, etc.)
- Final processed images (with per-nucleus normalization if applied)
- Segmentation overlays on any of the above

The script calls batch_processor.py with return_images=True to get all processing stages,
then displays a configurable grid of images based on user preferences.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import CheckButtons
import numpy as np
from skimage.segmentation import find_boundaries
from skimage.measure import find_contours
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dna_condensation.pipeline.batch_processor import main as run_batch_processor
from dna_condensation.pipeline.config import config as shared_config, Config
from dna_condensation.core.config_validator import ND2SelectionValidator


class ImageVisualizer:
    """Handles visualization of images at different processing stages with ND2 file selection support."""
    
    def __init__(self, config_file=None, selection_override=None):
        """
        Initialize with configuration and optional selection override.
        
        Args:
            config_file: Path to config file (optional)
            selection_override: Override for nd2_selection_settings (optional)
                              Useful for visualization-specific selection (e.g., limiting count for display)
        """
        # Prefer shared config to avoid multiple dev-mode prints; allow override file if provided
        if config_file:
            self.config = Config(config_file)
        else:
            self.config = shared_config

        # Get visualization settings from config
        viz_config = self.config.get("image_visualization", {})
        self.n_images = viz_config.get("n_images", 6)
        self.image_stages = viz_config.get("stages", ["raw", "preprocessed", "segmentation"])
        self.show_overlay = viz_config.get("show_segmentation_overlay", True)
        self.overlay_alpha = viz_config.get("overlay_alpha", 0.3)
        self.contour_color = viz_config.get("contour_color", "red")
        self.contour_width = viz_config.get("contour_width", 1.0)
        self.figure_size = viz_config.get("figure_size", [15, 10])
        self.colormap = viz_config.get("colormap", "gray")
        # Colormap used for labeled-object overlays (distinct colors per object)
        self.label_colormap = viz_config.get("label_colormap", "tab20")
        # If true and ND2 with transfection_channel_index set, show protein channel beneath overlays in segmentation panel
        self.show_protein_as_base = viz_config.get("show_protein_as_base", False)

        # Handle ND2 selection configuration
        self.selection_config = selection_override if selection_override is not None else self.config.get("nd2_selection_settings")
        self._validate_and_adjust_selection_config()

        # Get input source for timestamping
        self.input_source = self.config.get("input_source", "unknown")
        self.nd2_cfg = self.config.get("nd2_selection_settings", {}) or {}
        
        # Set up output directory
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_and_adjust_selection_config(self):
        """
        Validate ND2 selection config and adjust for visualization needs.
        
        For visualization, we want to ensure the count doesn't exceed what can be
        reasonably displayed, while preserving other selection criteria.
        """
        if self.selection_config is None:
            # Create a visualization-specific config to limit to displayable images
            self.selection_config = {
                'count': self.n_images,
                'seed': 42
            }
            print(f"No ND2 selection config found. Using visualization default: {self.n_images} images")
            return
        
        # Validate the selection config
        validator = ND2SelectionValidator()
        is_valid, errors = validator.validate_selection_config(self.selection_config)
        
        if not is_valid:
            print("❌ ND2 selection configuration errors:")
            for error in errors:
                print(f"  - {error}")
            print("Using default visualization settings instead.")
            self.selection_config = {
                'count': self.n_images,
                'seed': 42
            }
            return
        
        # Adjust count for visualization if needed
        config_count = self.selection_config.get('count')
        if config_count is None or config_count > self.n_images:
            # Create a copy of the config and limit count for visualization
            self.selection_config = dict(self.selection_config)
            self.selection_config['count'] = self.n_images
            print(f"Limiting display to {self.n_images} images for visualization")
        
        # Print selection summary
        print(validator.get_validation_summary(self.selection_config))
        
        # Set up output directory
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_timestamped_filename(self, base_name, extension="png"):
        """Generate a timestamped filename for saving plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.input_source}_{base_name}_{timestamp}.{extension}"
    
    def _get_save_path(self, save_path, default_name):
        """Get the save path, using timestamped default if none provided."""
        if save_path is None:
            filename = self._generate_timestamped_filename(default_name)
            return self.output_dir / filename
        return Path(save_path)
        
    def get_processed_images(self):
        """Get all processing stages from batch processor with selection config applied."""
        print("Running batch processor with ND2 selection settings...")
        
        # Temporarily override the config's ND2 selection settings for this visualization
        original_selection_config = self.config.get("nd2_selection_settings")
        
        try:
            # Apply our visualization-specific selection config
            self.config.set("nd2_selection_settings", self.selection_config)
            
            # Run batch processor with our modified config
            result = run_batch_processor(return_images=True, skip_validation=True)
            
            if result is None:
                raise RuntimeError("Failed to get images from batch processor")
                
            return result
        finally:
            # Restore original config
            if original_selection_config is not None:
                self.config.set("nd2_selection_settings", original_selection_config)
            else:
                # Remove the key if it wasn't there originally
                self.config.pop("nd2_selection_settings", None)
    
    def _normalize_image_for_display(self, image, channel_index=None):
        """Normalize image for consistent display."""
        if image is None:
            return None
            
        # Extract channel if needed
        if image.ndim == 3 and channel_index is not None:
            img = image[:, :, channel_index]
        else:
            img = image
            
        # Normalize to 0-1 range for display
        img = img.astype(np.float32)
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        
        return img
    
    def _add_segmentation_overlay(self, ax, mask, color="red", alpha=0.3, contour_width=1.0, show_overlay=True, show_contours=True):
        """Add segmentation overlay with support for color-coded labeled objects.

        If mask contains integer labels (>1), each object will be drawn with a distinct
        color using self.label_colormap. Otherwise, a single-color overlay is used.
        """
        if mask is None:
            return None, None
            
        overlay_artist = None
        contour_artist = None
        
        # Add filled overlay if requested
        if show_overlay:
            # Make overlays easier to see: half as transparent (double alpha, cap at 1.0)
            used_alpha = min(1.0, float(alpha) * 2.0)
            # Determine if mask is labeled (per-object IDs)
            try:
                mask_max = int(np.max(mask))
            except Exception:
                mask_max = 0

            is_integer_mask = np.issubdtype(mask.dtype, np.integer)
            is_labeled = is_integer_mask and mask_max > 1

            if is_labeled:
                # Create color lookup table for labels using the configured colormap
                cmap = plt.get_cmap(self.label_colormap)
                n_colors = getattr(cmap, 'N', 256)

                # Build LUT: index 0 is fully transparent; 1..max get distinct colors
                lut = np.zeros((mask_max + 1, 4), dtype=float)
                for lbl in range(1, mask_max + 1):
                    rgba = list(cmap((lbl % n_colors) / max(1, n_colors - 1)))
                    rgba[3] = used_alpha  # increased opacity
                    lut[lbl] = rgba

                colored_mask = lut[mask.astype(int)]  # (H, W, 4)
                overlay_artist = ax.imshow(colored_mask, alpha=1.0)
            else:
                # Binary mask → single-color overlay
                colored_mask = np.zeros((*mask.shape, 4), dtype=float)
                segmented = mask > 0
                if color == "red":
                    colored_mask[segmented] = [1, 0, 0, used_alpha]
                elif color == "blue":
                    colored_mask[segmented] = [0, 0, 1, used_alpha]
                elif color == "green":
                    colored_mask[segmented] = [0, 1, 0, used_alpha]
                elif color == "yellow":
                    colored_mask[segmented] = [1, 1, 0, used_alpha]
                else:
                    colored_mask[segmented] = [1, 0, 0, used_alpha]  # Default to red
                overlay_artist = ax.imshow(colored_mask, alpha=1.0)  # colored_mask already encodes alpha
        
        # Add contours if requested
        if show_contours:
            try:
                boundaries = find_boundaries(mask > 0, mode='outer')
                contour_artist = ax.contour(boundaries, levels=[0.5], colors=color, linewidths=contour_width)
            except Exception:
                # Fallback to find_contours if boundaries fail
                try:
                    contours = find_contours(mask.astype(float), 0.5)
                    for contour in contours:
                        line_artist = ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=contour_width)
                        if contour_artist is None:
                            contour_artist = line_artist
                except Exception:
                    pass  # Skip contours if both methods fail
        
        return overlay_artist, contour_artist
    
    def visualize_processing_stages(self, save_path=None, auto_save=True):
        """Create interactive visualization showing different processing stages."""
        # Get processed images
        result = self.get_processed_images()
        
        raw_images = result['raw_images']
        global_preprocessed = result['global_preprocessed']
        per_nucleus_preprocessed = result['per_nucleus_preprocessed']
        final_images = result['final_images']
        masks = result['masks']
        image_names = result['image_names']
        channel_index = result['channel_index']
        
        # Limit to requested number of images
        n_display = min(self.n_images, len(raw_images))
        
        # Determine number of columns based on stages requested
        n_stages = len(self.image_stages)
        n_cols = n_stages
        n_rows = n_display
        
        # Create figure with extra space for controls
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figure_size)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Stage mapping
        stage_data = {
            'raw': raw_images,
            'global_preprocessed': global_preprocessed,
            'per_nucleus_preprocessed': per_nucleus_preprocessed,
            'final': final_images,
            'preprocessed': global_preprocessed  # Alias for backward compatibility
        }
        
        stage_titles = {
            'raw': 'Raw',
            'global_preprocessed': 'Global Preprocessed',
            'per_nucleus_preprocessed': 'Per-Nucleus Normalized',
            'final': 'Final Processed',
            'preprocessed': 'Preprocessed',
            'segmentation': 'Analysis Images + Labels'
        }
        
        # Store overlay artists for toggling
        overlay_artists = []
        contour_artists = []
        # For interactive protein-base toggle on the segmentation (rightmost) panel
        seg_base_artists = [None] * n_display  # AxesImage per row
        seg_base_images = [None] * n_display   # dict per row: {'nuc': np.ndarray, 'prot': np.ndarray or None}
        
        # Plot images
        for row in range(n_display):
            row_overlays = []
            row_contours = []
            
            for col, stage in enumerate(self.image_stages):
                ax = axes[row, col]
                
                # Determine base image and overlay
                if stage == 'segmentation':
                    # Use final processed images as base (same images used for analysis)
                    base_images = final_images
                    enable_overlay_controls = True
                else:
                    base_images = stage_data.get(stage, raw_images)
                    enable_overlay_controls = False
                
                # Get and normalize image
                if base_images and row < len(base_images):
                    # Compute nuclear-base image (from analysis images)
                    nuc_img = self._normalize_image_for_display(base_images[row], channel_index)
                    # Compute optional protein-base image from RAW images (ND2 only)
                    prot_img = None
                    if (
                        stage == 'segmentation'
                        and str(self.input_source).lower() == 'nd2'
                        and self.nd2_cfg.get('transfection_channel_index') is not None
                        and raw_images and row < len(raw_images)
                        and raw_images[row] is not None
                        and raw_images[row].ndim == 3
                    ):
                        prot_idx = int(self.nd2_cfg.get('transfection_channel_index'))
                        if 0 <= prot_idx < raw_images[row].shape[2]:
                            prot_img = self._normalize_image_for_display(raw_images[row], prot_idx)

                    # Decide which base to show initially
                    img = prot_img if (stage == 'segmentation' and self.show_protein_as_base and prot_img is not None) else nuc_img
                    
                    if img is not None:
                        im_artist = ax.imshow(img, cmap=self.colormap)
                        # Save base artist and both options for segmentation panel only
                        if stage == 'segmentation':
                            seg_base_artists[row] = im_artist
                            seg_base_images[row] = {'nuc': nuc_img, 'prot': prot_img}
                        
                        # Add segmentation overlay if this is the segmentation stage
                        if enable_overlay_controls and masks and row < len(masks):
                            overlay_artist, contour_artist = self._add_segmentation_overlay(
                                ax, masks[row], 
                                color=self.contour_color,
                                alpha=self.overlay_alpha,
                                contour_width=self.contour_width,
                                show_overlay=self.show_overlay,
                                show_contours=True  # Start with contours enabled
                            )
                            row_overlays.append(overlay_artist)
                            row_contours.append(contour_artist)
                        else:
                            row_overlays.append(None)
                            row_contours.append(None)
                else:
                    ax.text(0.5, 0.5, 'No Image', ha='center', va='center', transform=ax.transAxes)
                    row_overlays.append(None)
                    row_contours.append(None)
                
                # Set title
                if row == 0:
                    title = stage_titles.get(stage, stage.title())
                    ax.set_title(title, fontsize=12, fontweight='bold')
                
                # Add image name on the left
                if col == 0:
                    img_name = image_names[row] if row < len(image_names) else f'Image {row+1}'
                    # Truncate long names
                    if len(img_name) > 25:
                        img_name = img_name[:22] + '...'
                    ax.set_ylabel(img_name, fontsize=10, rotation=0, ha='right', va='center')
                
                ax.set_xticks([])
                ax.set_yticks([])
            
            overlay_artists.append(row_overlays)
            contour_artists.append(row_contours)
        
        # Add interactive controls
        self._add_interactive_controls(fig, overlay_artists, contour_artists, seg_base_artists, seg_base_images)
        # Track protein-base state on the figure for toggling
        fig._protein_base_on = bool(self.show_protein_as_base)
        
        plt.tight_layout()
        
        # Adjust layout to make room for controls
        plt.subplots_adjust(left=0.15)
        
        # Save plot if enabled
        if auto_save:
            save_path = self._get_save_path(save_path, "processing_stages_comparison")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.show()
        return fig
    
    def _add_interactive_controls(self, fig, overlay_artists, contour_artists, seg_base_artists=None, seg_base_images=None):
        """Add interactive toggle buttons for overlays and contours."""
        # Create axes for the checkboxes
        checkbox_ax = plt.axes([0.01, 0.65, 0.14, 0.22])
        
        # Create checkboxes
        labels = ['Fill Overlay', 'Contours', 'Protein Base']
        visibility = [self.show_overlay, True, bool(self.show_protein_as_base)]  # Start with overlays as configured
        checkbox = CheckButtons(checkbox_ax, labels, visibility)
        
        # Customize checkbox appearance
        for lbl in checkbox.labels:
            lbl.set_fontsize(10)
        
        def toggle_overlay(label):
            """Toggle overlay visibility."""
            if label == 'Fill Overlay':
                # Toggle fill overlays
                for row_overlays in overlay_artists:
                    for overlay in row_overlays:
                        if overlay is not None:
                            overlay.set_visible(not overlay.get_visible())
            elif label == 'Contours':
                # Toggle contours
                for row_contours in contour_artists:
                    for contour in row_contours:
                        if contour is not None:
                            # Handle both ContourSet and Line2D objects
                            if hasattr(contour, 'collections'):
                                # ContourSet object
                                for collection in contour.collections:
                                    collection.set_visible(not collection.get_visible())
                            elif hasattr(contour, '__iter__'):
                                # List of Line2D objects
                                for line in contour:
                                    if hasattr(line, 'set_visible'):
                                        line.set_visible(not line.get_visible())
                            else:
                                # Single artist
                                contour.set_visible(not contour.get_visible())
            elif label == 'Protein Base':
                # Toggle base image between nuclear and protein in segmentation column
                fig._protein_base_on = not getattr(fig, '_protein_base_on', False)
                if seg_base_artists is not None and seg_base_images is not None:
                    for i, artist in enumerate(seg_base_artists):
                        if artist is None:
                            continue
                        img_pair = seg_base_images[i] if i < len(seg_base_images) else None
                        if not isinstance(img_pair, dict):
                            continue
                        target = 'prot' if fig._protein_base_on else 'nuc'
                        new_img = img_pair.get(target)
                        if new_img is not None:
                            artist.set_data(new_img)
            
            plt.draw()
        
        checkbox.on_clicked(toggle_overlay)
        
        # Store reference to prevent garbage collection
        fig._interactive_controls = checkbox

    def visualize_single_stage(self, stage='final', save_path=None, auto_save=True):
        """Visualize a single processing stage in a grid."""
        # Get processed images
        result = self.get_processed_images()
        
        # Get stage data
        stage_data = {
            'raw': result['raw_images'],
            'global_preprocessed': result['global_preprocessed'],
            'per_nucleus_preprocessed': result['per_nucleus_preprocessed'],
            'final': result['final_images'],
            'preprocessed': result['global_preprocessed']
        }
        
        images = stage_data.get(stage, result['raw_images'])
        masks = result['masks']
        image_names = result['image_names']
        channel_index = result['channel_index']
        
        # Limit to requested number of images
        n_display = min(self.n_images, len(images))
        
        # Calculate grid dimensions
        n_cols = min(3, n_display)
        n_rows = int(np.ceil(n_display / n_cols))
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figure_size)
        if n_display == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Plot images
        for i in range(n_display):
            ax = axes[i]
            
            # Get and normalize image
            img = self._normalize_image_for_display(images[i], channel_index)
            
            if img is not None:
                ax.imshow(img, cmap=self.colormap)
                
                # Add segmentation overlay if requested
                if self.show_overlay and masks and i < len(masks):
                    self._add_segmentation_overlay(
                        ax, masks[i],
                        color=self.contour_color,
                        alpha=self.overlay_alpha,
                        contour_width=self.contour_width
                    )
            else:
                ax.text(0.5, 0.5, 'No Image', ha='center', va='center', transform=ax.transAxes)
            
            # Set title
            img_name = image_names[i] if i < len(image_names) else f'Image {i+1}'
            if len(img_name) > 30:
                img_name = img_name[:27] + '...'
            ax.set_title(img_name, fontsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused axes
        for i in range(n_display, len(axes)):
            axes[i].axis('off')
        
        # Add main title
        stage_title = {
            'raw': 'Raw Images',
            'global_preprocessed': 'Global Preprocessed Images',
            'per_nucleus_preprocessed': 'Per-Nucleus Normalized Images',
            'final': 'Final Processed Images',
            'preprocessed': 'Preprocessed Images'
        }.get(stage, f'{stage.title()} Images')
        
        overlay_text = ' with Segmentation Overlay' if self.show_overlay else ''
        fig.suptitle(f'{stage_title}{overlay_text}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot if enabled
        if auto_save:
            save_path = self._get_save_path(save_path, f"single_stage_{stage}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.show()
        return fig


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize DNA condensation analysis images with interactive controls')
    parser.add_argument('--mode', choices=['stages', 'single'], default='stages',
                       help='Visualization mode: stages (compare stages with interactive controls) or single (one stage grid)')
    parser.add_argument('--stage', choices=['raw', 'preprocessed', 'final'], default='final',
                       help='Stage to visualize in single mode')
    parser.add_argument('--n-images', type=int, default=6,
                       help='Number of images to display')
    parser.add_argument('--save', type=str, 
                       help='Custom path to save the visualization (default: auto-timestamped in output/)')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable automatic saving of plots')
    parser.add_argument('--no-overlay', action='store_true',
                       help='Disable faint segmentation overlay on segmentation stage')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ImageVisualizer()
    
    # Override config with command line args
    visualizer.n_images = args.n_images
    if args.no_overlay:
        visualizer.show_overlay = False
    
    # Determine save settings
    auto_save = not args.no_save
    save_path = args.save if args.save else None
    
    # Run visualization
    if args.mode == 'stages':
        visualizer.visualize_processing_stages(save_path=save_path, auto_save=auto_save)
    else:
        visualizer.visualize_single_stage(stage=args.stage, save_path=save_path, auto_save=auto_save)


if __name__ == "__main__":
    main()
