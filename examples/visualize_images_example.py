#!/usr/bin/env python3
"""
Example usage of the image visualization script.

This demonstrates how to use visualize_images.py to display different
processing stages of DNA condensation analysis.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dna_condensation.visualization.visualize_images import ImageVisualizer


def example_basic_usage():
    """Basic usage example showing processing stages with auto-save."""
    print("=== Basic Visualization Example ===")
    
    # Create visualizer with default config
    visualizer = ImageVisualizer()
    
    # Show comparison of processing stages (auto-saved with timestamp)
    print("Displaying processing stages comparison...")
    print(f"Will auto-save to: {visualizer.output_dir}")
    visualizer.visualize_processing_stages()


def example_custom_config():
    """Example with custom configuration and auto-save."""
    print("\n=== Custom Configuration Example ===")
    
    # Create visualizer
    visualizer = ImageVisualizer()
    
    # Customize settings
    visualizer.n_images = 4
    visualizer.image_stages = ["raw", "final"]
    visualizer.show_overlay = True
    visualizer.contour_color = "blue"
    visualizer.overlay_alpha = 0.5
    
    print("Displaying custom configuration...")
    print(f"Will auto-save to: {visualizer.output_dir}")
    visualizer.visualize_processing_stages()


def example_no_save():
    """Example showing visualization without saving."""
    print("\n=== No Save Example ===")
    
    visualizer = ImageVisualizer()
    
    print("Displaying without saving...")
    visualizer.visualize_processing_stages(auto_save=False)


def example_custom_save_path():
    """Example with custom save path."""
    print("\n=== Custom Save Path Example ===")
    
    visualizer = ImageVisualizer()
    
    custom_path = "my_custom_visualization.png"
    print(f"Saving to custom path: {custom_path}")
    visualizer.visualize_processing_stages(save_path=custom_path)


def example_single_stage():
    """Example showing single processing stage in grid."""
    print("\n=== Single Stage Grid Example ===")
    
    # Create visualizer
    visualizer = ImageVisualizer()
    visualizer.n_images = 9  # 3x3 grid
    
    # Show final processed images with segmentation overlay (auto-saved)
    print("Displaying final processed images...")
    print(f"Will auto-save to: {visualizer.output_dir}")
    visualizer.visualize_single_stage(stage='final')


if __name__ == "__main__":
    print("DNA Condensation Image Visualization Examples")
    print("=" * 50)
    
    # Run examples
    try:
        example_basic_usage()
        example_custom_config()
        example_no_save()
        example_custom_save_path()
        example_single_stage()
        
        print("\n✓ All examples completed successfully!")
        print(f"Check the output directory for saved visualizations!")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("Make sure you have:")
        print("1. Proper ND2 files configured in config.yaml")
        print("2. All dependencies installed")
        print("3. Pipeline can run successfully")
