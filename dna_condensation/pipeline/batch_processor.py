from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Imports
import os
from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.z_stack_handling import batch_collapse_z_axis
from dna_condensation.core.preprocessor import bulk_preprocess_images, per_nucleus_intensity_normalization
from dna_condensation.core.segmentation import bulk_segment_images
from dna_condensation.visualization.plotting import plot_image, plot_multiple, plot_image_mask, plot_preprocessing_comparison

# Initialize config
config = Config()

'''
Sample ND2 File Analysis: 48hr_dk16_wtLSD1_well1_20x001.nd2
==================================================
File size: 40.4 MB
Total frames: 10

Dimension structure: x * y * c * t * z
Shape: (1024, 1024, 2, 1, 10)

Dimension details:
  X: 1024 (width)
  Y: 1024 (height)
  C: 2 (channels)
  T: 1 (time/frames)
  Z: 10 (z-stack/depth)

Channel information:
  Channel 0: Channel_0
  Channel 1: Channel_1
'''

def main():
  nd2_folder_path = config.get("raw_nd2_path")

  # # Iterate over every ND2 file and check channel count consistency
  # nd2_files = [f for f in os.listdir(nd2_folder_path) if f.lower().endswith('.nd2')]
  # channel_counts = set()
  # for nd2_file in nd2_files:
  #   nd2_path = os.path.join(nd2_folder_path, nd2_file)
  #   info = characterize_nd2(nd2_path, verbose=False)
  #   channel_counts.add(info.get('axes'))
  # print(channel_counts)

  # if len(channel_counts) == 1:
  #   print(f"All ND2 files have the same channel count: {channel_counts.pop()}")
  # else:
  #   print(f"Inconsistent channel counts found: {channel_counts}")
  
  # Example main function for batch processing
  if nd2_folder_path is None:
    print("Raw ND2 folder path not specified in config.")
    return

  if not os.path.exists(nd2_folder_path):
    print(f"ND2 folder not found: {nd2_folder_path}")
    return

  print(f"Processing ND2 folder: {nd2_folder_path}")
  nd2_objects = get_nd2_objects(nd2_folder_path)
  
  print(f"Collapsing z-axis for {len(nd2_objects)} ND2 files")
  collapsed_images = batch_collapse_z_axis(nd2_objects, method='mean')
  
  # Get segmentation parameters from config
  channel_index = config.get("segmentation_channel_index")
  segmentation_method = config.get("segmentation_method")
  
  # Get preprocessing configuration
  preprocessing_config = config.get("preprocessing")
  
  # Apply global preprocessing if enabled
  global_preprocessed = None
  if any([preprocessing_config.get("background_correction"), 
          preprocessing_config.get("deconvolution"), 
          preprocessing_config.get("intensity_normalization")]):
    
    # Build preprocessing methods list based on config
    methods = []
    if preprocessing_config.get("deconvolution"):
      methods.append("deconvolution")
    if preprocessing_config.get("background_correction"):
      methods.append("background_correction")
    if preprocessing_config.get("intensity_normalization"):
      methods.append("intensity_normalization")
    
    print(f"Applying global preprocessing: {' → '.join(methods)}")
    global_preprocessed = bulk_preprocess_images(
      collapsed_images,
      channel_index=channel_index,
      methods=methods,
      bg_ball_radius=preprocessing_config.get("bg_ball_radius", 50),
      deconv_iterations=preprocessing_config.get("deconv_iterations", 10),
      norm_method=preprocessing_config.get("norm_method", "percentile")
    )
  else:
    global_preprocessed = collapsed_images
    print("No global preprocessing applied")
  
  # Get size filtering configuration
  size_filter_config = config.get("size_filtering")
  
  # Get plotting configuration
  plot_config = config.get("plot")
  plot_segmentation = plot_config.get("plot_segmentation") 
  
  filter_status = f"with {size_filter_config.get('min_size_percentage')}% size filter" if size_filter_config.get('enabled') else "no size filtering"
  print(f'Segmenting processed images with {segmentation_method.upper()} model (using channel {channel_index}, {filter_status})')
  
  # Use globally preprocessed images for segmentation
  masks = bulk_segment_images(global_preprocessed, channel_index=channel_index, method=segmentation_method, 
                             size_filter_config=size_filter_config)

  # Apply per-nucleus normalization if enabled
  per_nucleus_preprocessed = None
  per_nucleus_stats = None
  if preprocessing_config.get("per_nucleus_normalization") and masks[0] is not None:
    print("Applying per-nucleus intensity normalization...")
    per_nucleus_preprocessed = []
    per_nucleus_stats = []
    
    for i, (image, labels) in enumerate(zip(global_preprocessed, masks)):
      if image is not None and labels is not None:
        # Extract channel for per-nucleus processing
        if image.ndim == 3:
          channel_image = image[:, :, channel_index]
        else:
          channel_image = image
          
        norm_image, stats = per_nucleus_intensity_normalization(
          channel_image, labels, target_mean=1.0, verbose=(i == 0)
        )
        per_nucleus_preprocessed.append(norm_image)
        per_nucleus_stats.append(stats)
      else:
        per_nucleus_preprocessed.append(None)
        per_nucleus_stats.append({})
  
  # Visualization if enabled
  if plot_segmentation and masks[0] is not None:
    # Prepare image for visualization - extract the same channel used for segmentation
    first_image = collapsed_images[0]
    if first_image.ndim == 3:
      # Multi-channel image - extract the channel used for segmentation
      display_image = first_image[:, :, channel_index]
      print(f"Extracted channel {channel_index} for visualization: {display_image.shape}")
    else:
      # Single-channel image
      display_image = first_image
      print(f"Using single-channel image for visualization: {display_image.shape}")
    
    # Show comprehensive preprocessing comparison if per-nucleus normalization was applied
    if per_nucleus_preprocessed and per_nucleus_preprocessed[0] is not None:
      global_display = global_preprocessed[0]
      if global_display.ndim == 3:
        global_display = global_display[:, :, channel_index]
      
      plot_preprocessing_comparison(
        original_image=display_image,
        global_preprocessed=global_display,
        per_nucleus_preprocessed=per_nucleus_preprocessed[0],
        labels=masks[0],
        title="DNA Condensation Preprocessing Pipeline"
      )
    else:
      # Fallback to standard visualization
      plot_image_mask(display_image, masks[0])

  '''CHECK THE FORMAT OF MASKS'''

if __name__ == "__main__":
  main()