from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Imports
import os
from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.preprocessor import batch_collapse_z_axis
from dna_condensation.core.segmentation import bulk_segment_images
from dna_condensation.visualization.plotting import plot_image, plot_multiple, plot_image_mask

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
  
  # Get size filtering configuration
  size_filter_config = config.get("size_filtering")
  
  filter_status = f"with {size_filter_config.get('min_size_percentage')}% size filter" if size_filter_config.get('enabled') else "no size filtering"
  print(f'Segmenting collapsed images with {segmentation_method.upper()} model (using channel {channel_index}, {filter_status})')
  
  masks = bulk_segment_images(collapsed_images, channel_index=channel_index, method=segmentation_method, 
                             size_filter_config=size_filter_config)

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
  
  plot_image_mask(display_image, masks[0])

  '''CHECK THE FORMAT OF MASKS'''

if __name__ == "__main__":
  main()