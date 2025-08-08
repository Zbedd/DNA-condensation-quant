from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Imports
import os
import shutil
from datetime import datetime
from dna_condensation.pipeline.config import Config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.z_stack_handling import batch_collapse_z_axis
from dna_condensation.core.preprocessor import bulk_preprocess_images, per_nucleus_intensity_normalization
from dna_condensation.core.segmentation import bulk_segment_images
from dna_condensation.visualization.plotting import plot_image, plot_multiple, plot_image_mask, plot_preprocessing_comparison
from dna_condensation.analysis.analysis_pipeline import run_analysis_from_batch_processor

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
  
  # Apply global preprocessing if enabled in config
  global_preprocessed = None
  if any([preprocessing_config.get("background_correction"), 
          preprocessing_config.get("deconvolution"), 
          preprocessing_config.get("intensity_normalization")]):
    
    # Build sequential preprocessing pipeline based on enabled methods
    # Order matters: deconvolution → background correction → intensity normalization
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
    # Skip preprocessing - use raw collapsed images
    global_preprocessed = collapsed_images
    print("No global preprocessing applied")
  
  # Configure size-based filtering to remove segmentation artifacts
  size_filter_config = config.get("size_filtering")
  
  # Set up visualization options
  plot_config = config.get("plot")
  plot_segmentation = plot_config.get("plot_segmentation") 
  
  filter_status = f"with {size_filter_config.get('min_size_percentage')}% size filter" if size_filter_config.get('enabled') else "no size filtering"
  print(f'Segmenting processed images with {segmentation_method.upper()} model (using channel {channel_index}, {filter_status})')
  
  # Perform segmentation using globally preprocessed images
  # CRITICAL: return_labels=True ensures each nucleus gets unique ID for per-nucleus feature extraction
  masks = bulk_segment_images(global_preprocessed, channel_index=channel_index, method=segmentation_method, 
                             size_filter_config=size_filter_config, return_labels=True)

  # Apply per-nucleus intensity normalization if enabled (alternative to global normalization)
  # This normalizes each nucleus individually to target_mean, useful for homogeneity analysis
  per_nucleus_preprocessed = None
  per_nucleus_stats = None
  if preprocessing_config.get("per_nucleus_normalization") and masks[0] is not None:
    print("Applying per-nucleus intensity normalization...")
    per_nucleus_preprocessed = []
    per_nucleus_stats = []
    
    for i, (image, labels) in enumerate(zip(global_preprocessed, masks)):
      if image is not None and labels is not None:
        # Extract the same channel used for segmentation
        if image.ndim == 3:
          channel_image = image[:, :, channel_index]
        else:
          channel_image = image
          
        # Normalize each nucleus to have mean intensity = 1.0
        # This makes CV calculations more intuitive (CV = std when mean = 1)
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

  # === DNA CONDENSATION ANALYSIS ===
  print("\n" + "="*60)
  print("RUNNING DNA CONDENSATION ANALYSIS")
  print("="*60)
  
  # Determine which images to use for analysis (final preprocessed images)
  final_images = per_nucleus_preprocessed if per_nucleus_preprocessed else global_preprocessed
  
  # Extract image names for analysis
  image_names = [str(Path(nd2_obj.filename).name) for nd2_obj in nd2_objects]

  # Prepare timestamped output directory structure
  root_output = Path("dna_condensation_analysis_results")
  root_output.mkdir(parents=True, exist_ok=True)
  
  # Create unique timestamped subfolder for this analysis run
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  run_output = root_output / ts
  run_output.mkdir(parents=True, exist_ok=True)
  
  # Archive current configuration for reproducibility
  try:
    cfg_src = Path(__file__).parent / 'config.yaml'
    if cfg_src.exists():
      cfg_dest = run_output / f"config_{ts}.txt"
      shutil.copy2(cfg_src, cfg_dest)
      print(f"Configuration archived: {cfg_dest.name}")
  except Exception as _:
    pass
  
  # Execute comprehensive feature extraction and statistical analysis
  try:
    analysis_results = run_analysis_from_batch_processor(
      final_images, masks, image_names, 
      output_dir=str(run_output)
    )
    print("\n✓ DNA condensation analysis completed successfully!")
    print(f"Results saved to: {run_output}/")
    
    # Generate single-metric significance plots if configured
    plot_config = config.get("plot", {})
    single_metric_config = plot_config.get("single_metric_plot", {})

    if single_metric_config.get("enabled", False):
      try:
        import pandas as pd
        from dna_condensation.visualization.visualize_statistics import StatisticalVisualizer

        # Create dedicated subfolder for single metric plots
        single_plots_dir = run_output / "single_metric_plots"
        single_plots_dir.mkdir(parents=True, exist_ok=True)

        # Load extracted features for plotting
        features_df = pd.read_csv(run_output / "all_features.csv")

        # Parse metric configuration (handles string, list, or null values)
        raw_metric_cfg = single_metric_config.get("metric")
        
        # Convert config to list of valid metric names
        if isinstance(raw_metric_cfg, list):
          cfg_metrics = [m for m in raw_metric_cfg if isinstance(m, str) and m.strip() and m.lower() not in ["none", "null"]]
        elif isinstance(raw_metric_cfg, str) and raw_metric_cfg.strip() and raw_metric_cfg.lower() not in ["none", "null"]:
          cfg_metrics = [raw_metric_cfg.strip()]
        else:
          cfg_metrics = []

        metrics_to_plot = []
        if cfg_metrics:
          metrics_to_plot = cfg_metrics
        else:
          # Auto-discover all analyzable metrics
          # Prefer features tested in group_comparisons.csv
          comp_path = run_output / "group_comparisons.csv"
          if comp_path.exists():
            try:
              comp_df = pd.read_csv(comp_path)
              if "feature" in comp_df.columns:
                metrics_to_plot = sorted(comp_df["feature"].dropna().unique().tolist())
            except Exception:
              metrics_to_plot = []
          # Fallback to numeric columns in all_features.csv minus non-feature columns
          if not metrics_to_plot:
            exclude_cols = {
              'image_name', 'nucleus_id', 'centroid_x', 'centroid_y',
              'dk_group', 'dk_number', 'condition', 'well', 'timepoint'
            }
            numeric_cols = features_df.select_dtypes(include=['number']).columns.tolist()
            metrics_to_plot = [c for c in numeric_cols if c not in exclude_cols]

        # Filter out any metrics not present in features_df
        metrics_to_plot = [m for m in metrics_to_plot if m in features_df.columns]
        metrics_to_plot = sorted(dict.fromkeys(metrics_to_plot))  # de-duplicate & sort

        if not metrics_to_plot:
          print("📊 No valid metrics found to plot.")
        else:
          visualizer = StatisticalVisualizer()
          print(f"\n📊 Creating standalone plot(s) for {len(metrics_to_plot)} metric(s)...")
          for metric_to_plot in metrics_to_plot:
            try:
              save_path = single_plots_dir / f"single_metric_{metric_to_plot}.png"
              visualizer.plot_single_metric_with_significance(
                df=features_df,
                metric=metric_to_plot,
                group_column='condition',
                save_path=save_path
              )
              print(f"✓ Saved: {save_path}")
            except Exception as e:
              print(f"✗ Failed plotting {metric_to_plot}: {e}")
      except Exception as e:
        print(f"✗ Failed to create single metric plot(s): {e}")
        print("  Check that metric names are valid or leave as null to auto-discover.")
    else:
      print("📊 Single metric plot disabled in config")

    # Create nuclei panel if configured
    nuclei_cfg = plot_config.get("nuclei_panel", {})
    if nuclei_cfg.get("enabled", False):
      try:
        print("\n🧬 Creating nuclei panel...")
        import pandas as pd
        from dna_condensation.visualization.visualize_nuclei import create_nuclei_panel
        
        features_df = pd.read_csv(run_output / "all_features.csv")
        out_dir = run_output
        out_path = out_dir / nuclei_cfg.get("save_name", "nuclei_panel.png")
        
        create_nuclei_panel(
          images=final_images,
          masks=masks,
          image_names=image_names,
          features_df=features_df,
          n_nuclei=int(nuclei_cfg.get("n_nuclei", 12)),
          metrics=nuclei_cfg.get("metrics", ["glcm_homogeneity_mean", "intensity_entropy"]),
          group=nuclei_cfg.get("group"),
          group_column=nuclei_cfg.get("group_column", "condition"),
          colormap=nuclei_cfg.get("colormap", "magma"),
          channel_index=nuclei_cfg.get("channel_index"),
          save_path=out_path,
          random_state=42,
        )
        print(f"✓ Nuclei panel saved to: {out_path}")
      except Exception as e:
        print(f"✗ Failed to create nuclei panel: {e}")
    else:
      print("🧬 Nuclei panel disabled in config")
    
  except Exception as e:
    print(f"\n✗ Analysis failed: {e}")
    print("Continuing with basic processing...")

  '''CHECK THE FORMAT OF MASKS'''

if __name__ == "__main__":
  main()