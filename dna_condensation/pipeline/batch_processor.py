from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Imports
import os
import shutil
import numpy as np
import warnings
from datetime import datetime
from dna_condensation.pipeline.config import config
from dna_condensation.core.image_loader import get_nd2_objects
from dna_condensation.core.config_validator import ND2SelectionValidator
from dna_condensation.core.z_stack_handling import batch_collapse_z_axis
from dna_condensation.core.preprocessor import bulk_preprocess_images, per_nucleus_intensity_normalization
from dna_condensation.core.segmentation import bulk_segment_images
from dna_condensation.core.plotting import plot_image, plot_multiple, plot_image_mask, plot_preprocessing_comparison
from dna_condensation.analysis.analysis_pipeline import run_analysis_from_batch_processor

# Use shared config from pipeline.config (single source of truth)

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

def _check_direct_execution_with_count_limit(input_source):
  """
  Check if batch_processor is being run directly with count limitations for ND2 files.
  Warns user that not all ND2 images will be processed and requires confirmation.
  """
  # Only apply to ND2 input source
  if input_source != "nd2":
    return
  
  # Check if ND2 selection settings specify a count limit
  selection_config = config.get("nd2_selection_settings")
  if selection_config is None:
    return
  
  count_setting = selection_config.get("count")
  if count_setting is None:
    return
  
  # Display warning message
  print("\n" + "=" * 60)
  print("⚠️  WARNING: LIMITED ND2 FILE PROCESSING")
  print("=" * 60)
  print(f"\nYou are running batch_processor.py with a count limit of {count_setting} ND2 files.")
  print("This means NOT ALL ND2 images in your folder will be processed for analysis.")
  print("\nCurrent ND2 selection settings:")
  
  # Show current selection configuration
  validator = ND2SelectionValidator()
  print(validator.get_validation_summary(selection_config))
  
  print(f"\nTo process ALL ND2 files, either:")
  print(f"  1. Set 'count: null' in your config.yaml")
  print(f"  2. Remove the 'count' setting entirely")
  print(f"\nTo continue with the current limit of {count_setting} files, type 'y' and press Enter.")
  print("To cancel and modify your settings, press any other key and Enter.")
  
  # Get user confirmation
  try:
    user_input = input("\nContinue with limited processing? (y/N): ").strip().lower()
    if user_input != 'y':
      print("\n❌ Processing cancelled. Please modify your nd2_selection_settings in config.yaml")
      print("   Set 'count: null' to process all ND2 files.")
      exit(1)
    else:
      print(f"\n✅ Continuing with analysis of {count_setting} ND2 files...\n")
  except KeyboardInterrupt:
    print("\n\n❌ Processing cancelled by user.")
    exit(1)


def main(return_images=False, skip_validation=False):
  # Get input source configuration
  input_source = config.get("input_source")
  
  # Check if running directly with count limitation for ND2 files
  # Skip validation if explicitly requested (for programmatic calls) or when returning images
  if not skip_validation and not return_images:
    _check_direct_execution_with_count_limit(input_source)
  
  # Input-specific preparation - only loads/prepares raw images and metadata
  if input_source == "nd2":
    raw_images, image_names, metadata = prepare_nd2_inputs()
    output_dir = Path(config.get("nd2_output_path", "./output"))
  elif input_source == "bbbc022":
    raw_images, image_names, metadata = prepare_bbbc022_inputs()
    output_dir = Path(config.get("validation_output_path", "dna_condensation/validation/output"))
  else:
    raise ValueError(f"Invalid input_source '{input_source}'. Must be 'nd2' or 'bbbc022'.")
  
  # Single common pipeline - same preprocessing, segmentation, and analysis for both sources
  result = run_unified_pipeline(raw_images, image_names, metadata, output_dir, input_source, return_images=return_images)
  
  if return_images:
    return result
  return None


def prepare_nd2_inputs():
  """Load and prepare ND2 files with optional selection filtering - returns raw collapsed images only"""
  nd2_folder_path = config.get("raw_nd2_path")

  if nd2_folder_path is None:
    raise ValueError("Raw ND2 folder path not specified in config.")

  if not os.path.exists(nd2_folder_path):
    raise ValueError(f"ND2 folder not found: {nd2_folder_path}")

  # Get and validate ND2 selection settings (avoid duplicate printing; loader will print summary)
  selection_config = config.get("nd2_selection_settings")
  validator = ND2SelectionValidator()
  is_valid, errors = validator.validate_selection_config(selection_config)
  
  if not is_valid:
    print("❌ ND2 selection configuration errors:")
    for error in errors:
      print(f"  - {error}")
    raise ValueError("Invalid ND2 selection configuration. Please check your config.yaml file.")
  print(f"Processing ND2 folder: {nd2_folder_path}")
  
  # Load ND2 objects with selection filtering (also returns the selector used)
  nd2_objects = get_nd2_objects(nd2_folder_path, selection_config)
  
  print(f"Collapsing z-axis for {len(nd2_objects)} selected ND2 files")
  collapsed_images = batch_collapse_z_axis(nd2_objects, method=config.get("z_collapse_method", "mean"))
  
  # Extract image names for ND2 files - metadata will be extracted from filenames
  image_names = [str(Path(nd2_obj.filename).name) for nd2_obj in nd2_objects]
  
  # For ND2, we pass None as metadata to signal that analysis should extract from filenames
  metadata = None

  return collapsed_images, image_names, metadata


def prepare_bbbc022_inputs():
  """Load and prepare BBBC022 images - returns raw images only"""
  print("\n" + "="*60)
  print("LOADING BBBC022 VALIDATION DATASET")
  print("="*60)
  
  # Get BBBC022 configuration
  bbbc022_config = config.get("bbbc022_settings", {})
  validation_output_path = Path(config.get("validation_output_path", "dna_condensation/validation/output"))
  validation_output_path.mkdir(parents=True, exist_ok=True)
  
  # Load raw BBBC022 images using existing loader
  from dna_condensation.core.image_loader import load_bbbc022_images
  
  raw_images, metadata = load_bbbc022_images(
    count=bbbc022_config.get('count', 20),
    seed=bbbc022_config.get('seed', 42),
    output_dir=str(validation_output_path / "bbbc022_data"),
    use_cache=True,
    group_mapping=bbbc022_config.get('group_mapping')
  )
  
  # Ensure dtype is uint8 for downstream preprocessing/segmentation
  raw_converted = []
  for i, img in enumerate(raw_images):
    if img is None:
      raw_converted.append(None)
      continue
    if img.dtype == np.uint8:
      raw_converted.append(img)
    else:
      # Robust float handling assuming 0-255 range from loader
      if np.issubdtype(img.dtype, np.floating):
        raw_converted.append(np.clip(img, 0, 255).astype(np.uint8))
      elif img.dtype == np.uint16:
        # Scale down to 8-bit preserving dynamic range
        # Use simple right shift if full 16-bit, else min-max normalize
        maxv = img.max()
        minv = img.min()
        if maxv > 255 and (maxv - minv) > 0:
          scaled = (img.astype(np.float32) - minv) * (255.0 / (maxv - minv))
          raw_converted.append(np.clip(scaled, 0, 255).astype(np.uint8))
        else:
          raw_converted.append(np.clip(img, 0, 255).astype(np.uint8))
      else:
        raw_converted.append(img.astype(np.uint8))
  
  raw_images = raw_converted
  print(f"BBBC022 images converted to dtype uint8 for segmentation")
  
  print(f"✓ Loaded {len(raw_images)} BBBC022 images")
  
  # Use image names provided by loader metadata
  image_names = [m.get('image_name') for m in metadata]
  
  # Return images with names and unmodified metadata (loader is the single source of truth)
  return raw_images, image_names, metadata


def run_unified_pipeline(raw_images, image_names, metadata, output_dir, input_source, return_images=False):
  """Single unified pipeline for preprocessing, segmentation, and analysis"""
  print("\n" + "="*60)
  print("RUNNING UNIFIED PROCESSING PIPELINE")
  print("="*60)
  
  # Get configuration parameters
  channel_index = config.get_nuclear_channel_index()
  segmentation_method = config.get("segmentation_method")
  preprocessing_config = config.get("preprocessing")
  
  # === GLOBAL PREPROCESSING ===
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
      raw_images,
      channel_index=channel_index,
      methods=methods,
      bg_ball_radius=preprocessing_config.get("bg_ball_radius", 50),
      deconv_iterations=preprocessing_config.get("deconv_iterations", 10),
      norm_method=preprocessing_config.get("norm_method", "percentile")
    )
  else:
    # Skip preprocessing - use raw images
    global_preprocessed = raw_images
    print("No global preprocessing applied")
  
  # === SEGMENTATION ===
  # Configure size-based filtering to remove segmentation artifacts
  size_filter_config = config.get("size_filtering")
  
  filter_status = f"with {size_filter_config.get('min_size_percentage')}% size filter" if size_filter_config.get('enabled') else "no size filtering"
  print(f'Segmenting processed images with {segmentation_method.upper()} model (using channel {channel_index}, {filter_status})')
  
  # Perform segmentation using globally preprocessed images
  # CRITICAL: return_labels=True ensures each nucleus gets unique ID for per-nucleus feature extraction
  masks = bulk_segment_images(global_preprocessed, channel_index=channel_index, method=segmentation_method, 
                             size_filter_config=size_filter_config, return_labels=True)

  # === PER-NUCLEUS PREPROCESSING ===
  # Apply per-nucleus intensity normalization if enabled (alternative to global normalization)
  # This normalizes each nucleus individually to target_mean, useful for homogeneity analysis
  per_nucleus_preprocessed = None
  if preprocessing_config.get("per_nucleus_normalization") and masks[0] is not None:
    print("Applying per-nucleus intensity normalization...")
    per_nucleus_preprocessed = []
    
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
      else:
        per_nucleus_preprocessed.append(None)
  
  # === VISUALIZATION ===
  # Set up visualization options
  plot_config = config.get("plot")
  plot_segmentation = plot_config.get("plot_segmentation") 
  
  if plot_segmentation and masks[0] is not None:
    # Prepare image for visualization - extract the same channel used for segmentation
    first_image = raw_images[0]
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
  
  # === DETERMINE FINAL IMAGES ===
  # Determine which images to use for analysis (final preprocessed images)
  # This logic is now handled inside the analysis pipeline
  
  # === RETURN IMAGES IF REQUESTED ===
  if return_images:
    # Determine final images for analysis and visualization
    if preprocessing_config.get("per_nucleus_normalization"):
      final_images_for_analysis = per_nucleus_preprocessed
    else:
      final_images_for_analysis = global_preprocessed

    return {
      'raw_images': raw_images,
      'global_preprocessed': global_preprocessed,
      'per_nucleus_preprocessed': per_nucleus_preprocessed,
      'final_images': final_images_for_analysis,
      'masks': masks,
      'image_names': image_names,
      'metadata': metadata,
      'channel_index': channel_index
    }
  
  # === ANALYSIS PIPELINE ===
  run_common_analysis_pipeline(
      global_preprocessed_images=global_preprocessed,
      per_nucleus_preprocessed_images=per_nucleus_preprocessed,
      masks=masks,
      image_names=image_names,
      metadata=metadata,
      output_dir=output_dir,
      input_source=input_source
  )


def run_common_analysis_pipeline(global_preprocessed_images, per_nucleus_preprocessed_images, masks, image_names, metadata, output_dir, input_source):
  """Run the analysis pipeline - same for both ND2 and BBBC022"""
  print("\n" + "="*60)
  print("RUNNING DNA CONDENSATION ANALYSIS")
  print("="*60)
  
  # Prepare timestamped output directory structure
  output_dir = Path(output_dir)
  root_output = output_dir / "dna_condensation_analysis_results"
  root_output.mkdir(parents=True, exist_ok=True)
  
  # Create unique timestamped subfolder for this analysis run
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  run_output = root_output / f"{input_source}_{ts}"
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
    # Get image aggregation setting from config
    statistical_config = config.get("statistical_analysis", {})
    use_image_aggregation = statistical_config.get("use_image_aggregation", True)
  # Handle metadata sources differently:
  # - If metadata is None: use filename parsing via original extract_experimental_metadata
  # - If metadata is provided: use it directly by patching the extractor during this run
    from dna_condensation.analysis import feature_extractor as _fe
    from dna_condensation.analysis import analysis_pipeline as _ap
    _orig_extract = getattr(_fe, 'extract_experimental_metadata', None)
    _orig_ap_extract = getattr(_ap, 'extract_experimental_metadata', None)

    if metadata is None:
      # Use original metadata extraction from filenames
      print("Using filename-based metadata extraction from image names")
      analysis_results = run_analysis_from_batch_processor(
        global_preprocessed_images=global_preprocessed_images,
        per_nucleus_preprocessed_images=per_nucleus_preprocessed_images,
        masks=masks,
        image_names=image_names,
        output_dir=str(run_output),
        use_image_aggregation=use_image_aggregation,
        config=config._config
      )
    else:
      # Patch metadata extraction to use provided metadata
      print("Using provided metadata from loader")
      
      def _patched_extract(image_names_list):
        import pandas as pd

        # Create fast lookup by image_name
        meta_by_name = {m.get('image_name', m.get('filename', '')): m for m in metadata}

        records = []
        for name in image_names_list:
          m = meta_by_name.get(name, {})
          if not m:
            raise ValueError(f"No metadata found for image '{name}'. Available images: {list(meta_by_name.keys())}")
          # Keep metadata minimal and generic for downstream pipeline
          records.append({
            'image_name': name,
            'condition': m.get('condition', 'unknown'),
          })
        return pd.DataFrame(records)

      # Apply patch
      if callable(_orig_extract):
        _fe.extract_experimental_metadata = _patched_extract
      if callable(_orig_ap_extract):
        _ap.extract_experimental_metadata = _patched_extract

      analysis_results = run_analysis_from_batch_processor(
        global_preprocessed_images=global_preprocessed_images,
        per_nucleus_preprocessed_images=per_nucleus_preprocessed_images,
        masks=masks,
        image_names=image_names,
        output_dir=str(run_output),
        use_image_aggregation=use_image_aggregation,
        config=config._config
      )
      
      # Restore original functions
      if callable(_orig_extract):
        _fe.extract_experimental_metadata = _orig_extract
      if callable(_orig_ap_extract):
        _ap.extract_experimental_metadata = _orig_ap_extract
    print("\n✓ DNA condensation analysis completed successfully!")
    print(f"Results saved to: {run_output}/")
    
    # Restore original extractor to avoid side effects
    if callable(_orig_extract):
      _fe.extract_experimental_metadata = _orig_extract
    if callable(_orig_ap_extract):
      _ap.extract_experimental_metadata = _orig_ap_extract
    
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
          
          # Use the same aggregation setting as statistical analysis for consistency
          use_image_aggregation_plots = config.get("analysis_settings", {}).get("use_image_aggregation", True)

          # Check for sufficient data for image-level aggregation ONCE before the loop
          if use_image_aggregation_plots:
              group_image_counts = features_df.groupby('condition')['image_name'].nunique()
              if not (group_image_counts >= 2).all():
                  print(f"\nInfo: Insufficient images per group for image-level statistical plots. Images per group: {group_image_counts.to_dict()}")
                  print("      Falling back to per-nucleus plotting for all single metric visualizations.")
                  warnings.warn("Using per-nucleus data for plotting (potential pseudoreplication)")
                  use_image_aggregation_plots = False # Fallback for this plotting section
          
          for metric_to_plot in metrics_to_plot:
            try:
              save_path = single_plots_dir / f"single_metric_{metric_to_plot}.png"
              visualizer.plot_single_metric_with_significance(
                df=features_df,
                metric=metric_to_plot,
                group_column='condition',
                save_path=save_path,
                use_image_aggregation=use_image_aggregation_plots
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
        
        # Determine which images to use for the panel
        panel_images = per_nucleus_preprocessed_images if per_nucleus_preprocessed_images else global_preprocessed_images

        features_df = pd.read_csv(run_output / "all_features.csv")
        out_dir = run_output
        out_path = out_dir / nuclei_cfg.get("save_name", "nuclei_panel.png")
        
        create_nuclei_panel(
          images=panel_images,
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


if __name__ == "__main__":
  main()