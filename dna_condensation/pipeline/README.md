# DNA Condensation Pipeline

This directory contains the core pipeline orchestration for DNA condensation analysis, supporting both ND2 microscopy files and BBBC022 validation datasets.

## Architecture Overview

The pipeline follows a **"normalize then unify"** design pattern that handles different input sources through source-specific preparation followed by unified batch processing.

```
┌─────────────────┐    ┌─────────────────┐
│   ND2 Files     │    │  BBBC022 Data   │
└─────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│prepare_nd2_     │    │prepare_bbbc022_ │
│inputs()         │    │inputs()         │
└─────────────────┘    └─────────────────┘
         │                        │
         └────────┬─────────────────┘
                  ▼
         ┌─────────────────┐
         │run_unified_     │
         │pipeline()       │
         └─────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Batch Processing│
         │ Functions       │
         └─────────────────┘
```

## Files

### `batch_processor.py`
Main pipeline orchestration with three key functions:
- **`main()`**: Entry point that routes to appropriate input handler
- **Source-specific preparation**: `prepare_nd2_inputs()` and `prepare_bbbc022_inputs()`
- **Unified processing**: `run_unified_pipeline()` for source-agnostic batch processing

### `config.py`
Configuration management class that loads and validates settings from `config.yaml`.

### `config.yaml`
Central configuration file controlling all pipeline behavior including input sources, preprocessing methods, segmentation parameters, and output settings.

## Data Flow

### ND2 Pipeline
```
Raw ND2 Files → ND2Objects → Z-Stack Collapse → Raw Images Array
│
├─ Input: Multiple .nd2 files from microscopy
├─ Processing: get_nd2_objects() → batch_collapse_z_axis()
├─ Output: numpy arrays + filenames
└─ Metadata: None (extracted from filenames later)
```

### BBBC022 Pipeline
```
BBBC022 Dataset → Load Images → Type Conversion → Raw Images Array
│
├─ Input: Broad Bioimage Benchmark Collection
├─ Processing: load_bbbc022_images() → convert_to_uint8()
├─ Output: numpy arrays + enhanced metadata
└─ Metadata: Well mappings + experimental conditions
```

### Unified Processing (Both Sources)
```
Raw Images Array → Preprocessing → Segmentation → Analysis
│
├─ bulk_preprocess_images(): Background correction, normalization
├─ bulk_segment_images(): Nuclear segmentation with size filtering
├─ per_nucleus_intensity_normalization(): Optional per-nucleus processing
└─ run_analysis_from_batch_processor(): Feature extraction + statistics
```

## Key Architecture Features

### 1. Source-Agnostic Design
After preparation, both ND2 and BBBC022 data become identical:
- `raw_images[]`: Array of numpy image arrays
- `image_names[]`: Array of image identifiers
- `metadata`: Either None (ND2) or structured data (BBBC022)

### 2. Batch Processing Functions
Three core functions process entire image arrays simultaneously:

**`bulk_preprocess_images(raw_images[])`**
- Applies background correction, deconvolution, intensity normalization
- Processes all images with identical parameters
- Returns preprocessed image array

**`bulk_segment_images(preprocessed[])`**
- Nuclear segmentation using configurable methods (OTSU, watershed, YOLO)
- Size-based filtering to remove artifacts
- Returns label arrays with unique nucleus IDs

**`run_analysis_from_batch_processor(final[], masks[], names[])`**
- Feature extraction from all nuclei across all images
- Statistical analysis and group comparisons
- Generates comprehensive analysis reports

### 3. Memory-Intensive Design
- All images loaded into memory simultaneously
- Optimized for throughput over memory efficiency
- Enables vectorized operations across entire datasets

## Configuration Control

### Input Source Selection
```yaml
input_source: "nd2"        # Options: "nd2" or "bbbc022"
```

### ND2 Processing
```yaml
raw_nd2_path: "path/to/nd2/files"     # Folder containing .nd2 files
nd2_output_path: "./output"           # Analysis results destination
z_collapse_method: "mean"             # Z-stack collapse method
```

### BBBC022 Processing
```yaml
validation_output_path: "dna_condensation/validation/output"
bbbc022_settings:
  count: 20                          # Number of images to sample
  channels: ['OrigHoechst']          # DNA staining channels
  group_mapping:                     # Control vs treatment wells
    control: ["A01", "A02", ...]
    treatment: ["B01", "B02", ...]
```

### Processing Parameters
```yaml
segmentation_channel_index: 0        # Channel for nucleus detection
segmentation_method: "otsu"          # Segmentation algorithm
preprocessing:
  background_correction: true        # Enable background correction
  intensity_normalization: true     # Enable intensity normalization
  per_nucleus_normalization: false  # Per-nucleus intensity normalization
```

## Pipeline Limitations

### Image Selection Control
- **No subset selection**: Processes ALL files in specified directory
- **No pattern filtering**: Cannot filter by filename patterns
- **No quality filtering**: Cannot skip corrupted/problematic files
- **No progressive processing**: Cannot process in smaller batches

### Memory Requirements
- **High memory usage**: All images loaded simultaneously
- **No streaming support**: Cannot process datasets larger than available RAM
- **Fixed batch size**: No configurable memory management

## Usage Examples

### Basic ND2 Processing
```python
from dna_condensation.pipeline.batch_processor import main

# Configure input_source: "nd2" in config.yaml
# Set raw_nd2_path to your ND2 folder
result = main()  # Processes all .nd2 files in folder
```

### Visualization Mode
```python
# Return processed images for visualization
result = main(return_images=True)

raw_images = result['raw_images']
final_images = result['final_images']
masks = result['masks']
```

### BBBC022 Validation
```python
# Configure input_source: "bbbc022" in config.yaml
# Set group mappings for control vs treatment
result = main()  # Processes configured BBBC022 subset
```

## Output Structure

Analysis results are saved with timestamps in the format:
```
output/
└── dna_condensation_analysis_results/
    └── {input_source}_{YYYYMMDD_HHMMSS}/
        ├── all_features.csv              # Per-nucleus features
        ├── group_comparisons.csv         # Statistical comparisons
        ├── analysis_summary.txt          # Human-readable summary
        └── config_{timestamp}.txt        # Archived configuration
```

## Integration Points

### Visualization
The pipeline integrates with `visualize_images.py` through:
- `main(return_images=True)`: Returns all processing stages
- Interactive visualization of preprocessing steps
- Toggleable segmentation overlays

### Analysis
Results feed into downstream analysis tools:
- Statistical visualization (`visualize_statistics.py`)
- Nuclei panel generation (`visualize_nuclei.py`)
- Custom analysis scripts via exported CSV files

## Future Enhancements

Potential improvements to address current limitations:
- **Selective processing**: Image subset selection and pattern filtering
- **Memory management**: Configurable batch sizes and streaming support
- **Quality control**: Automatic detection and handling of problematic files
- **Progressive analysis**: Processing in smaller chunks for large datasets
