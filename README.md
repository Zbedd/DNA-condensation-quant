# DNA Condensation Quantification Pipeline

A comprehensive analysis pipeline for quantifying DNA condensation patterns in fluorescence microscopy images. This tool automates the process of segmenting nuclei, extracting morphological and texture features, and performing statistical comparisons to identify changes in chromatin organization.

## Overview

This pipeline processes ND2 microscopy files to measure DNA condensation through multiple feature classes including morphological properties, texture characteristics, spatial organization patterns, and intensity distributions. The analysis is designed for comparing experimental conditions and identifying subtle changes in nuclear architecture that may indicate altered chromatin states.

**Key Capabilities:**
- Multi-channel ND2 file processing with z-stack collapse
- Automated nucleus segmentation using multiple algorithms (AI-based YOLO, watershed, Otsu)
- Comprehensive feature extraction (texture, morphology, spatial, intensity)
- Per-nucleus intensity normalization for homogeneity analysis
- Statistical analysis with multiple comparison correction
- Interactive visualization and reporting

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for accelerated processing)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Zbedd/DNA-condensation-quant.git
cd DNA-condensation-quant
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install numpy pandas scipy scikit-image scikit-learn matplotlib seaborn
pip install opencv-python nd2reader pyyaml
pip install ultralytics  # For YOLO segmentation (optional)
```

## Quick Start

### 1. Configure Analysis Parameters

Edit `dna_condensation/pipeline/config.yaml` to specify your data paths and analysis settings:

```yaml
# File paths
raw_nd2_path: "/path/to/your/nd2/files"
output_path: "./output"

# Microscope settings
nd2_selection_settings:
    nuclear_channel_index: 1   # DNA staining channel in ND2 (e.g., Channel 1)
bbbc022_settings:
    nuclear_channel_index: 0   # Single-channel BBBC022 (use 0)
segmentation_method: "otsu"    # Options: "yolo", "watershed", "otsu"

# Analysis parameters
preprocessing_methods: ["background_correction", "intensity_normalization"]
feature_selection: ["morphological", "texture", "intensity"]
```

### 2. Run Batch Processing

Process all ND2 files in your specified directory:

```python
from dna_condensation.pipeline.batch_processor import DNACondensationBatchProcessor

# Initialize processor with your config
processor = DNACondensationBatchProcessor()

# Process all files
results = processor.run()
```

### 3. Alternative: Single File Analysis

For processing individual files with custom parameters:

```python
from dna_condensation.analysis.analysis_pipeline import DNACondensationPipeline

# Load and analyze single file
pipeline = DNACondensationPipeline()
features_df = pipeline.analyze_nd2_file(
    "sample.nd2", 
    condition="treatment_A",
    segmentation_method="otsu"
)
```

## Analysis Pipeline

### Image Processing Workflow

1. **ND2 File Loading**: Multi-channel z-stack extraction with metadata preservation
2. **Z-Stack Collapse**: Maximum intensity projection for each channel
3. **Preprocessing**: Background correction and intensity normalization
4. **Segmentation**: Nucleus detection using configurable algorithms
5. **Feature Extraction**: Comprehensive measurement across multiple domains
6. **Statistical Analysis**: Group comparisons with multiple testing correction

## Metrics Tracked

This pipeline extracts a comprehensive set of features to quantify nuclear changes. The metrics are grouped into categories based on their scientific purpose and their relationship with intensity normalization.

**Data Processing Pipeline:**
- **Globally preprocessed images**: Background corrected and globally normalized, but NOT per-nucleus normalized
- **Per-nucleus normalized images**: Each nucleus individually normalized to have mean intensity = 1.0

### Intensity-Based Metrics
These metrics directly measure the amount and distribution of the fluorescence signal. They are sensitive to absolute intensity levels and are calculated on **globally preprocessed** data (background corrected, globally normalized) that has **not** undergone per-nucleus normalization. This preserves relative intensity differences between nuclei for valid experimental comparisons.

- **Statistical Measures**: `mean_intensity`, `std_intensity`, `coefficient_of_variation` (CV), `min_intensity`, `max_intensity`, `intensity_range`
- **Distribution Shape**: `intensity_skewness`, `intensity_kurtosis`, `intensity_entropy`
- **Percentile-Based**: `intensity_p{10, 25, 50, 75, 90, 95, 99}`, `high_intensity_fraction`
- **Derived Intensity**: `nuclear_density`
- **Spatial Intensity**: `radial_shell_{0...N-1}`, `center_to_edge_ratio`
- **Granulometry**: `granulometry_spots_r{radius}`, `granulometry_area_r{radius}`, `granulometry_area_fraction_r{radius}`
- **DNA Condensation Metrics**: 
  - `ccp` (Chromatin Condensation Parameter): Edge fraction based on Sobel gradient magnitudes within each nucleus
  - `condensation_index` (CI): Combined metric using P95 intensity and nuclear area, normalized using control group statistics: CI = z(P95) - z(log area)

### Texture-Based Metrics
These metrics quantify the spatial arrangement and patterns of pixels within a nucleus, providing insight into chromatin texture (e.g., smooth, coarse, clumped). To ensure that these comparisons are not confounded by absolute brightness differences, these features are calculated on **per-nucleus normalized** data where each nucleus has been individually scaled to have mean intensity = 1.0.

- **GLCM Contrast**: `glcm_contrast_mean`, `glcm_contrast_std`
- **GLCM Dissimilarity**: `glcm_dissimilarity_mean`, `glcm_dissimilarity_std`
- **GLCM Homogeneity**: `glcm_homogeneity_mean`, `glcm_homogeneity_std`
- **GLCM Energy (Uniformity)**: `glcm_energy_mean`, `glcm_energy_std`
- **GLCM Correlation**: `glcm_correlation_mean`, `glcm_correlation_std`

### Morphological Metrics
These metrics describe the size and shape of the nucleus. They are derived from the segmentation mask and are independent of pixel intensity.

- **Size**: `area`, `perimeter`, `major_axis_length`, `minor_axis_length`, `equivalent_diameter`
- **Shape**: `eccentricity`, `solidity`, `extent`, `aspect_ratio`, `circularity`, `orientation`

### Feature Categories

**Morphological Features:**
- Basic measurements (area, perimeter, aspect ratio)
- Shape descriptors (eccentricity, solidity, circularity)
- Derived metrics for nuclear envelope characterization

**Texture Features (GLCM-based):**
- Homogeneity patterns indicating chromatin organization
- Contrast measurements for heterogeneity assessment
- Energy and entropy for structural complexity

**Spatial Features:**
- Radial intensity profiles from nuclear center
- Concentric shell analysis for spatial organization
- Edge-to-center intensity relationships

**Intensity Features:**
- Statistical distributions (mean, variance, skewness)
- High-intensity fraction analysis
- Coefficient of variation for uniformity assessment

## Configuration Options

### Segmentation Methods

**Otsu Thresholding (Recommended)**
- Fast, reliable for well-stained nuclei
- Conservative boundary detection
- Good for batch processing

**Watershed Segmentation**
- Better separation of touching nuclei
- More computationally intensive
- Suitable for crowded fields

**YOLO AI Segmentation**
- Highest accuracy for complex cases
- Requires GPU for reasonable speed
- Pre-trained on diverse microscopy data

### Preprocessing Strategies

**Standard Workflow** (cross-sample comparison):
```
background_correction → intensity_normalization
```

**Homogeneity Analysis Workflow** (within-nucleus patterns):
```
background_correction → per_nucleus_intensity_normalization
```
Notes on normalization and metrics:
- Texture features (GLCM) are computed on per-nucleus normalized intensities to emphasize pattern over brightness.
- Intensity distribution metrics including CV, P95, and CI components are computed on non-normalized data.
- CCP is computed as a percentile-thresholded Sobel edge fraction within each nucleus and uses the non-normalized image; its per-nucleus gradient percentile provides scale robustness.

**Deconvolution Workflow** (enhanced resolution):
```
deconvolution → background_correction → intensity_normalization
```

## Output Structure

Results are organized in timestamped directories:

```
output/
├── YYYY-MM-DD_HH-MM-SS_analysis/
│   ├── features/
│   │   ├── features_combined.csv           # All extracted features
│   │   ├── features_metadata.csv           # Per-image metadata
│   │   └── individual_files/               # Per-file feature tables
│   ├── statistics/
│   │   ├── group_comparisons.csv           # Statistical test results
│   │   ├── effect_sizes.csv                # Cohen's d calculations
│   │   └── significance_summary.csv        # Multiple comparison results
│   ├── visualizations/
│   │   ├── feature_distributions.png       # Violin plots by group
│   │   ├── comparison_summary.png          # Volcano plots
│   │   ├── correlation_matrix.png          # Feature relationships
│   │   └── nuclei_panels/                  # Representative nuclei
│   ├── config_archive.yaml                 # Analysis parameters used
│   └── processing_log.txt                  # Detailed execution log
```

## Advanced Usage

### Custom Feature Extraction

Extract specific feature subsets for focused analysis:

```python
from dna_condensation.analysis.feature_extractor import DNACondensationFeatureExtractor

extractor = DNACondensationFeatureExtractor()
features = extractor.extract_features(
    image, 
    labels, 
    feature_types=['morphological', 'texture']
)
```

### Statistical Analysis

Perform group comparisons with custom parameters:

```python
from dna_condensation.analysis.statistical_analysis import compare_groups

results = compare_groups(
    features_df,
    group_col='condition',
    features_to_test=['glcm_homogeneity_mean', 'area', 'circularity'],
    alpha=0.05,
    correction_method='fdr_bh'
)
```

### Visualization Customization

Generate publication-ready figures:

```python
from dna_condensation.visualization.visualize_statistics import plot_comparison_summary

plot_comparison_summary(
    comparison_results,
    save_path='./figures/treatment_comparison.png',
    title='Treatment vs Control Comparison',
    figsize=(12, 8)
)
```

## Performance Optimization

### GPU Acceleration
- Enable CUDA for YOLO segmentation: `use_gpu=True`
- Monitor GPU memory usage for large batch processing
- Consider image downsampling for initial parameter optimization

### Memory Management
- Process files individually for large datasets
- Use temporary file cleanup: `cleanup_temp=True`
- Monitor disk space in output directories

### Batch Processing Tips
- Use size filtering to remove debris: `min_size_percentage=20`
- Enable verbose logging for progress monitoring
- Consider parallel processing for independent files

## Troubleshooting

### Common Issues

**Low Segmentation Quality:**
- Adjust `nd2_selection_settings.nuclear_channel_index` (ND2) or `bbbc022_settings.nuclear_channel_index` (BBBC022) to the DNA staining channel
- Try different segmentation methods (otsu → watershed → yolo)
- Check image preprocessing parameters

**Memory Errors:**
- Reduce batch size or process files individually
- Enable temporary file cleanup
- Close unused applications

**Feature Extraction Failures:**
- Verify nucleus labels are properly generated
- Check for empty or corrupted image regions
- Review size filtering thresholds

### Validation Tools

Use the provided test scripts to validate your setup:

```bash
python test_segmentation_comparison.py  # Compare segmentation methods
python test_batch_processor_updated.py  # Validate full pipeline
```

## Contributing

This pipeline is designed for extensibility. Key areas for contribution:

- Additional feature extraction methods
- Alternative segmentation algorithms  
- Enhanced visualization capabilities
- Performance optimizations

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{dna_condensation_pipeline,
  title={DNA Condensation Quantification Pipeline},
  author={[Author Names]},
  year={2025},
  url={https://github.com/Zbedd/DNA-condensation-quant}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact the development team.