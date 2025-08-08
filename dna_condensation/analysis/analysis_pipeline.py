"""
Comprehensive DNA condensation analysis pipeline.

This module integrates feature extraction, statistical analysis, and visualization
to provide a complete analysis workflow for DNA condensation experiments.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dna_condensation.analysis.feature_extractor import DNACondensationFeatureExtractor, extract_experimental_metadata
from dna_condensation.analysis.statistical_analysis import DNACondensationStatistics, identify_key_features
from dna_condensation.visualization.visualize_statistics import StatisticalVisualizer

class DNACondensationAnalysisPipeline:
    """
    Complete analysis pipeline for DNA condensation experiments.
    
    This class orchestrates the entire analysis workflow:
    1. Feature extraction from segmented nuclei
    2. Quality control and data preparation  
    3. Statistical analysis and hypothesis testing
    4. Visualization of results
    5. Report generation
    """
    
    def __init__(self, output_dir: str = "analysis_results"):
        """
        Initialize analysis pipeline.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save all analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_extractor = DNACondensationFeatureExtractor()
        self.statistics = DNACondensationStatistics()
        self.visualizer = StatisticalVisualizer()
        
        # Data storage
        self.features_df = None
        self.metadata_df = None
        self.comparison_results = None
        self.pca_results = None
        
    def run_full_analysis(self, 
                         images: List[np.ndarray],
                         masks: List[np.ndarray], 
                         image_names: List[str],
                         group_column: str = 'condition') -> Dict:
        """
        Run the complete analysis pipeline.
        
        Parameters:
        -----------
        images : List[np.ndarray]
            List of preprocessed intensity images
        masks : List[np.ndarray]
            List of segmentation masks (labeled)
        image_names : List[str]
            Names/identifiers for each image
        group_column : str
            Column to use for statistical grouping
            
        Returns:
        --------
        Dict
            Dictionary containing all analysis results
        """
        print("=" * 60)
        print("DNA CONDENSATION ANALYSIS PIPELINE")
        print("=" * 60)
        
        # Step 1: Feature extraction
        print("\n1. FEATURE EXTRACTION")
        print("-" * 30)
        self.features_df = self.feature_extractor.extract_features_batch(
            images, masks, image_names
        )
        print(f"Extracted features for {len(self.features_df)} nuclei")
        
        # Step 2: Add experimental metadata
        print("\n2. EXPERIMENTAL METADATA")
        print("-" * 30)
        self.metadata_df = extract_experimental_metadata(image_names)
        
        # Merge features with metadata
        self.features_df = self.features_df.merge(
            self.metadata_df, on='image_name', how='left'
        )
        print("Merged features with experimental metadata")
        print(f"Groups found: {', '.join(self.features_df[group_column].unique())}")
        
        # Step 3: Quality control
        print("\n3. QUALITY CONTROL")
        print("-" * 30)
        self.features_df = self.statistics.quality_control(self.features_df)
        
        # Step 4: Statistical analysis
        print("\n4. STATISTICAL ANALYSIS")
        print("-" * 30)
        
        # Get feature columns for analysis
        feature_cols = self._get_analysis_features()
        print(f"Analyzing {len(feature_cols)} features")
        
        # Descriptive statistics
        desc_stats = self.statistics.descriptive_statistics(
            self.features_df, group_column
        )
        desc_stats.to_csv(self.output_dir / 'descriptive_statistics.csv', index=False)
        
        # Group comparisons
        self.comparison_results = self.statistics.compare_groups(
            self.features_df, feature_cols, group_column
        )
        self.comparison_results.to_csv(self.output_dir / 'group_comparisons.csv', index=False)
        
        # Pairwise comparisons
        pairwise_results = self.statistics.pairwise_comparisons(
            self.features_df, feature_cols, group_column
        )
        pairwise_results.to_csv(self.output_dir / 'pairwise_comparisons.csv', index=False)
        
        # Multivariate analysis
        self.pca_results = self.statistics.multivariate_analysis(
            self.features_df, feature_cols, group_column
        )
        
        # Step 5: Visualization
        print("\n5. VISUALIZATION")
        print("-" * 30)
        
        # Create comprehensive visualization report
        figures = self.visualizer.create_comprehensive_report(
            self.features_df,
            self.comparison_results,
            self.pca_results['pca'],
            feature_cols,
            group_column,
            self.output_dir
        )
        
        # Step 6: Generate summary report
        print("\n6. SUMMARY REPORT")
        print("-" * 30)
        summary_report = self.statistics.generate_summary_report(
            self.features_df, feature_cols, group_column
        )
        
        # Save summary report
        with open(self.output_dir / 'analysis_summary.txt', 'w') as f:
            f.write(summary_report)
        
        print(summary_report)
        
        # Step 7: Key findings
        print("\n7. KEY FINDINGS")
        print("-" * 30)
        key_features = identify_key_features(self.comparison_results)
        print(f"Top discriminative features:")
        for i, feature in enumerate(key_features[:10], 1):
            row = self.comparison_results[self.comparison_results['feature'] == feature].iloc[0]
            print(f"  {i:2d}. {feature}: effect size = {row['effect_size']:.3f}, p = {row['p_value']:.4f}")
        
        # Save all data
        self.features_df.to_csv(self.output_dir / 'all_features.csv', index=False)
        
        print(f"\nAll results saved to: {self.output_dir}")
        
        return {
            'features': self.features_df,
            'metadata': self.metadata_df,
            'descriptive_stats': desc_stats,
            'group_comparisons': self.comparison_results,
            'pairwise_comparisons': pairwise_results,
            'pca_results': self.pca_results,
            'key_features': key_features,
            'figures': figures,
            'summary_report': summary_report
        }
    
    def _get_analysis_features(self) -> List[str]:
        """
        Identify numeric feature columns suitable for statistical analysis.
        
        Excludes metadata columns like identifiers, coordinates, and experimental
        grouping variables to focus on biological/morphological measurements.
        
        Returns:
        --------
        List[str]
            List of feature column names for analysis
        """
        # Define columns to exclude from statistical analysis
        exclude_cols = {
            # Identifiers and coordinates (not biological features)
            'image_name', 'nucleus_id', 'centroid_x', 'centroid_y',
            # Experimental metadata (grouping variables, not features)
            'dk_group', 'dk_number', 'condition', 'well', 'timepoint'
        }
        
        # Select numeric columns that represent biological measurements
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        return feature_cols
    
    def analyze_specific_comparison(self, 
                                  group1: str, 
                                  group2: str,
                                  group_column: str = 'condition') -> Dict:
        """
        Perform detailed analysis for a specific pair of groups.
        
        Parameters:
        -----------
        group1, group2 : str
            Groups to compare
        group_column : str
            Column containing group information
            
        Returns:
        --------
        Dict
            Detailed comparison results
        """
        # Filter data for these two groups
        comparison_data = self.features_df[
            self.features_df[group_column].isin([group1, group2])
        ].copy()
        
        if len(comparison_data) == 0:
            raise ValueError(f"No data found for groups {group1} and {group2}")
        
        print(f"\nDETAILED COMPARISON: {group1} vs {group2}")
        print("=" * 50)
        
        feature_cols = self._get_analysis_features()
        
        # Perform statistical comparison
        comparison_results = self.statistics.compare_groups(
            comparison_data, feature_cols, group_column
        )
        
        # Get significant features
        p_col = 'p_corrected' if 'p_corrected' in comparison_results.columns else 'p_value'
        significant_features = comparison_results[
            comparison_results[p_col] < 0.05
        ].sort_values('effect_size', ascending=False)
        
        print(f"\nSignificant differences found in {len(significant_features)} features:")
        for _, row in significant_features.head(10).iterrows():
            print(f"  {row['feature']}: effect size = {row['effect_size']:.3f}, p = {row[p_col]:.4f}")
        
        # Create specific visualizations
        output_subdir = self.output_dir / f"{group1}_vs_{group2}"
        output_subdir.mkdir(exist_ok=True)
        
        if len(significant_features) > 0:
            top_features = significant_features.head(6)['feature'].tolist()
            
            # Plot distributions of top features
            fig = self.visualizer.plot_feature_distributions(
                comparison_data, top_features, group_column,
                output_subdir / 'top_features_distributions.png'
            )
            
        return {
            'comparison_data': comparison_data,
            'comparison_results': comparison_results,
            'significant_features': significant_features,
            'n_group1': len(comparison_data[comparison_data[group_column] == group1]),
            'n_group2': len(comparison_data[comparison_data[group_column] == group2])
        }
    
    def get_biological_interpretation(self) -> str:
        """
        Generate biological interpretation of results.
        """
        if self.comparison_results is None:
            return "No analysis results available"
            
        interpretation = []
        interpretation.append("BIOLOGICAL INTERPRETATION")
        interpretation.append("=" * 40)
        interpretation.append("")
        
        # Get significant results (align with summary logic)
        comp = self.comparison_results
        if 'significant' in comp.columns:
            significant = comp[comp['significant'] == True]
            criterion = "BH-corrected (FDR)"
        else:
            p_col = 'p_corrected' if 'p_corrected' in comp.columns else 'p_value'
            significant = comp[comp[p_col] < 0.05]
            criterion = f"{p_col} < 0.05"
        
        if len(significant) == 0:
            interpretation.append("No statistically significant differences found between groups.")
            interpretation.append("This could indicate:")
            interpretation.append("- Similar DNA condensation patterns across conditions")
            interpretation.append("- Need for larger sample sizes")
            interpretation.append("- Requirement for different analytical approaches")
            return "\n".join(interpretation)
        
        interpretation.append(f"Found {len(significant)} significant differences ({criterion}).")
        interpretation.append("")
        
        # Analyze feature categories
        intensity_features = significant[significant['feature'].str.contains('intensity|cv|entropy')]
        morphology_features = significant[significant['feature'].str.contains('area|perimeter|eccentricity|solidity')]
        spatial_features = significant[significant['feature'].str.contains('radial|center')]
        texture_features = significant[significant['feature'].str.contains('glcm|granulometry')]
        
        if len(intensity_features) > 0:
            interpretation.append(f"INTENSITY FEATURES ({len(intensity_features)} significant):")
            interpretation.append("- Differences in DNA staining intensity distribution")
            interpretation.append("- May indicate varying levels of chromatin condensation")
            interpretation.append("- Higher CV suggests more heterogeneous DNA organization")
            interpretation.append("")
        
        if len(morphology_features) > 0:
            interpretation.append(f"MORPHOLOGICAL FEATURES ({len(morphology_features)} significant):")
            interpretation.append("- Changes in nuclear shape and size")
            interpretation.append("- May reflect altered chromatin organization")
            interpretation.append("- Could indicate cell cycle or differentiation effects")
            interpretation.append("")
        
        if len(spatial_features) > 0:
            interpretation.append(f"SPATIAL FEATURES ({len(spatial_features)} significant):")
            interpretation.append("- Differences in radial DNA distribution")
            interpretation.append("- Suggests peripheral vs central chromatin organization")
            interpretation.append("- May indicate heterochromatin redistribution")
            interpretation.append("")
        
        if len(texture_features) > 0:
            interpretation.append(f"TEXTURE FEATURES ({len(texture_features)} significant):")
            interpretation.append("- Changes in local DNA organization patterns")
            interpretation.append("- Reflects chromatin clustering and granularity")
            interpretation.append("- Indicates altered higher-order chromatin structure")
            interpretation.append("")
        
        # Overall conclusion
        interpretation.append("CONCLUSION:")
        interpretation.append("The analysis reveals measurable differences in DNA condensation")
        interpretation.append("patterns between experimental conditions, suggesting that the")
        interpretation.append("tested factors influence chromatin organization in detectable ways.")
        
        return "\n".join(interpretation)


def run_analysis_from_batch_processor(images: List[np.ndarray],
                                    masks: List[np.ndarray],
                                    image_names: List[str],
                                    output_dir: str = "dna_condensation_analysis") -> Dict:
    """
    Convenience function to run analysis directly from batch processor results.
    
    Parameters:
    -----------
    images : List[np.ndarray]
        Preprocessed images from batch processor
    masks : List[np.ndarray] 
        Segmentation masks from batch processor
    image_names : List[str]
        Image names from batch processor
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    Dict
        Complete analysis results
    """
    # Initialize pipeline
    pipeline = DNACondensationAnalysisPipeline(output_dir)
    
    # Run full analysis
    results = pipeline.run_full_analysis(images, masks, image_names)
    
    # Generate biological interpretation
    biological_interpretation = pipeline.get_biological_interpretation()
    with open(Path(output_dir) / 'biological_interpretation.txt', 'w') as f:
        f.write(biological_interpretation)
    
    print("\n" + biological_interpretation)
    
    return results
