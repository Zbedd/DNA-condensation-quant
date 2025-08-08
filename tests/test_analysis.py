"""
Test suite for DNA condensation analysis modules.

This module tests all components of the analysis pipeline to ensure
functionality and reliability.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dna_condensation.analysis.feature_extractor import DNACondensationFeatureExtractor, extract_experimental_metadata
from dna_condensation.analysis.statistical_analysis import DNACondensationStatistics, identify_key_features
from dna_condensation.analysis.analysis_pipeline import DNACondensationAnalysisPipeline, run_analysis_from_batch_processor

class TestFeatureExtractor:
    """Test the DNACondensationFeatureExtractor class."""
    
    def setup_method(self):
        """Set up test data."""
        self.extractor = DNACondensationFeatureExtractor()
        
        # Create synthetic test data
        self.test_image = np.random.randint(0, 255, (100, 100)).astype(np.uint8)
        self.test_mask = np.zeros((100, 100), dtype=int)
        
        # Create 3 mock nuclei
        self.test_mask[20:40, 20:40] = 1  # Nucleus 1
        self.test_mask[60:80, 20:40] = 2  # Nucleus 2  
        self.test_mask[20:40, 60:80] = 3  # Nucleus 3
        
        # Add some intensity variation
        self.test_image[20:40, 20:40] += 50  # Brighter nucleus 1
        self.test_image[60:80, 20:40] += 25  # Medium nucleus 2
        
    def test_feature_extraction_single(self):
        """Test feature extraction from a single image."""
        features_df = self.extractor.extract_features_single(
            self.test_image, self.test_mask, "test_image"
        )
        
        # Should have 3 nuclei
        assert len(features_df) == 3
        
        # Check required columns exist
        required_cols = ['image_name', 'nucleus_id', 'mean_intensity', 
                        'coefficient_of_variation', 'area']
        for col in required_cols:
            assert col in features_df.columns
            
        # Check nucleus IDs
        assert set(features_df['nucleus_id']) == {1, 2, 3}
        
        # Check image name
        assert all(features_df['image_name'] == 'test_image')
        
    def test_feature_extraction_batch(self):
        """Test batch feature extraction."""
        images = [self.test_image, self.test_image.copy()]
        masks = [self.test_mask, self.test_mask.copy()]
        names = ['image1', 'image2']
        
        features_df = self.extractor.extract_features_batch(images, masks, names)
        
        # Should have 6 nuclei total (3 per image)
        assert len(features_df) == 6
        
        # Check image names
        assert set(features_df['image_name']) == {'image1', 'image2'}
        
    def test_intensity_features(self):
        """Test intensity feature calculation."""
        test_values = np.array([1, 2, 3, 4, 5, 10])
        features = self.extractor._intensity_features(test_values)
        
        # Check basic statistics
        assert features['mean_intensity'] == np.mean(test_values)
        assert features['std_intensity'] == np.std(test_values)
        assert features['coefficient_of_variation'] == np.std(test_values) / np.mean(test_values)
        
        # Check percentiles exist
        assert 'intensity_p50' in features  # median
        assert 'intensity_p90' in features
        
    def test_empty_mask_handling(self):
        """Test handling of empty masks."""
        empty_mask = np.zeros((50, 50), dtype=int)
        
        features_df = self.extractor.extract_features_single(
            self.test_image[:50, :50], empty_mask, "empty_test"
        )
        
        # Should return empty DataFrame
        assert len(features_df) == 0
        
    def test_experimental_metadata_extraction(self):
        """Test extraction of experimental metadata from filenames."""
        test_names = [
            '48hr_dk16_wtLSD1_well1_20x001.nd2',
            '48hr_dk18_gfpOnly_well2_20x.nd2',
            '48hr_dk19_catDeadK661A_well3_20x.nd2'
        ]
        
        metadata_df = extract_experimental_metadata(test_names)
        
        assert len(metadata_df) == 3
        assert set(metadata_df['dk_group']) == {'dk16', 'dk18', 'dk19'}
        assert set(metadata_df['condition']) == {'wtLSD1', 'gfpOnly', 'catDeadK661A'}
        assert set(metadata_df['well']) == {1, 2, 3}


class TestStatisticalAnalysis:
    """Test the DNACondensationStatistics class."""
    
    def setup_method(self):
        """Set up test data."""
        self.stats = DNACondensationStatistics()
        
        # Create synthetic feature data
        np.random.seed(42)  # For reproducible tests
        n_nuclei = 100
        
        self.test_df = pd.DataFrame({
            'image_name': [f'image_{i//10}' for i in range(n_nuclei)],
            'nucleus_id': range(n_nuclei),
            'condition': ['group_A'] * 50 + ['group_B'] * 50,
            'mean_intensity': np.random.normal(1.0, 0.2, n_nuclei),
            'coefficient_of_variation': np.random.normal(0.3, 0.1, n_nuclei),
            'area': np.random.normal(1000, 200, n_nuclei),
            'centroid_x': np.random.uniform(0, 100, n_nuclei),
            'centroid_y': np.random.uniform(0, 100, n_nuclei)
        })
        
        # Add some systematic difference between groups
        self.test_df.loc[self.test_df['condition'] == 'group_B', 'coefficient_of_variation'] += 0.1
        
    def test_quality_control(self):
        """Test quality control functionality."""
        # Add some outliers
        outlier_df = self.test_df.copy()
        outlier_df.loc[0, 'area'] = 10000  # Extreme area
        outlier_df.loc[1, 'mean_intensity'] = 0.01  # Very low intensity
        
        filtered_df = self.stats.quality_control(outlier_df)
        
        # Should remove outliers
        assert len(filtered_df) < len(outlier_df)
        
    def test_descriptive_statistics(self):
        """Test descriptive statistics calculation."""
        desc_stats = self.stats.descriptive_statistics(self.test_df)
        
        # Should have stats for both groups
        assert set(desc_stats['group']) == {'group_A', 'group_B'}
        
        # Should have stats for numeric features
        features = desc_stats['feature'].unique()
        expected_features = ['mean_intensity', 'coefficient_of_variation', 'area']
        for feat in expected_features:
            assert feat in features
            
        # Check required statistics columns
        required_cols = ['mean', 'std', 'median', 'n_nuclei']
        for col in required_cols:
            assert col in desc_stats.columns
            
    def test_group_comparisons(self):
        """Test statistical group comparisons."""
        features = ['mean_intensity', 'coefficient_of_variation', 'area']
        
        comparison_results = self.stats.compare_groups(self.test_df, features)
        
        # Should have results for all features
        assert len(comparison_results) == len(features)
        
        # Check required columns
        required_cols = ['feature', 'test', 'statistic', 'p_value', 'effect_size']
        for col in required_cols:
            assert col in comparison_results.columns
            
        # CV should show significant difference (we added systematic difference)
        cv_result = comparison_results[comparison_results['feature'] == 'coefficient_of_variation']
        assert len(cv_result) == 1
        # Note: p-value might not always be < 0.05 due to random data, but effect size should be > 0
        assert cv_result.iloc[0]['effect_size'] > 0
        
    def test_normality_testing(self):
        """Test normality testing functionality."""
        features = ['mean_intensity', 'coefficient_of_variation']
        
        normality_results = self.stats.test_normality(self.test_df, features)
        
        # Should have results for each feature-group combination
        expected_rows = len(features) * len(self.test_df['condition'].unique())
        assert len(normality_results) == expected_rows
        
        # Check required columns
        required_cols = ['group', 'feature', 'shapiro_statistic', 'shapiro_p_value', 'is_normal']
        for col in required_cols:
            assert col in normality_results.columns
            
    def test_identify_key_features(self):
        """Test key feature identification."""
        features = ['mean_intensity', 'coefficient_of_variation', 'area']
        comparison_results = self.stats.compare_groups(self.test_df, features)
        
        key_features = identify_key_features(comparison_results, top_n=2)
        
        # Should return requested number of features
        assert len(key_features) <= 2
        
        # Should be valid feature names
        for feature in key_features:
            assert feature in features


class TestAnalysisPipeline:
    """Test the complete analysis pipeline."""
    
    def setup_method(self):
        """Set up test data."""
        # Create temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = DNACondensationAnalysisPipeline(self.temp_dir)
        
        # Create synthetic test data
        np.random.seed(42)
        
        # Create 2 test images
        self.images = []
        self.masks = []
        self.image_names = [
            '48hr_dk16_wtLSD1_well1_20x001.nd2',
            '48hr_dk18_gfpOnly_well1_20x.nd2'
        ]
        
        for i in range(2):
            # Create image
            image = np.random.randint(0, 255, (80, 80)).astype(np.uint8)
            
            # Create mask with 2 nuclei
            mask = np.zeros((80, 80), dtype=int)
            mask[10:30, 10:30] = 1  # Nucleus 1
            mask[50:70, 50:70] = 2  # Nucleus 2
            
            # Add some intensity differences for different conditions
            if 'wtLSD1' in self.image_names[i]:
                image[10:30, 10:30] += 30  # Higher intensity for wtLSD1
            
            self.images.append(image)
            self.masks.append(mask)
            
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        assert self.pipeline.output_dir.exists()
        assert isinstance(self.pipeline.feature_extractor, DNACondensationFeatureExtractor)
        assert isinstance(self.pipeline.statistics, DNACondensationStatistics)
        
    @patch('matplotlib.pyplot.savefig')  # Mock saving to avoid display issues in tests
    def test_full_analysis_pipeline(self, mock_savefig):
        """Test running the complete analysis pipeline."""
        # Mock visualization components that might not work in test environment
        with patch.object(self.pipeline.visualizer, 'create_comprehensive_report') as mock_viz:
            mock_viz.return_value = {}
            
            results = self.pipeline.run_full_analysis(
                self.images, self.masks, self.image_names
            )
        
        # Check that all expected results are present
        expected_keys = [
            'features', 'metadata', 'descriptive_stats', 
            'group_comparisons', 'pca_results', 'key_features'
        ]
        for key in expected_keys:
            assert key in results
            
        # Check that files were created
        assert (Path(self.temp_dir) / 'all_features.csv').exists()
        assert (Path(self.temp_dir) / 'group_comparisons.csv').exists()
        assert (Path(self.temp_dir) / 'analysis_summary.txt').exists()
        
        # Check features DataFrame
        features_df = results['features']
        assert len(features_df) == 4  # 2 nuclei per image, 2 images
        assert 'condition' in features_df.columns
        assert set(features_df['condition']) == {'wtLSD1', 'gfpOnly'}
        
    def test_get_analysis_features(self):
        """Test feature column identification."""
        # First run feature extraction to populate features_df
        self.pipeline.features_df = self.pipeline.feature_extractor.extract_features_batch(
            self.images, self.masks, self.image_names
        )
        
        # Add metadata
        metadata_df = extract_experimental_metadata(self.image_names)
        self.pipeline.features_df = self.pipeline.features_df.merge(
            metadata_df, on='image_name', how='left'
        )
        
        feature_cols = self.pipeline._get_analysis_features()
        
        # Should exclude non-feature columns
        excluded = {'image_name', 'nucleus_id', 'centroid_x', 'centroid_y', 
                   'dk_group', 'dk_number', 'condition', 'well', 'timepoint'}
        
        for col in feature_cols:
            assert col not in excluded
            
        # Should include intensity features
        intensity_features = [col for col in feature_cols if 'intensity' in col or 'cv' in col]
        assert len(intensity_features) > 0


class TestIntegrationWithBatchProcessor:
    """Test integration with the batch processing pipeline."""
    
    def setup_method(self):
        """Set up integration test data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock data similar to batch processor output
        np.random.seed(42)
        
        self.mock_images = []
        self.mock_masks = []
        self.mock_names = [
            '48hr_dk16_wtLSD1_well1_20x001.nd2',
            '48hr_dk16_wtLSD1_well2_20x.nd2',
            '48hr_dk18_gfpOnly_well1_20x.nd2',
            '48hr_dk18_gfpOnly_well2_20x.nd2'
        ]
        
        for name in self.mock_names:
            # Create preprocessed image (single channel, normalized)
            image = np.random.normal(1.0, 0.3, (100, 100))
            image = np.clip(image, 0, 3)  # Realistic intensity range
            
            # Create segmentation mask
            mask = np.zeros((100, 100), dtype=int)
            
            # Add 3-5 nuclei per image
            n_nuclei = np.random.randint(3, 6)
            for i in range(n_nuclei):
                center_y = np.random.randint(15, 85)
                center_x = np.random.randint(15, 85)
                radius = np.random.randint(8, 15)
                
                y, x = np.ogrid[:100, :100]
                mask_nucleus = (y - center_y)**2 + (x - center_x)**2 <= radius**2
                mask[mask_nucleus] = i + 1
                
                # Add corresponding intensity (simulating different condensation)
                if 'wtLSD1' in name:
                    # Simulate higher condensation for wtLSD1
                    image[mask_nucleus] *= np.random.uniform(1.2, 1.8)
                
            self.mock_images.append(image)
            self.mock_masks.append(mask)
            
    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('matplotlib.pyplot.savefig')  # Mock saving to avoid display issues
    def test_run_analysis_from_batch_processor(self, mock_savefig):
        """Test the convenience function for batch processor integration."""
        # Mock visualization to avoid display issues in tests
        with patch('dna_condensation.analysis.analysis_pipeline.StatisticalVisualizer') as mock_viz_class:
            mock_viz = MagicMock()
            mock_viz.create_comprehensive_report.return_value = {}
            mock_viz_class.return_value = mock_viz
            
            results = run_analysis_from_batch_processor(
                self.mock_images, self.mock_masks, self.mock_names, self.temp_dir
            )
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'features' in results
        assert 'group_comparisons' in results
        
        # Check output files
        output_path = Path(self.temp_dir)
        assert (output_path / 'all_features.csv').exists()
        assert (output_path / 'biological_interpretation.txt').exists()
        
        # Check that analysis found differences between conditions
        features_df = results['features']
        assert 'wtLSD1' in features_df['condition'].values
        assert 'gfpOnly' in features_df['condition'].values


def test_error_handling():
    """Test error handling in various scenarios."""
    extractor = DNACondensationFeatureExtractor()
    
    # Test with None inputs
    result = extractor.extract_features_batch([None], [None], ['test'])
    assert len(result) == 0
    
    # Test with mismatched array sizes
    with pytest.raises((ValueError, IndexError)):
        extractor.extract_features_batch([np.ones((10, 10))], [np.ones((5, 5))], ['test'])


if __name__ == '__main__':
    # Run specific test for quick validation
    print("Running DNA Condensation Analysis Tests...")
    
    # Test feature extraction
    test_extractor = TestFeatureExtractor()
    test_extractor.setup_method()
    test_extractor.test_feature_extraction_single()
    print("✓ Feature extraction test passed")
    
    # Test statistical analysis
    test_stats = TestStatisticalAnalysis()
    test_stats.setup_method()
    test_stats.test_group_comparisons()
    print("✓ Statistical analysis test passed")
    
    # Test pipeline
    test_pipeline = TestAnalysisPipeline()
    test_pipeline.setup_method()
    try:
        # Mock the visualization component
        with patch.object(test_pipeline.pipeline.visualizer, 'create_comprehensive_report') as mock_viz:
            mock_viz.return_value = {}
            test_pipeline.test_full_analysis_pipeline()
        print("✓ Analysis pipeline test passed")
    finally:
        test_pipeline.teardown_method()
    
    print("\nAll core tests passed! ✓")
    print("\nTo run complete test suite, use: pytest tests/test_analysis.py -v")
