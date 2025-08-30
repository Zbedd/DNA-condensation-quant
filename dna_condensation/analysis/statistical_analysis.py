"""
Statistical analysis for DNA condensation data.

This module provides comprehensive statistical analysis tools for comparing
DNA condensation features across experimental conditions.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
from typing import Dict, List, Tuple, Optional, Union

class DNACondensationStatistics:
    """
    Statistical analysis tools for DNA condensation feature data.
    
    Provides methods for:
    - Descriptive statistics and quality control
    - Hypothesis testing (parametric and non-parametric)
    - Multiple comparison correction
    - Multivariate analysis (PCA, clustering)
    - Effect size calculations
    """
    
    def __init__(self, alpha: float = 0.05, use_image_aggregation: bool = True):
        """
        Initialize statistical analysis.
        
        Parameters:
        -----------
        alpha : float
            Significance level for hypothesis testing
        use_image_aggregation : bool
            If True, aggregate to image level before statistical testing to avoid pseudoreplication
        """
        self.alpha = alpha
        self.use_image_aggregation = use_image_aggregation
        self.results = {}
        
    def _aggregate_by_image(self, df: pd.DataFrame, group_column: str = 'condition') -> pd.DataFrame:
        """
        Aggregate nucleus-level data to image level to avoid pseudoreplication.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Nucleus-level feature dataframe
        group_column : str
            Column containing experimental group information
            
        Returns:
        --------
        pd.DataFrame
            Image-level aggregated data with median values per feature
        """
        if 'image_name' not in df.columns:
            warnings.warn("No 'image_name' column found. Cannot aggregate by image. Using per-nucleus data.")
            return df
            
        # Identify feature columns (exclude metadata)
        exclude_cols = {
            'image_name', 'nucleus_id', 'centroid_x', 'centroid_y',
            'dk_group', 'dk_number', 'condition', 'well', 'timepoint'
        }
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Group by image and experimental condition, take median of features
        groupby_cols = ['image_name', group_column]
        
        # Aggregate features using median (robust to outliers)
        agg_dict = {col: 'median' for col in feature_cols}
        df_aggregated = df.groupby(groupby_cols).agg(agg_dict).reset_index()
        
        print(f"Aggregated from {len(df)} nuclei to {len(df_aggregated)} images for statistical testing")
        
        return df_aggregated
        
    def quality_control(self, df: pd.DataFrame, 
                       outlier_threshold: float = 3.0) -> pd.DataFrame:
        """
        Perform quality control on feature data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature dataframe with nucleus-level data
        outlier_threshold : float
            Standard deviations for outlier detection
            
        Returns:
        --------
        pd.DataFrame
            Filtered dataframe with outliers removed
        """
        print("=== Quality Control ===")
        print(f"Initial data: {len(df)} nuclei")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print(f"Missing values found in {missing_counts[missing_counts > 0].to_dict()}")
        
        # Remove nuclei with extreme area (likely segmentation errors)
        if 'area' in df.columns:
            area_mean = df['area'].mean()
            area_std = df['area'].std()
            area_outliers = np.abs(df['area'] - area_mean) > outlier_threshold * area_std
            df = df[~area_outliers]
            print(f"Removed {area_outliers.sum()} nuclei with extreme area")
        
        # Remove nuclei with very low signal (skip for tiny datasets to avoid over-filtering in tests)
        if 'mean_intensity' in df.columns:
            if len(df) >= 10:
                intensity_outliers = df['mean_intensity'] < np.percentile(df['mean_intensity'], 1)
                df = df[~intensity_outliers]
                print(f"Removed {intensity_outliers.sum()} nuclei with very low signal")
            else:
                print("Skipped low-signal filtering for small sample size (<10 nuclei)")
        
        # Check normalization (if per-nucleus normalization was applied)
        if 'mean_intensity' in df.columns:
            overall_mean = df['mean_intensity'].mean()
            print(f"Overall mean intensity after QC: {overall_mean:.3f}")
            if abs(overall_mean - 1.0) > 0.1:
                warnings.warn(f"Mean intensity ({overall_mean:.3f}) differs from expected 1.0")
        
        print(f"Final data: {len(df)} nuclei")
        return df
    
    def descriptive_statistics(self, df: pd.DataFrame, 
                              group_column: str = 'condition') -> pd.DataFrame:
        """
        Generate descriptive statistics by group.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature dataframe (nucleus-level or image-level)
        group_column : str
            Column name for grouping (e.g., 'condition', 'dk_group')
            
        Returns:
        --------
        pd.DataFrame
            Descriptive statistics table
        """
        # Apply image-level aggregation if enabled
        if self.use_image_aggregation:
            analysis_df = self._aggregate_by_image(df, group_column)
            print(f"Descriptive statistics using image-level aggregation (n={len(analysis_df)} images)")
        else:
            analysis_df = df
            print(f"Descriptive statistics using per-nucleus data (n={len(analysis_df)} nuclei)")
        
        # Select numeric columns for analysis
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in 
                       ['nucleus_id', 'centroid_x', 'centroid_y', 'dk_number', 'well']]
        
        # Group by condition and calculate statistics
        desc_stats = []
        
        for group in analysis_df[group_column].unique():
            group_data = analysis_df[analysis_df[group_column] == group]
            n_observations = len(group_data)
            
            # For descriptive stats, report original nucleus counts and image counts
            if self.use_image_aggregation:
                original_group_data = df[df[group_column] == group]
                n_nuclei = len(original_group_data)
                n_images = original_group_data['image_name'].nunique() if 'image_name' in original_group_data.columns else 1
            else:
                n_nuclei = n_observations
                n_images = group_data['image_name'].nunique() if 'image_name' in group_data.columns else 1
            
            for feature in feature_cols:
                values = group_data[feature].dropna()
                if len(values) > 0:
                    desc_stats.append({
                        'group': group,
                        'feature': feature,
                        'n_nuclei': n_nuclei,
                        'n_images': n_images,
                        'n_statistical_units': n_observations,  # New field showing actual units used for stats
                        'mean': values.mean(),
                        'std': values.std(),
                        'median': values.median(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75),
                        'min': values.min(),
                        'max': values.max(),
                        'cv': values.std() / values.mean() if values.mean() != 0 else np.nan
                    })
        
        return pd.DataFrame(desc_stats)
    
    def test_normality(self, df: pd.DataFrame, 
                      features: List[str],
                      group_column: str = 'condition') -> pd.DataFrame:
        """
        Test normality of feature distributions by group.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature dataframe
        features : List[str]
            List of feature columns to test
        group_column : str
            Column for grouping
            
        Returns:
        --------
        pd.DataFrame
            Normality test results
        """
        normality_results = []
        
        for feature in features:
            for group in df[group_column].unique():
                group_data = df[df[group_column] == group][feature].dropna()
                
                if len(group_data) > 3:  # Minimum for Shapiro-Wilk
                    try:
                        statistic, p_value = stats.shapiro(group_data)
                        normality_results.append({
                            'group': group,
                            'feature': feature,
                            'n': len(group_data),
                            'shapiro_statistic': statistic,
                            'shapiro_p_value': p_value,
                            'is_normal': p_value > self.alpha
                        })
                    except Exception as e:
                        warnings.warn(f"Normality test failed for {feature} in {group}: {e}")
        
        return pd.DataFrame(normality_results)
    
    def compare_groups(self, df: pd.DataFrame,
                      features: List[str],
                      group_column: str = 'condition',
                      test_type: str = 'auto') -> pd.DataFrame:
        """
        Compare features between groups using appropriate statistical tests.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature dataframe (nucleus-level or image-level)
        features : List[str]
            Features to compare
        group_column : str
            Column for grouping
        test_type : str
            'auto', 'parametric', or 'nonparametric'
            
        Returns:
        --------
        pd.DataFrame
            Statistical comparison results
        """
        # Apply image-level aggregation if enabled
        if self.use_image_aggregation:
            analysis_df = self._aggregate_by_image(df, group_column)
            print(f"Statistical testing using image-level aggregation (n={len(analysis_df)} images)")
        else:
            analysis_df = df
            print(f"Statistical testing using per-nucleus data (n={len(analysis_df)} nuclei)")
        
        groups = analysis_df[group_column].unique()
        comparison_results = []
        
        # Test normality if auto mode
        if test_type == 'auto':
            normality_df = self.test_normality(analysis_df, features, group_column)
        else:
            normality_df = pd.DataFrame()  # Empty DataFrame for non-auto modes
        
        for feature in features:
            feature_data = analysis_df[feature].dropna()
            if len(feature_data) == 0:
                continue
                
            # Prepare data for comparison
            group_data = []
            group_names = []
            
            for group in groups:
                group_values = analysis_df[analysis_df[group_column] == group][feature].dropna()
                if len(group_values) > 0:
                    group_data.append(group_values)
                    group_names.append(group)
            
            if len(group_data) < 2:
                continue
                
            # Determine test type
            if test_type == 'auto':
                # Check if all groups are normally distributed
                if len(normality_df) > 0:
                    feature_normality = normality_df[normality_df['feature'] == feature]
                    all_normal = feature_normality['is_normal'].all() if len(feature_normality) > 0 else False
                else:
                    all_normal = False
                use_parametric = all_normal
            elif test_type == 'parametric':
                use_parametric = True
            else:
                use_parametric = False
            
            # Perform statistical test
            if len(group_data) == 2:
                # Two-group comparison
                if use_parametric:
                    # Equal variance test
                    _, levene_p = stats.levene(group_data[0], group_data[1])
                    equal_var = levene_p > self.alpha
                    
                    # t-test
                    statistic, p_value = stats.ttest_ind(group_data[0], group_data[1], 
                                                       equal_var=equal_var)
                    test_name = f"t-test ({'equal' if equal_var else 'unequal'} variance)"
                else:
                    # Mann-Whitney U test
                    statistic, p_value = stats.mannwhitneyu(group_data[0], group_data[1], 
                                                          alternative='two-sided')
                    test_name = "Mann-Whitney U"
                    
            else:
                # Multi-group comparison
                if use_parametric:
                    # One-way ANOVA
                    statistic, p_value = stats.f_oneway(*group_data)
                    test_name = "One-way ANOVA"
                else:
                    # Kruskal-Wallis test
                    
                    # Filter out groups with fewer than 2 samples, as they are not comparable
                    min_sample_size = 2
                    valid_groups = [g for g in group_data if len(g) >= min_sample_size]

                    if len(valid_groups) < 2:
                        warnings.warn(
                            f"Skipping Kruskal-Wallis for feature '{feature}': "
                            f"Fewer than 2 groups have sufficient samples (n>={min_sample_size})."
                        )
                        continue

                    try:
                        # Use only the valid groups for the test
                        statistic, p_value = stats.kruskal(*valid_groups)
                        test_name = "Kruskal-Wallis"
                    except ValueError as e:
                        if "All numbers are identical" in str(e):
                            warnings.warn(
                                f"Skipping Kruskal-Wallis for feature '{feature}': All aggregated values are identical. "
                                f"This can happen with very small sample sizes or features with no variance."
                            )
                            continue  # Skip to the next feature
                        else:
                            raise e  # Re-raise other ValueErrors
            
            # Calculate effect size (Cohen's d for two groups, eta-squared for multiple)
            effect_size = self._calculate_effect_size(group_data, use_parametric)
            
            comparison_results.append({
                'feature': feature,
                'test': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': effect_size,
                'n_groups': len(group_data),
                'total_n': sum(len(g) for g in group_data)
            })
        
        results_df = pd.DataFrame(comparison_results)

        # Multiple comparison correction (robust even for 0/1 rows)
        results_df = self._apply_multiple_comparison_correction(results_df)
            
        # Store results
        self.results['group_comparisons'] = results_df
            
        return results_df
    
    def pairwise_comparisons(self, df: pd.DataFrame,
                           features: List[str],
                           group_column: str = 'condition') -> pd.DataFrame:
        """
        Perform pairwise comparisons between all groups.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature dataframe (nucleus-level or image-level)
        features : List[str]
            Features to compare
        group_column : str
            Column for grouping
            
        Returns:
        --------
        pd.DataFrame
            Pairwise comparison results
        """
        # Apply image-level aggregation if enabled
        if self.use_image_aggregation:
            analysis_df = self._aggregate_by_image(df, group_column)
            print(f"Pairwise comparisons using image-level aggregation (n={len(analysis_df)} images)")
        else:
            analysis_df = df
            print(f"Pairwise comparisons using per-nucleus data (n={len(analysis_df)} nuclei)")
        
        groups = analysis_df[group_column].unique()
        pairwise_results = []
        
        for feature in features:
            for i, group1 in enumerate(groups):
                for j, group2 in enumerate(groups):
                    if i >= j:  # Avoid duplicate comparisons
                        continue
                        
                    data1 = analysis_df[analysis_df[group_column] == group1][feature].dropna()
                    data2 = analysis_df[analysis_df[group_column] == group2][feature].dropna()
                    
                    if len(data1) > 0 and len(data2) > 0:
                        # Use Mann-Whitney U (conservative choice)
                        statistic, p_value = stats.mannwhitneyu(data1, data2, 
                                                              alternative='two-sided')
                        
                        # Effect size (rank-biserial correlation for Mann-Whitney)
                        effect_size = 1 - (2 * statistic) / (len(data1) * len(data2))
                        
                        pairwise_results.append({
                            'feature': feature,
                            'group1': group1,
                            'group2': group2,
                            'statistic': statistic,
                            'p_value': p_value,
                            'effect_size': effect_size,
                            'n1': len(data1),
                            'n2': len(data2)
                        })
            
        pairwise_df = pd.DataFrame(pairwise_results)
            
        # Multiple comparison correction (robust even for 0/1 rows)
        pairwise_df = self._apply_multiple_comparison_correction(pairwise_df)
                
        return pairwise_df
    
    def multivariate_analysis(self, df: pd.DataFrame,
                            features: List[str],
                            group_column: str = 'condition') -> Dict:
        """
        Perform multivariate analysis (PCA, clustering).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature dataframe (nucleus-level or image-level)
        features : List[str]
            Features for multivariate analysis
        group_column : str
            Column for grouping
            
        Returns:
        --------
        Dict
            Multivariate analysis results
        """
        # Apply image-level aggregation if enabled
        if self.use_image_aggregation:
            analysis_df = self._aggregate_by_image(df, group_column)
            print(f"Multivariate analysis using image-level aggregation (n={len(analysis_df)} images)")
        else:
            analysis_df = df
            print(f"Multivariate analysis using per-nucleus data (n={len(analysis_df)} nuclei)")
        
        # Prepare data
        feature_data = analysis_df[features].fillna(0)  # Fill NaN with 0
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # PCA
        pca = PCA()
        pca_data = pca.fit_transform(scaled_data)
        
        # Create PCA results dataframe
        pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
        pca_df[group_column] = analysis_df[group_column].values
        
        # K-means clustering
        n_clusters = min(len(analysis_df[group_column].unique()), 4)  # Reasonable number
        from dna_condensation.pipeline.config import config
        kmeans = KMeans(n_clusters=n_clusters, random_state=config.get_seed(42))
        clusters = kmeans.fit_predict(scaled_data)

        results = {
            'pca': {
                'data': pca_df,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'feature_loadings': pd.DataFrame(
                    pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(len(pca.components_))],
                    index=features
                )
            },
            'clustering': {
                'labels': clusters,
                'centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_
            },
            'scaler': scaler
        }
        
        return results
    
    def _calculate_effect_size(self, group_data: List[np.ndarray], 
                              parametric: bool) -> float:
        """Calculate appropriate effect size measure."""
        if len(group_data) == 2:
            # Cohen's d for two groups
            group1, group2 = group_data
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                 (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                (len(group1) + len(group2) - 2))
            if pooled_std > 0:
                return abs(np.mean(group1) - np.mean(group2)) / pooled_std
            else:
                return 0
        else:
            # Eta-squared for multiple groups
            all_data = np.concatenate(group_data)
            grand_mean = np.mean(all_data)
            
            # Between-group sum of squares
            ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 
                           for group in group_data)
            
            # Total sum of squares
            ss_total = np.sum((all_data - grand_mean) ** 2)
            
            if ss_total > 0:
                return ss_between / ss_total
            else:
                return 0
    
    def _apply_multiple_comparison_correction(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Apply Benjamini-Hochberg correction (FDR) with robust handling of NaNs and small N.

        - Detect p-value column robustly (prefers 'p_value', falls back to common aliases)
        - Coerces p-values to numeric
        - If no finite p-values, returns DataFrame with p_corrected NaN and significant False
        - If exactly one finite p-value, BH reduces to identity: p_corrected = p_value, significant = p < alpha
        - Else, uses statsmodels.multipletests(method='fdr_bh') on finite subset and aligns results
        """
        import numpy as np
        import pandas as pd
        from statsmodels.stats.multitest import multipletests

        if results_df is None or len(results_df) == 0:
            # Ensure columns exist for downstream consumers
            results_df = results_df.copy()
            results_df['p_corrected'] = np.nan
            results_df['significant'] = False
            return results_df

        df = results_df.copy()

        # Detect p-value column name
        p_col_candidates = ['p_value', 'p', 'pval', 'pvalue']
        p_col = None
        for c in p_col_candidates:
            if c in df.columns:
                p_col = c
                break
        if p_col is None:
            # Create empty column to keep downstream stable
            df['p_value'] = np.nan
            p_col = 'p_value'

        # Coerce p-values to numeric and build finite mask
        df[p_col] = pd.to_numeric(df[p_col], errors='coerce')
        finite_mask = np.isfinite(df[p_col].to_numpy())

        # Initialize outputs
        df['p_corrected'] = np.nan
        df['significant'] = False

        n_finite = int(finite_mask.sum())
        if n_finite == 0:
            return df
        elif n_finite == 1:
            # Identity for a single test
            idx = np.flatnonzero(finite_mask)[0]
            p = float(df.iloc[idx][p_col])
            df.at[df.index[idx], 'p_corrected'] = p
            df.at[df.index[idx], 'significant'] = bool(p < self.alpha)
            return df
        else:
            # Apply BH on finite subset and align
            pvals = df.loc[finite_mask, p_col].to_numpy(dtype=float)
            rejected, p_corrected, _, _ = multipletests(pvals, alpha=self.alpha, method='fdr_bh')
            df.loc[finite_mask, 'p_corrected'] = p_corrected
            df.loc[finite_mask, 'significant'] = rejected
            return df
    
    def generate_summary_report(self, df: pd.DataFrame,
                               features: List[str],
                               group_column: str = 'condition') -> str:
        """Generate a comprehensive statistical summary report."""
        
        report = []
        report.append("=" * 60)
        report.append("DNA CONDENSATION STATISTICAL ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Analysis method information
        if self.use_image_aggregation:
            report.append("STATISTICAL APPROACH:")
            report.append("  Image-level aggregation enabled (recommended to avoid pseudoreplication)")
            report.append("  Statistical tests performed on image-level medians")
            report.append("")
        else:
            report.append("STATISTICAL APPROACH:")
            report.append("  Per-nucleus analysis (WARNING: may have pseudoreplication issues)")
            report.append("  Statistical tests performed on individual nuclei")
            report.append("")
        
        # Data overview
        report.append("DATA OVERVIEW:")
        report.append(f"  Total nuclei: {len(df)}")
        report.append(f"  Total images: {df['image_name'].nunique()}")
        report.append(f"  Groups: {', '.join(df[group_column].unique())}")
        report.append("")
        
        # Group sizes with image counts
        report.append("GROUP SIZES:")
        for group in df[group_column].unique():
            group_data = df[df[group_column] == group]
            n_nuclei = len(group_data)
            n_images = group_data['image_name'].nunique() if 'image_name' in group_data.columns else 1
            if self.use_image_aggregation:
                report.append(f"  {group}: {n_nuclei} nuclei from {n_images} images (statistical n={n_images})")
            else:
                report.append(f"  {group}: {n_nuclei} nuclei from {n_images} images (statistical n={n_nuclei})")
        report.append("")
        
        # Key findings from comparisons
        if 'group_comparisons' in self.results:
            comp_df = self.results['group_comparisons']
            significant = comp_df[comp_df['significant'] == True] if 'significant' in comp_df.columns else comp_df[comp_df['p_value'] < self.alpha]
            
            report.append("SIGNIFICANT DIFFERENCES:")
            if len(significant) > 0:
                for _, row in significant.iterrows():
                    report.append(f"  {row['feature']}: {row['test']}, p = {row['p_value']:.4f}, effect size = {row['effect_size']:.3f}")
            else:
                report.append("  No significant differences found")
            report.append("")
        
        return "\n".join(report)


def identify_key_features(comparison_results: pd.DataFrame, 
                         top_n: int = 10) -> List[str]:
    """
    Identify the most important features for distinguishing groups.
    
    Parameters:
    -----------
    comparison_results : pd.DataFrame
        Results from group comparisons
    top_n : int
        Number of top features to return
        
    Returns:
    --------
    List[str]
        List of most discriminative features
    """
    # Sort by effect size (descending) and p-value (ascending)
    if 'p_corrected' in comparison_results.columns:
        p_col = 'p_corrected'
    else:
        p_col = 'p_value'
        
    # Create a ranking score combining effect size and significance
    comparison_results = comparison_results.copy()
    comparison_results['rank_score'] = (
        comparison_results['effect_size'] * 
        (1 - comparison_results[p_col])  # Higher score for lower p-values
    )
    
    top_features = comparison_results.nlargest(top_n, 'rank_score')['feature'].tolist()
    
    return top_features
