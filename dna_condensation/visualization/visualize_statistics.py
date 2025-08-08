"""
Statistical visualization for DNA condensation analysis.

This module provides comprehensive visualization tools for statistical results
and feature distributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import warnings

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")

class StatisticalVisualizer:
    """
    Create comprehensive visualizations for DNA condensation statistical analysis.
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        figure_size : Tuple[int, int]
            Default figure size (width, height)
        dpi : int
            Resolution for saved figures
        """
        self.figure_size = figure_size
        self.dpi = dpi
        self.color_palette = sns.color_palette("husl", 8)
        
    def plot_feature_distributions(self, df: pd.DataFrame,
                                  features: List[str],
                                  group_column: str = 'condition',
                                  save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot feature distributions by group using violin plots with overlaid points.
        
        Creates violin plots to show distribution shape, with strip plots to show
        individual data points. This visualization reveals both population-level
        differences and individual nucleus variation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature data with group labels
        features : List[str]
            List of feature columns to plot
        group_column : str
            Column containing group labels
        save_path : Optional[Path]
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            Generated figure object
        """
        n_features = len(features)
        n_cols = min(3, n_features)  # Maximum 3 columns for readability
        n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        
        # Handle different subplot configurations
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Create violin plot to show distribution shape
            sns.violinplot(data=df, x=group_column, y=feature, ax=ax)
            
            # Overlay individual data points to show raw data
            sns.stripplot(data=df, x=group_column, y=feature, ax=ax, 
                         size=2, alpha=0.6, color='black')
            
            # Format plot appearance
            ax.set_title(f'{feature.replace("_", " ").title()}')
            ax.tick_params(axis='x', rotation=45)
            
        # Clean up empty subplots in the grid
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.remove()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_comparison_summary(self, comparison_results: pd.DataFrame,
                               save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create a summary volcano plot and bar chart of statistical comparisons.
        
        Shows both effect sizes vs significance (volcano plot) and 
        the number of significant features to give an overview of 
        analysis results.
        
        Parameters:
        -----------
        comparison_results : pd.DataFrame
            Results from statistical comparisons with p-values and effect sizes
        save_path : Optional[Path]
            Path to save figure
            
        Returns:
        --------
        plt.Figure
            Generated figure with two subplots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Use corrected p-values if available, otherwise raw p-values
        p_col = 'p_corrected' if 'p_corrected' in comparison_results.columns else 'p_value'
        
        # Color-code points by statistical significance
        colors = ['red' if p < 0.05 else 'blue' for p in comparison_results[p_col]]
        
        # Plot 1: Volcano plot (effect size vs -log10(p-value))
        ax1.scatter(comparison_results['effect_size'], -np.log10(comparison_results[p_col]), 
                   c=colors, alpha=0.7)
        
        # Add significance line
        ax1.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, 
                   label='p = 0.05')
        
        ax1.set_xlabel('Effect Size')
        ax1.set_ylabel('-log10(p-value)')
        ax1.set_title('Statistical Significance vs Effect Size')
        ax1.legend()
        
        # Add feature labels for significant results
        significant = comparison_results[comparison_results[p_col] < 0.05]
        for _, row in significant.iterrows():
            ax1.annotate(row['feature'], 
                        (row['effect_size'], -np.log10(row[p_col])),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        # Plot 2: Feature ranking by effect size
        top_features = comparison_results.nlargest(15, 'effect_size')
        
        bars = ax2.barh(range(len(top_features)), top_features['effect_size'])
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels([f.replace('_', ' ').title() for f in top_features['feature']])
        ax2.set_xlabel('Effect Size')
        ax2.set_title('Top Features by Effect Size')
        
        # Color bars by significance
        for i, (_, row) in enumerate(top_features.iterrows()):
            color = 'red' if row[p_col] < 0.05 else 'blue'
            bars[i].set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_pca_analysis(self, pca_results: Dict,
                         group_column: str = 'condition',
                         save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot PCA results with explained variance and feature loadings.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        pca_data = pca_results['data']
        explained_var = pca_results['explained_variance_ratio']
        loadings = pca_results['feature_loadings']
        
        # Plot 1: PCA scatter plot (PC1 vs PC2)
        groups = pca_data[group_column].unique()
        for i, group in enumerate(groups):
            group_data = pca_data[pca_data[group_column] == group]
            ax1.scatter(group_data['PC1'], group_data['PC2'], 
                       label=group, alpha=0.7, s=30)
        
        ax1.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
        ax1.set_title('PCA: Groups in Principal Component Space')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Explained variance
        cumvar = np.cumsum(explained_var)
        ax2.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, 
                label='Individual')
        ax2.plot(range(1, len(cumvar) + 1), cumvar, 'ro-', 
                label='Cumulative')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance Ratio')
        ax2.set_title('PCA Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature loadings for PC1
        top_loadings_pc1 = loadings['PC1'].abs().nlargest(10)
        ax3.barh(range(len(top_loadings_pc1)), 
                [loadings.loc[feat, 'PC1'] for feat in top_loadings_pc1.index])
        ax3.set_yticks(range(len(top_loadings_pc1)))
        ax3.set_yticklabels([f.replace('_', ' ').title() for f in top_loadings_pc1.index])
        ax3.set_xlabel('Loading Value')
        ax3.set_title('Top Feature Loadings - PC1')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feature loadings for PC2
        top_loadings_pc2 = loadings['PC2'].abs().nlargest(10)
        ax4.barh(range(len(top_loadings_pc2)), 
                [loadings.loc[feat, 'PC2'] for feat in top_loadings_pc2.index])
        ax4.set_yticks(range(len(top_loadings_pc2)))
        ax4.set_yticklabels([f.replace('_', ' ').title() for f in top_loadings_pc2.index])
        ax4.set_xlabel('Loading Value')
        ax4.set_title('Top Feature Loadings - PC2')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                features: List[str],
                                save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot correlation heatmap of features.
        """
        # Calculate correlation matrix
        corr_matrix = df[features].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8})
        
        ax.set_title('Feature Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_group_means_heatmap(self, df: pd.DataFrame,
                                features: List[str],
                                group_column: str = 'condition',
                                save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot heatmap of feature means by group.
        """
        # Calculate group means
        group_means = df.groupby(group_column)[features].mean()
        
        # Standardize for visualization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_means = pd.DataFrame(
            scaler.fit_transform(group_means.T).T,
            index=group_means.index,
            columns=group_means.columns
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create heatmap
        sns.heatmap(scaled_means, annot=True, cmap='RdBu_r', center=0,
                   ax=ax, fmt='.2f', cbar_kws={'label': 'Standardized Mean'})
        
        ax.set_title('Feature Means by Group (Standardized)')
        ax.set_xlabel('Features')
        ax.set_ylabel('Groups')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_radial_profiles(self, df: pd.DataFrame,
                           group_column: str = 'condition',
                           n_shells: int = 5,
                           save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot radial intensity profiles by group.
        """
        radial_cols = [f'radial_shell_{i}' for i in range(n_shells)]
        
        # Check if radial columns exist
        available_radial = [col for col in radial_cols if col in df.columns]
        if not available_radial:
            warnings.warn("No radial profile columns found")
            return plt.figure()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        groups = df[group_column].unique()
        shell_positions = np.arange(len(available_radial))
        
        for group in groups:
            group_data = df[df[group_column] == group]
            means = [group_data[col].mean() for col in available_radial]
            stds = [group_data[col].std() for col in available_radial]
            
            ax.errorbar(shell_positions, means, yerr=stds, 
                       label=group, marker='o', capsize=5)
        
        ax.set_xlabel('Radial Shell (center → edge)')
        ax.set_ylabel('Mean Intensity')
        ax.set_title('Radial Intensity Profiles by Group')
        ax.set_xticks(shell_positions)
        ax.set_xticklabels([f'Shell {i}' for i in range(len(available_radial))])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def create_comprehensive_report(self, df: pd.DataFrame,
                                   comparison_results: pd.DataFrame,
                                   pca_results: Dict,
                                   features: List[str],
                                   group_column: str = 'condition',
                                   output_dir: Path = None) -> Dict[str, plt.Figure]:
        """
        Create a comprehensive visualization report.
        """
        if output_dir is None:
            output_dir = Path('.')
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        figures = {}
        
        # 1. Feature distributions
        print("Creating feature distribution plots...")
        key_features = comparison_results.nlargest(12, 'effect_size')['feature'].tolist()
        fig1 = self.plot_feature_distributions(df, key_features, group_column,
                                              output_dir / 'feature_distributions.png')
        figures['distributions'] = fig1
        
        # 2. Statistical comparison summary
        print("Creating comparison summary...")
        fig2 = self.plot_comparison_summary(comparison_results,
                                           output_dir / 'comparison_summary.png')
        figures['comparisons'] = fig2
        
        # 3. PCA analysis
        print("Creating PCA plots...")
        fig3 = self.plot_pca_analysis(pca_results, group_column,
                                     output_dir / 'pca_analysis.png')
        figures['pca'] = fig3
        
        # 4. Correlation heatmap
        print("Creating correlation heatmap...")
        fig4 = self.plot_correlation_heatmap(df, key_features,
                                            output_dir / 'correlation_heatmap.png')
        figures['correlations'] = fig4
        
        # 5. Group means heatmap
        print("Creating group means heatmap...")
        fig5 = self.plot_group_means_heatmap(df, key_features, group_column,
                                            output_dir / 'group_means_heatmap.png')
        figures['group_means'] = fig5
        
        # 6. Radial profiles
        print("Creating radial profiles...")
        fig6 = self.plot_radial_profiles(df, group_column,
                                        save_path=output_dir / 'radial_profiles.png')
        figures['radial_profiles'] = fig6
        
        print(f"All visualizations saved to {output_dir}")
        
        return figures

    def plot_single_metric_with_significance(self, df: pd.DataFrame,
                                           metric: str,
                                           group_column: str = 'condition',
                                           save_path: Optional[Path] = None,
                                           title: Optional[str] = None,
                                           ylabel: Optional[str] = None) -> plt.Figure:
        """
        Create a standalone figure for a specific metric across groups with significance testing.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data containing the metric and group information
        metric : str
            Name of the metric/feature to plot
        group_column : str
            Column name containing group labels
        save_path : Optional[Path]
            Path to save the figure
        title : Optional[str]
            Custom title for the plot
        ylabel : Optional[str]
            Custom y-axis label
            
        Returns:
        --------
        plt.Figure
            The created figure
        """
        from scipy.stats import kruskal, mannwhitneyu
        from itertools import combinations
        import numpy as np

        # Check if metric exists in dataframe
        if metric not in df.columns:
            available_metrics = [col for col in df.columns if col not in ['image_name', 'nucleus_id', 'centroid_x', 'centroid_y', group_column]]
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {available_metrics[:10]}...")

        # Remove missing values
        plot_data = df[[metric, group_column]].dropna()
        if len(plot_data) == 0:
            raise ValueError(f"No valid data for metric '{metric}'")

        groups = sorted(plot_data[group_column].unique())
        n_groups = len(groups)
        if n_groups < 2:
            raise ValueError(f"Need at least 2 groups for comparison, found {n_groups}")

        # Create figure
        fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.5), 8))

        # Create violin plots with individual points
        parts = ax.violinplot([plot_data[plot_data[group_column] == group][metric].values 
                              for group in groups], 
                             positions=range(len(groups)), showmeans=True, showmedians=True)

        # Color the violin plots
        colors = sns.color_palette("husl", n_groups)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        # Add individual data points (jittered)
        for i, group in enumerate(groups):
            group_data = plot_data[plot_data[group_column] == group][metric].values
            y_jitter = np.random.normal(i, 0.05, len(group_data))
            ax.scatter(y_jitter, group_data, alpha=0.6, s=20, color=colors[i], edgecolor='black', linewidth=0.5)

        # Prepare data for stats
        group_data_lists = [plot_data[plot_data[group_column] == group][metric].values for group in groups]
        # Detect degenerate case: all values identical across all data
        all_unique = plot_data[metric].nunique()
        all_constant = all_unique <= 1 or all(np.nanstd(g) == 0 for g in group_data_lists if len(g) > 0)

        # Overall significance test (Kruskal-Wallis for non-parametric), with safeguards
        pairwise_results = []
        if not all_constant:
            try:
                kw_stat, kw_p = kruskal(*group_data_lists)
            except Exception:
                kw_stat, kw_p = np.nan, 1.0
        else:
            kw_stat, kw_p = np.nan, 1.0

        # Pairwise comparisons with Mann-Whitney U tests (only if overall test significant)
        alpha = 0.05
        if (kw_p is not None) and (not np.isnan(kw_p)) and kw_p < alpha:
            for i, j in combinations(range(n_groups), 2):
                group1_data = group_data_lists[i]
                group2_data = group_data_lists[j]
                if len(group1_data) > 0 and len(group2_data) > 0:
                    # Skip degenerate pairs
                    if (np.nanstd(group1_data) == 0 and np.nanstd(group2_data) == 0 and
                        np.allclose(np.nanmean(group1_data), np.nanmean(group2_data))):
                        continue
                    try:
                        u_stat, p_val = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                        # Bonferroni correction
                        n_comparisons = len(list(combinations(range(n_groups), 2)))
                        p_corrected = min(p_val * n_comparisons, 1.0)
                        pairwise_results.append({
                            'group1': groups[i],
                            'group2': groups[j],
                            'p_value': p_val,
                            'p_corrected': p_corrected,
                            'significant': p_corrected < alpha,
                            'positions': (i, j)
                        })
                    except Exception:
                        continue

        # Add significance bars
        y_max = plot_data[metric].max()
        y_min = plot_data[metric].min()
        y_range = max(y_max - y_min, 1e-6)
        bar_height = y_max + 0.05 * y_range
        sig_height_increment = 0.08 * y_range
        current_height = bar_height
        for result in pairwise_results:
            if result['significant']:
                pos1, pos2 = result['positions']
                ax.plot([pos1, pos1, pos2, pos2], 
                        [current_height - sig_height_increment/3, current_height, current_height, current_height - sig_height_increment/3], 
                        'k-', linewidth=1.5)
                if result['p_corrected'] < 0.001:
                    sig_text = '***'
                elif result['p_corrected'] < 0.01:
                    sig_text = '**'
                elif result['p_corrected'] < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'
                ax.text((pos1 + pos2) / 2, current_height + sig_height_increment/4, sig_text, 
                        ha='center', va='bottom', fontweight='bold', fontsize=12)
                current_height += sig_height_increment

        # Customize plot and annotations
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_xlabel('Experimental Groups', fontsize=12, fontweight='bold')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        else:
            clean_metric = metric.replace('_', ' ').title()
            ax.set_ylabel(clean_metric, fontsize=12, fontweight='bold')
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            clean_metric = metric.replace('_', ' ').title()
            ax.set_title(f'{clean_metric} Across Experimental Groups', fontsize=14, fontweight='bold', pad=20)

        # Add statistical test info
        if (kw_p is None) or np.isnan(kw_p):
            kw_text = 'Kruskal-Wallis: n/a'
        elif kw_p < 0.001:
            kw_text = 'Kruskal-Wallis: p < 0.001'
        else:
            kw_text = f'Kruskal-Wallis: p = {kw_p:.3f}'
        ax.text(0.02, 0.98, kw_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Add sample sizes
        sample_text = []
        for group in groups:
            n = len(plot_data[plot_data[group_column] == group])
            sample_text.append(f'{group}: n={n}')
        ax.text(0.02, 0.02, '\n'.join(sample_text), transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Adjust y-axis limits
        if pairwise_results and any(r['significant'] for r in pairwise_results):
            ax.set_ylim(bottom=y_min - 0.05 * y_range, top=current_height + 0.1 * y_range)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved single metric plot to {save_path}")
            plt.close(fig)  # Prevent too many open figures in batch mode
        return fig