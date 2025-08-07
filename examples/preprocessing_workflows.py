#!/usr/bin/env python3
"""
PREPROCESSING WORKFLOW EXAMPLES

This file demonstrates the recommended order for applying preprocessing corrections
for different analysis purposes. Follow these examples for optimal results.

Author: DNA Condensation Analysis Pipeline
Date: August 2025
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from dna_condensation.core.preprocessor import (
    deconvolution, background_correction, intensity_normalization, 
    per_nucleus_intensity_normalization, bulk_preprocess_images
)

def example_standard_workflow(images, channel_index=0):
    """
    Standard preprocessing workflow for comparing images across experiments.
    Use this when you need to standardize intensity ranges between different samples.
    """
    print("=== STANDARD PREPROCESSING WORKFLOW ===")
    print("Purpose: Compare measurements across different experiments/samples")
    print("Order: [deconvolution] → background_correction → intensity_normalization")
    
    # Option 1: Use bulk processing (recommended)
    preprocessed = bulk_preprocess_images(
        images=images,
        channel_index=channel_index,
        methods=['background_correction', 'intensity_normalization'],
        # Optional: Add deconvolution if needed
        # methods=['deconvolution', 'background_correction', 'intensity_normalization'],
        bg_ball_radius=50,  # Adjust based on nucleus size
        norm_method='percentile',
        norm_percentile_range=(1, 99)
    )
    
    print(f"✓ Processed {len(preprocessed)} images")
    return preprocessed

def example_homogeneity_workflow(images, labels_list, channel_index=0):
    """
    Specialized workflow for homogeneity analysis within individual nuclei.
    Use this when measuring texture, CV, or intensity distribution patterns.
    """
    print("=== HOMOGENEITY ANALYSIS WORKFLOW ===")
    print("Purpose: Analyze intensity patterns within individual nuclei")
    print("Order: [deconvolution] → background_correction → segmentation → per_nucleus_normalization")
    
    # Step 1: Background correction only (no global intensity normalization!)
    bg_corrected = bulk_preprocess_images(
        images=images,
        channel_index=channel_index,
        methods=['background_correction'],  # Critical: Skip intensity_normalization
        bg_ball_radius=50
    )
    
    # Step 2: Per-nucleus normalization (after segmentation)
    normalized_images = []
    all_stats = []
    
    for i, (image, labels) in enumerate(zip(bg_corrected, labels_list)):
        if image is not None and labels is not None:
            norm_image, stats = per_nucleus_intensity_normalization(
                image=image if image.ndim == 2 else image[:, :, channel_index],
                labels=labels,
                target_mean=1.0,  # Makes CV calculations intuitive
                verbose=(i == 0)  # Show details for first image only
            )
            normalized_images.append(norm_image)
            all_stats.append(stats)
        else:
            normalized_images.append(None)
            all_stats.append({})
    
    print(f"✓ Applied per-nucleus normalization to {len(normalized_images)} images")
    return normalized_images, all_stats

def example_quality_control_workflow(images, channel_index=0):
    """
    Comprehensive workflow with all preprocessing steps for high-quality analysis.
    Use this for publication-quality results or when image quality is variable.
    """
    print("=== COMPREHENSIVE QUALITY CONTROL WORKFLOW ===")
    print("Purpose: Maximum quality preprocessing for publication/analysis")
    print("Order: deconvolution → background_correction → intensity_normalization")
    
    # Apply all preprocessing steps
    high_quality = bulk_preprocess_images(
        images=images,
        channel_index=channel_index,
        methods=['deconvolution', 'background_correction', 'intensity_normalization'],
        # Deconvolution parameters
        deconv_sigma=1.0,
        deconv_iterations=10,
        # Background correction parameters
        bg_ball_radius=50,
        # Normalization parameters
        norm_method='percentile',
        norm_percentile_range=(0.5, 99.5)  # More conservative clipping
    )
    
    print(f"✓ Applied comprehensive preprocessing to {len(high_quality)} images")
    return high_quality

def demonstrate_parameter_selection():
    """
    Guidelines for selecting optimal parameters for different scenarios.
    """
    print("=== PARAMETER SELECTION GUIDELINES ===")
    
    guidelines = {
        "bg_ball_radius": {
            "purpose": "Rolling ball background correction",
            "rule": "Set to ~1.5x average nucleus diameter",
            "typical_values": {
                "Small nuclei (lymphocytes)": "20-40 pixels",
                "Medium nuclei (HeLa, fibroblasts)": "40-70 pixels", 
                "Large nuclei (neurons, muscle)": "70-120 pixels"
            },
            "warning": "Too small: Removes actual signal. Too large: Doesn't remove background"
        },
        "deconv_iterations": {
            "purpose": "Richardson-Lucy deconvolution",
            "rule": "Start low, increase if undersharpened",
            "typical_values": {
                "Light deconvolution": "3-7 iterations",
                "Standard deconvolution": "8-15 iterations",
                "Heavy deconvolution": "15-30 iterations"
            },
            "warning": "Too many iterations can introduce artifacts"
        },
        "norm_method": {
            "purpose": "Intensity normalization strategy",
            "recommendations": {
                "percentile": "Most robust, good for variable backgrounds",
                "target_mean": "Preserves relative intensities well",
                "zscore": "Good for normally distributed intensities",
                "minmax": "Use only for very clean data"
            }
        },
        "target_mean": {
            "purpose": "Per-nucleus normalization target",
            "recommendations": {
                "1.0": "Recommended - makes CV = standard deviation",
                "128.0": "Alternative for uint8 compatibility",
                "Custom": "Match your analysis requirements"
            }
        }
    }
    
    for param, info in guidelines.items():
        print(f"\n{param.upper()}:")
        print(f"  Purpose: {info['purpose']}")
        if 'rule' in info:
            print(f"  Rule: {info['rule']}")
        if 'typical_values' in info:
            print("  Typical values:")
            for scenario, value in info['typical_values'].items():
                print(f"    {scenario}: {value}")
        if 'recommendations' in info:
            print("  Recommendations:")
            for method, desc in info['recommendations'].items():
                print(f"    {method}: {desc}")
        if 'warning' in info:
            print(f"  ⚠️  Warning: {info['warning']}")

def common_mistakes_to_avoid():
    """
    Common preprocessing mistakes and how to avoid them.
    """
    print("=== COMMON MISTAKES TO AVOID ===")
    
    mistakes = [
        {
            "mistake": "Applying intensity normalization before background correction",
            "why_bad": "Background gradients become 'normalized' and harder to remove",
            "correct": "Always do background_correction() before intensity_normalization()"
        },
        {
            "mistake": "Using both global and per-nucleus intensity normalization", 
            "why_bad": "Double normalization can distort intensity relationships",
            "correct": "Choose either global OR per-nucleus normalization, not both"
        },
        {
            "mistake": "Deconvolution after background correction",
            "why_bad": "Deconvolution works best on original intensity distributions",
            "correct": "Apply deconvolution() first, before other corrections"
        },
        {
            "mistake": "Wrong ball_radius for background correction",
            "why_bad": "Too small removes signal, too large doesn't remove background",
            "correct": "Set to ~1.5x nucleus diameter, test on sample images"
        },
        {
            "mistake": "Skipping background correction for homogeneity analysis",
            "why_bad": "Illumination gradients appear as false heterogeneity",
            "correct": "Background correction is CRITICAL for within-nucleus analysis"
        }
    ]
    
    for i, mistake in enumerate(mistakes, 1):
        print(f"\n{i}. {mistake['mistake']}")
        print(f"   Why it's bad: {mistake['why_bad']}")
        print(f"   ✓ Correct approach: {mistake['correct']}")

if __name__ == "__main__":
    print("DNA CONDENSATION PREPROCESSING WORKFLOW EXAMPLES")
    print("=" * 60)
    
    # Show parameter guidelines
    demonstrate_parameter_selection()
    print("\n" + "=" * 60)
    
    # Show common mistakes
    common_mistakes_to_avoid()
    
    print("\n" + "=" * 60)
    print("For implementation examples, call the example functions with your data:")
    print("- example_standard_workflow(images, channel_index)")
    print("- example_homogeneity_workflow(images, labels_list, channel_index)")
    print("- example_quality_control_workflow(images, channel_index)")
