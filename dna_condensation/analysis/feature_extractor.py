"""
Feature extraction for DNA condensation analysis.

This module implements comprehensive feature extraction for segmented nuclei,
focusing on morphological and intensity distribution features that report on
DNA condensation levels.
"""

import numpy as np
import pandas as pd
from skimage import measure, feature
from skimage.filters import rank
from skimage.morphology import disk, opening
from scipy import ndimage
from scipy.stats import entropy
import warnings
from typing import List, Dict, Tuple, Optional, Union

class DNACondensationFeatureExtractor:
    """
    Extract features from segmented nuclei to quantify DNA condensation.
    
    Features extracted:
    1. Intensity Distribution: CV, high-intensity fraction, skewness, kurtosis
    2. Spatial Metrics: Radial intensity profiles, center-to-edge ratios
    3. Morphological Features: Area, perimeter, solidity, eccentricity
    4. Texture Analysis: GLCM-based features, local binary patterns
    5. Granulometry: Multi-scale morphological analysis
    """
    
    def __init__(self, 
                 high_intensity_percentile: float = 90,
                 radial_shells: int = 5,
                 texture_distance: int = 1,
                 granulometry_radii: List[int] = None):
        """
        Initialize feature extractor with configurable parameters.
        
        Parameters:
        -----------
        high_intensity_percentile : float
            Percentile threshold for high-intensity fraction calculation
        radial_shells : int
            Number of concentric shells for radial profiling
        texture_distance : int
            Distance for GLCM texture analysis
        granulometry_radii : List[int]
            Radii for morphological granulometry analysis
        """
        self.high_intensity_percentile = high_intensity_percentile
        self.radial_shells = radial_shells
        self.texture_distance = texture_distance
        self.granulometry_radii = granulometry_radii or [1, 2, 3, 5, 7, 10]
        
    def extract_features_batch(self, 
                              images: List[np.ndarray], 
                              masks: List[np.ndarray],
                              image_names: List[str] = None) -> pd.DataFrame:
        """
        Extract features from multiple images and masks.
        
        Parameters:
        -----------
        images : List[np.ndarray]
            List of intensity images (2D)
        masks : List[np.ndarray] 
            List of labeled masks (2D, integer labels)
        image_names : List[str], optional
            Names/identifiers for each image
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with one row per nucleus, columns for all features
        """
        if image_names is None:
            image_names = [f"image_{i:03d}" for i in range(len(images))]
            
        all_features = []
        
        for i, (image, mask, name) in enumerate(zip(images, masks, image_names)):
            if image is None or mask is None:
                continue
                
            print(f"Processing {name}: {np.max(mask)} nuclei")
            features = self.extract_features_single(image, mask, name)
            all_features.append(features)
            
        if all_features:
            return pd.concat(all_features, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def extract_features_single(self, 
                               image: np.ndarray, 
                               mask: np.ndarray,
                               image_name: str = "unknown") -> pd.DataFrame:
        """
        Extract features from a single image-mask pair.
        
        Parameters:
        -----------
        image : np.ndarray
            2D intensity image
        mask : np.ndarray
            2D labeled mask (integer labels for each nucleus)
        image_name : str
            Identifier for this image
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with one row per nucleus
        """
        # Get region properties
        regions = measure.regionprops(mask, intensity_image=image)
        
        features_list = []
        
        for region in regions:
            try:
                features = self._extract_nucleus_features(region, image, mask, image_name)
                features_list.append(features)
            except Exception as e:
                warnings.warn(f"Failed to extract features for nucleus {region.label} in {image_name}: {e}")
                continue
                
        return pd.DataFrame(features_list)
    
    def _extract_nucleus_features(self, 
                                 region,
                                 image: np.ndarray,
                                 mask: np.ndarray, 
                                 image_name: str) -> Dict:
        """Extract all features for a single nucleus."""
        
        # Basic identifiers
        features = {
            'image_name': image_name,
            'nucleus_id': region.label,
            'centroid_y': region.centroid[0],
            'centroid_x': region.centroid[1],
        }
        
        # Get intensity values for this nucleus
        intensity_values = region.intensity_image[region.image].flatten()
        
        # 1. Intensity Distribution Features
        features.update(self._intensity_features(intensity_values))
        
        # 2. Morphological Features  
        features.update(self._morphological_features(region))
        
        # 3. Spatial/Radial Features
        features.update(self._radial_features(region, image))
        
        # 4. Texture Features
        features.update(self._texture_features(region))
        
        # 5. Granulometry Features
        features.update(self._granulometry_features(region))
        
        return features
    
    def _intensity_features(self, intensity_values: np.ndarray) -> Dict:
        """Extract intensity distribution features."""
        
        # Basic statistics
        mean_intensity = np.mean(intensity_values)
        std_intensity = np.std(intensity_values)
        
        features = {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'coefficient_of_variation': std_intensity / mean_intensity if mean_intensity > 0 else 0,
            'min_intensity': np.min(intensity_values),
            'max_intensity': np.max(intensity_values),
            'intensity_range': np.max(intensity_values) - np.min(intensity_values),
        }
        
        # Percentile-based features
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            features[f'intensity_p{p}'] = np.percentile(intensity_values, p)
        
        # High intensity fraction
        high_threshold = np.percentile(intensity_values, self.high_intensity_percentile)
        features['high_intensity_fraction'] = np.mean(intensity_values >= high_threshold)
        
        # Distribution shape
        from scipy.stats import skew, kurtosis
        features['intensity_skewness'] = skew(intensity_values)
        features['intensity_kurtosis'] = kurtosis(intensity_values)
        
        # Entropy (measure of intensity variability)
        hist, _ = np.histogram(intensity_values, bins=256)
        hist = hist / np.sum(hist)  # normalize
        features['intensity_entropy'] = entropy(hist + 1e-10)  # avoid log(0)
        
        return features
    
    def _morphological_features(self, region) -> Dict:
        """Extract morphological features."""
        
        return {
            'area': region.area,
            'perimeter': region.perimeter,
            'major_axis_length': region.major_axis_length,
            'minor_axis_length': region.minor_axis_length,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity,
            'extent': region.extent,
            'orientation': region.orientation,
            'equivalent_diameter': region.equivalent_diameter,
            'aspect_ratio': region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0,
            'circularity': 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0,
        }
    
    def _radial_features(self, region, image: np.ndarray) -> Dict:
        """Extract radial intensity profile features."""
        
        # Get coordinates relative to centroid
        coords = region.coords
        centroid = region.centroid
        
        # Calculate distances from centroid
        distances = np.sqrt(np.sum((coords - centroid) ** 2, axis=1))
        max_distance = np.max(distances)
        
        if max_distance == 0:
            # Single pixel nucleus
            return {f'radial_shell_{i}': region.mean_intensity for i in range(self.radial_shells)}
        
        # Create shells
        shell_means = {}
        for i in range(self.radial_shells):
            r_inner = i * max_distance / self.radial_shells
            r_outer = (i + 1) * max_distance / self.radial_shells
            
            # Find pixels in this shell
            mask = (distances >= r_inner) & (distances < r_outer)
            if i == self.radial_shells - 1:  # Include boundary in last shell
                mask = distances >= r_inner
                
            if np.any(mask):
                shell_coords = coords[mask]
                shell_intensities = [image[y, x] for y, x in shell_coords]
                shell_means[f'radial_shell_{i}'] = np.mean(shell_intensities)
            else:
                shell_means[f'radial_shell_{i}'] = 0
        
        # Additional radial metrics
        if self.radial_shells >= 2:
            shell_means['center_to_edge_ratio'] = (
                shell_means['radial_shell_0'] / shell_means[f'radial_shell_{self.radial_shells-1}']
                if shell_means[f'radial_shell_{self.radial_shells-1}'] > 0 else 0
            )
        
        return shell_means
    
    def _texture_features(self, region) -> Dict:
        """Extract texture features using GLCM."""
        
        # Get the nucleus image patch
        nucleus_image = region.intensity_image
        nucleus_mask = region.image
        
        # Apply mask to get only nucleus pixels
        masked_image = nucleus_image.copy()
        masked_image[~nucleus_mask] = 0
        
        # Normalize to 0-255 for GLCM
        if np.max(masked_image) > np.min(masked_image):
            normalized = ((masked_image - np.min(masked_image)) / 
                         (np.max(masked_image) - np.min(masked_image)) * 255).astype(np.uint8)
        else:
            normalized = masked_image.astype(np.uint8)
        
        try:
            # Compute GLCM
            from skimage.feature import graycomatrix, graycoprops
            
            glcm = graycomatrix(
                normalized, 
                distances=[self.texture_distance], 
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256,
                symmetric=True,
                normed=True
            )
            
            # Extract GLCM properties
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            texture_features = {}
            
            for prop in properties:
                values = graycoprops(glcm, prop)
                texture_features[f'glcm_{prop}_mean'] = np.mean(values)
                texture_features[f'glcm_{prop}_std'] = np.std(values)
            
            return texture_features
            
        except Exception as e:
            warnings.warn(f"GLCM computation failed: {e}")
            # Return zeros for all texture features
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            return {f'glcm_{prop}_{stat}': 0 for prop in properties for stat in ['mean', 'std']}
    
    def _granulometry_features(self, region) -> Dict:
        """Extract granulometry features (multi-scale morphological analysis)."""
        
        nucleus_image = region.intensity_image
        nucleus_mask = region.image
        
        # Apply mask
        masked_image = nucleus_image.copy()
        masked_image[~nucleus_mask] = 0
        
        granulometry = {}
        
        try:
            for radius in self.granulometry_radii:
                # Morphological opening with increasing radius
                structuring_element = disk(radius)
                opened = opening(masked_image, structuring_element)
                
                # Measure remaining "spots"
                opened_binary = opened > 0
                spots = measure.label(opened_binary)
                n_spots = np.max(spots)
                total_area = np.sum(opened_binary)
                
                granulometry[f'granulometry_spots_r{radius}'] = n_spots
                granulometry[f'granulometry_area_r{radius}'] = total_area
                
                # Relative measurements
                if nucleus_mask.sum() > 0:
                    granulometry[f'granulometry_area_fraction_r{radius}'] = total_area / nucleus_mask.sum()
                else:
                    granulometry[f'granulometry_area_fraction_r{radius}'] = 0
                    
        except Exception as e:
            warnings.warn(f"Granulometry computation failed: {e}")
            # Return zeros
            for radius in self.granulometry_radii:
                granulometry[f'granulometry_spots_r{radius}'] = 0
                granulometry[f'granulometry_area_r{radius}'] = 0
                granulometry[f'granulometry_area_fraction_r{radius}'] = 0
        
        return granulometry


def extract_experimental_metadata(image_names: List[str]) -> pd.DataFrame:
    """
    Extract experimental metadata from image names.
    
    Expected format: '48hr_dk[int]_[condition]_well[int]_20x*.nd2'
    """
    import re
    
    metadata = []
    
    for name in image_names:
        # Extract dk number (LSD1 group)
        dk_match = re.search(r'dk(\d+)', name)
        dk_number = int(dk_match.group(1)) if dk_match else None
        
        # Extract well number
        well_match = re.search(r'well(\d+)', name)
        well_number = int(well_match.group(1)) if well_match else None
        
        # Extract condition from filename
        condition = "unknown"
        if "wtLSD1" in name:
            condition = "wtLSD1"
        elif "gfpOnly" in name:
            condition = "gfpOnly"
        elif "catDeadK661A" in name:
            condition = "catDeadK661A" 
        elif "catDeadD556A" in name:
            condition = "catDeadD556A"
        elif "iddDeletion" in name:
            if "catDead" in name:
                condition = "iddDeletionCatDeadD556A"
            else:
                condition = "iddDeletion"
        
        metadata.append({
            'image_name': name,
            'dk_group': f'dk{dk_number}' if dk_number else 'unknown',
            'dk_number': dk_number,
            'condition': condition,
            'well': well_number,
            'timepoint': '48hr'  # Based on filename pattern
        })
    
    return pd.DataFrame(metadata)
