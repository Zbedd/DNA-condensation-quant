"""
Feature extraction for DNA condensation analysis.

This module implements comprehensive feature extraction for segmented nuclei,
focusing on morphological and intensity distribution features that report on
DNA condensation levels.
"""

import numpy as np
import pandas as pd
from skimage import measure, feature
from skimage.filters import rank, sobel
from skimage.morphology import disk, opening
from scipy import ndimage
from scipy.stats import entropy
import warnings
from typing import Dict, List, Optional
from dna_condensation.pipeline.config import Config
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
    
    Quality filtering is applied based on configuration to remove artifacts.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize feature extractor with configuration.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config if config is not None else Config()
        
        # Load quality filtering parameters from config
        quality_config = self.config.get('quality_filtering', {})
        self.quality_filtering_enabled = quality_config.get('enabled', True)
        
        # Area filtering percentiles (adaptive)
        area_config = quality_config.get('area_percentiles', {})
        self.min_area_percentile = area_config.get('min_percentile', 5)
        self.max_area_percentile = area_config.get('max_percentile', 95)
        
        # Intensity filtering thresholds
        intensity_config = quality_config.get('intensity_filtering', {})
        self.min_dynamic_range = intensity_config.get('min_dynamic_range', 1e-6)
        self.min_mean_intensity = intensity_config.get('min_mean_intensity', 1e-6)
        self.min_pixel_count = intensity_config.get('min_pixel_count', 10)
        
        # Geometry filtering
        geometry_config = quality_config.get('geometry_filtering', {})
        self.min_perimeter = geometry_config.get('min_perimeter', 4)
        
        # Feature extraction parameters - make configurable
        feature_params = self.config.get('feature_extraction', {})
        self.high_intensity_percentile = feature_params.get('high_intensity_percentile', 90)
        self.radial_shells = feature_params.get('radial_shells', 5)
        self.texture_distance = feature_params.get('texture_distance', 1)
        self.granulometry_radii = feature_params.get('granulometry_radii', [3, 5, 7, 10, 15])
        # Numerical stability for ratios (e.g., CV); aligns with README guidance
        self.epsilon = feature_params.get('epsilon', 1e-8)
    # Chromatin Condensation Parameter (CCP) config (see nuclear_stain_condensation_metrics.md)
        self.ccp_percentile = float(feature_params.get('ccp_percentile', 0.90))  # q in [0,1]
        self.ccp_erode_px = int(feature_params.get('ccp_erode_px', 0))  # optional 1px erosion
 
    def extract_features_batch(self, 
                              images: List[np.ndarray], 
                              masks: List[np.ndarray],
                              image_names: List[str] = None,
                              feature_types: Optional[List[str]] = None) -> pd.DataFrame:
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
        feature_types : List[str], optional
            Specific feature types to extract. Options: 'intensity', 'morphology', 
            'spatial', 'texture', 'granulometry'. If None, all are extracted.
            
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
                
            features = self.extract_features_single(image, mask, name, feature_types)
            all_features.append(features)
            
        if all_features:
            return pd.concat(all_features, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def extract_features_single(self, 
                               image: np.ndarray, 
                               mask: np.ndarray,
                               image_name: str = "unknown",
                               feature_types: Optional[List[str]] = None) -> pd.DataFrame:
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
        feature_types : List[str], optional
            Specific feature types to extract.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with one row per nucleus
        """
        # Compute gradient magnitude once per image for CCP (Sobel on raw I)
        # Note: this is per spec; CCP should not use per-nucleus normalization.
        grad_image = sobel(image)
 
        # Get region properties
        try:
            regions = measure.regionprops(mask, intensity_image=image)
        except TypeError as e:
            # Normalize error for tests expecting ValueError/IndexError on bad inputs
            raise ValueError(f"Invalid mask/image for regionprops: {e}")
        
        features_list = []
        
        # Adaptive area filtering: calculate area percentiles if quality filtering is enabled
        if self.quality_filtering_enabled and regions:
            areas = [region.area for region in regions]
            min_area_threshold = np.percentile(areas, self.min_area_percentile)
            max_area_threshold = np.percentile(areas, self.max_area_percentile)
        else:
            # Fallback to permissive thresholds if filtering disabled
            min_area_threshold = 0
            max_area_threshold = float('inf')
        
        for region in regions:
            # Adaptive quality filters based on actual data distribution
            if self.quality_filtering_enabled:
                if (region.area < min_area_threshold or      # Below min percentile
                    region.area > max_area_threshold or      # Above max percentile
                    region.perimeter < self.min_perimeter):  # Invalid geometry
                    continue
                
            try:
                features = self._extract_nucleus_features(region, image, mask, image_name, feature_types, grad_image)
                
                # Additional quality check after feature extraction
                if features is None:
                    continue
                    
                features_list.append(features)
            except Exception as e:
                warnings.warn(f"Failed to extract features for nucleus {region.label} in {image_name}: {e}")
                continue
                
        return pd.DataFrame(features_list)
    
    def _extract_nucleus_features(self, 
                                 region,
                                 image: np.ndarray,
                                 mask: np.ndarray, 
                                 image_name: str,
                                 feature_types: Optional[List[str]] = None,
                                 grad_image: Optional[np.ndarray] = None) -> Dict:
        """Extract all features for a single nucleus."""
        
        # If feature_types is None, extract all features
        if feature_types is None:
            feature_types = ['intensity', 'morphology', 'spatial', 'texture', 'granulometry']

        # Basic identifiers
        features = {
            'image_name': image_name,
            'nucleus_id': region.label,
            'centroid_y': region.centroid[0],
            'centroid_x': region.centroid[1],
        }
        
        # Get intensity values for this nucleus
        intensity_values = region.intensity_image[region.image].flatten()
        
        # Quality check: Skip regions with insufficient dynamic range
        intensity_range = np.max(intensity_values) - np.min(intensity_values)
        mean_intensity = np.mean(intensity_values)
        
        if (intensity_range < 1e-6 or  # Nearly zero dynamic range
            mean_intensity < 1e-6 or   # Nearly zero intensity (background)
            len(intensity_values) < 10):  # Too few pixels
            return None  # Skip this region
        
        # 1. Intensity Distribution Features
        if 'intensity' in feature_types:
            features.update(self._intensity_features(intensity_values, region))
        
        # 2. Morphological Features  
        if 'morphology' in feature_types:
            features.update(self._morphological_features(region))
        
        # 3. Spatial/Radial Features
        if 'spatial' in feature_types:
            features.update(self._radial_features(region, image, grad_image))
        
        # 4. Texture Features
        if 'texture' in feature_types:
            features.update(self._texture_features(region))
        
        # 5. Granulometry Features
        if 'granulometry' in feature_types:
            features.update(self._granulometry_features(region))
        
        return features
    
    def _intensity_features(self, intensity_values: np.ndarray, region=None) -> Dict:
        """
        Extract comprehensive intensity distribution features.
        
        These features quantify the distribution of pixel intensities within a nucleus
        to characterize DNA condensation. Higher CV and skewness typically indicate
        more heterogeneous chromatin organization.
        
        Args:
            intensity_values: 1D array of pixel intensities within the nucleus
            
        Returns:
            Dictionary with statistical measures of intensity distribution
        """
        
        # Core statistical measures
        mean_intensity = np.mean(intensity_values)
        std_intensity = np.std(intensity_values)
        total_intensity = np.sum(intensity_values)
        
        features = {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            # Coefficient of variation - computed on pre-normalized intensities with epsilon
            'coefficient_of_variation': std_intensity / (mean_intensity + self.epsilon),
            # Nuclear density - captures chromatin compaction (mean intensity * area / total intensity)
            'nuclear_density': (
                (mean_intensity * getattr(region, 'area', 1) / total_intensity)
                if (total_intensity > 0 and region is not None)
                else mean_intensity
            ),
            'min_intensity': float(np.min(intensity_values)),
            'max_intensity': float(np.max(intensity_values)),
            'intensity_range': float(np.max(intensity_values) - np.min(intensity_values)),
        }
        
        # Percentile-based features for robust distribution characterization
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            features[f'intensity_p{p}'] = float(np.percentile(intensity_values, p))
        
        # High intensity fraction - indicates proportion of bright chromatin
        high_threshold = np.percentile(intensity_values, self.high_intensity_percentile)
        features['high_intensity_fraction'] = float(np.mean(intensity_values >= high_threshold))

    # CI (Condensation Index) is computed centrally in batch_processor after analysis
        
        # Distribution shape characteristics - with robustness checks
        from scipy.stats import skew, kurtosis
        
        # Use CV to gate higher-order moments to avoid numerical instability
        cv = std_intensity / (mean_intensity + self.epsilon)
        if cv > 0.01 and len(intensity_values) > 10:
            try:
                features['intensity_skewness'] = float(skew(intensity_values))
                features['intensity_kurtosis'] = float(kurtosis(intensity_values))
            except Exception:
                features['intensity_skewness'] = 0.0
                features['intensity_kurtosis'] = 0.0
        else:
            features['intensity_skewness'] = 0.0
            features['intensity_kurtosis'] = 0.0
        
        # Entropy quantifies intensity variability (higher = more disordered)
        hist, _ = np.histogram(intensity_values, bins=256)
        total = float(np.sum(hist))
        if total > 0:
            hist = hist / total  # Normalize to probabilities
        features['intensity_entropy'] = float(entropy(hist + 1e-10))  # Small constant prevents log(0)
        
        return features
    
    def _morphological_features(self, region) -> Dict:
        """
        Extract morphological shape and size features from nucleus regions.
        
        These features characterize nucleus geometry and can indicate changes
        in nuclear envelope structure associated with condensation states.
        
        Returns:
            Dictionary with shape descriptors including area, perimeter, 
            eccentricity, and derived metrics like circularity and aspect ratio.
        """
        
        return {
            # Basic size measurements
            'area': region.area,  # Number of pixels in nucleus
            'perimeter': region.perimeter,  # Boundary length
            
            # Shape characterization via fitted ellipse
            'major_axis_length': region.major_axis_length,  # Length of major axis
            'minor_axis_length': region.minor_axis_length,  # Length of minor axis
            'eccentricity': region.eccentricity,  # How elliptical (0=circle, 1=line)
            'orientation': region.orientation,  # Angle of major axis
            
            # Shape quality metrics
            'solidity': region.solidity,  # Area/convex_hull_area (measures concavity)
            'extent': region.extent,  # Area/bounding_box_area (measures rectangularity)
            'equivalent_diameter': region.equivalent_diameter,  # Diameter of equivalent circle
            
            # Derived shape metrics
            'aspect_ratio': region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0,
            'circularity': 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0,  # Nuclear compactness: 1=perfect circle, increases with condensation
        }
    
    def _radial_features(self, region, image: np.ndarray, grad_image: Optional[np.ndarray] = None) -> Dict:
        """
        Extract radial intensity profile features to quantify spatial organization.
        
        Creates concentric shells from nucleus center to edge and measures
        average intensity in each shell. Useful for detecting center-to-edge
        intensity gradients that indicate chromatin condensation patterns.
        Also computes CCP (chromatin condensation parameter) as the fraction of
        pixels within the nucleus whose Sobel gradient magnitude exceeds a per-nucleus percentile.
        
        Returns:
            Dictionary with mean intensity for each radial shell plus 
            center-to-edge ratio metric.
        """
        
        # Get pixel coordinates relative to nucleus centroid
        coords = region.coords  # (N, 2) array of (y, x) coordinates
        centroid = region.centroid  # (y, x) center position
        
        # Calculate Euclidean distance from each pixel to centroid
        distances = np.sqrt(np.sum((coords - centroid) ** 2, axis=1))
        max_distance = np.max(distances)
        
        # Handle edge case of single-pixel nucleus
        if max_distance == 0:
            return {f'radial_shell_{i}': region.mean_intensity for i in range(self.radial_shells)}
        
        # Create concentric shells from center to edge
        shell_means = {}
        for i in range(self.radial_shells):
            # Define shell boundaries as fractions of max distance
            r_inner = i * max_distance / self.radial_shells
            r_outer = (i + 1) * max_distance / self.radial_shells
            
            # Find pixels in this shell (annular region)
            mask = (distances >= r_inner) & (distances < r_outer)
            
            # Include boundary pixels in outermost shell
            if i == self.radial_shells - 1:
                mask = distances >= r_inner
                
            if np.any(mask):
                # Get coordinates and intensities for pixels in this shell
                shell_coords = coords[mask]
                shell_intensities = [image[y, x] for y, x in shell_coords]
                shell_means[f'radial_shell_{i}'] = np.mean(shell_intensities)
            else:
                # Empty shell (shouldn't happen with proper shell sizing)
                shell_means[f'radial_shell_{i}'] = 0
        
        # Calculate center-to-edge intensity ratio (condensation indicator)
        if self.radial_shells >= 2:
            center_intensity = shell_means['radial_shell_0']
            edge_intensity = shell_means[f'radial_shell_{self.radial_shells-1}']
            
            shell_means['center_to_edge_ratio'] = (
                center_intensity / edge_intensity if edge_intensity > 0 else 0
            )
        
        # CCP: edge fraction using per-nucleus percentile threshold on Sobel gradient
        try:
            if grad_image is not None:
                # Crop gradient to region bbox
                minr, minc, maxr, maxc = region.bbox
                G_patch = grad_image[minr:maxr, minc:maxc]
                nucleus_mask = region.image.astype(bool)
                if self.ccp_erode_px > 0:
                    # Optional erosion to avoid boundary artifacts
                    eroded_mask = ndimage.binary_erosion(nucleus_mask, iterations=int(self.ccp_erode_px))
                else:
                    eroded_mask = nucleus_mask
                # If erosion removes everything, fall back to original mask
                if not eroded_mask.any():
                    eroded_mask = nucleus_mask
                # Percentile threshold within the (optionally eroded) nucleus mask
                vals = G_patch[eroded_mask]
                if vals.size >= 1:
                    q = float(np.quantile(vals, self.ccp_percentile))
                    area = int(nucleus_mask.sum())
                    if area > 0:
                        edge_count = int((G_patch[nucleus_mask] >= q).sum())
                        shell_means['ccp'] = edge_count / float(area)
                    else:
                        shell_means['ccp'] = 0.0
                else:
                    shell_means['ccp'] = 0.0
        except Exception as e:
            warnings.warn(f"CCP computation failed for nucleus {region.label}: {e}")
            shell_means['ccp'] = 0.0
 
        return shell_means
    
    def _texture_features(self, region) -> Dict:
        """
        Extract texture features using Gray-Level Co-occurrence Matrix (GLCM).
        
        This implementation computes GLCM only within the nucleus mask to avoid
        background artifacts that would make homogeneity measurements constant.
        Texture features quantify spatial patterns in intensity that indicate
        chromatin organization and DNA condensation.
        
        Returns:
            Dictionary with GLCM properties (contrast, dissimilarity, homogeneity, 
            energy, correlation) computed across multiple angles, with mean and std.
        """
        
        # Get the nucleus image patch and its boolean mask
        nucleus_image = region.intensity_image
        # Ensure 2D image (handle unexpected channel dimensions)
        if getattr(nucleus_image, 'ndim', 2) == 3:
            nucleus_image = np.mean(nucleus_image, axis=-1)
        nucleus_mask = region.image.astype(bool)

        # Skip texture analysis for tiny regions
        if nucleus_mask.sum() < 2:
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            return {f'glcm_{prop}_{stat}': 0 for prop in properties for stat in ['mean', 'std']}

        # Normalize intensities using only mask pixels to avoid background bias
        levels = 256
        masked_vals = nucleus_image[nucleus_mask]
        min_val = float(masked_vals.min())
        max_val = float(masked_vals.max())
        if max_val > min_val:
            normalized = ((nucleus_image - min_val) / (max_val - min_val) * (levels - 1)).astype(np.uint8)
        else:
            # Handle constant intensity regions
            normalized = np.zeros_like(nucleus_image, dtype=np.uint8)

        # Define angles for directional texture analysis
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°
        d = int(self.texture_distance)

        def _offset(angle: float, distance: int) -> Tuple[int, int]:
            """Convert angle to pixel offset (dy, dx) for neighbor sampling."""
            if angle == 0:
                return 0, distance  # Horizontal →
            elif angle == np.pi/4:
                return -distance, distance  # Diagonal ↗
            elif angle == np.pi/2:
                return -distance, 0  # Vertical ↑
            elif angle == 3*np.pi/4:
                return -distance, -distance  # Diagonal ↖
            # Generic fallback for arbitrary angles
            return int(round(-distance * np.sin(angle))), int(round(distance * np.cos(angle)))

        glcms = []
        H, W = normalized.shape
        
        # Build GLCM for each angle by counting co-occurrences within mask only
        for ang in angles:
            dy, dx = _offset(ang, d)
            
            # Find valid pixel pairs where both pixels are within image bounds
            y0_start = max(0, -dy)
            y0_end = min(H, H - dy)  # exclusive end
            x0_start = max(0, -dx)
            x0_end = min(W, W - dx)  # exclusive end
            
            # Skip if no valid region exists for this angle/distance
            if (y0_end - y0_start) <= 0 or (x0_end - x0_start) <= 0:
                continue

            # Calculate corresponding coordinates for neighbor pixels
            y1_start = y0_start + dy
            y1_end = y0_end + dy
            x1_start = x0_start + dx
            x1_end = x0_end + dx

            # Extract intensity values and masks for pixel pairs
            I0 = normalized[y0_start:y0_end, x0_start:x0_end]  # Reference pixels
            I1 = normalized[y1_start:y1_end, x1_start:x1_end]  # Neighbor pixels
            M0 = nucleus_mask[y0_start:y0_end, x0_start:x0_end]  # Reference mask
            M1 = nucleus_mask[y1_start:y1_end, x1_start:x1_end]  # Neighbor mask
            
            # Only count pairs where both pixels are inside the nucleus
            valid = M0 & M1
            if not np.any(valid):
                continue

            # Get intensity pairs for co-occurrence counting
            p_i = I0[valid].ravel()  # Reference intensities
            p_j = I1[valid].ravel()  # Neighbor intensities

            # Build co-occurrence matrix by counting intensity pairs
            glcm = np.zeros((levels, levels), dtype=np.float64)
            np.add.at(glcm, (p_i, p_j), 1)  # Increment count for each (i,j) pair
            
            # Make symmetric (count both (i,j) and (j,i) directions)
            glcm = glcm + glcm.T
            
            # Normalize to probabilities
            total = glcm.sum()
            if total > 0:
                glcm = glcm / total
            else:
                continue  # Skip empty matrices

            glcms.append(glcm)

        # Return zeros if no valid GLCMs were computed
        if len(glcms) == 0:
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            return {f'glcm_{prop}_{stat}': 0 for prop in properties for stat in ['mean', 'std']}

        # Compute texture properties from each GLCM
        contrasts = []
        dissimilarities = []
        homogeneities = []
        energies = []
        correlations = []

        # Pre-compute coordinate grids for efficient property calculation
        i_idx = np.arange(levels, dtype=np.float64)
        j_idx = np.arange(levels, dtype=np.float64)
        I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
        diff = I - J  # Intensity differences
        absdiff = np.abs(diff)  # Absolute differences
        denom_h = 1.0 + diff * diff  # Homogeneity denominator

        for P in glcms:
            # Calculate marginal distributions
            px = P.sum(axis=1)  # Row sums
            py = P.sum(axis=0)  # Column sums
            mu_x = (i_idx * px).sum()  # Mean of row indices
            mu_y = (j_idx * py).sum()  # Mean of column indices
            sig_x = np.sqrt(((i_idx - mu_x) ** 2 * px).sum())  # Std of row indices
            sig_y = np.sqrt(((j_idx - mu_y) ** 2 * py).sum())  # Std of column indices

            # Compute GLCM texture properties
            contrasts.append(float((P * (diff ** 2)).sum()))  # Local intensity variation
            dissimilarities.append(float((P * absdiff).sum()))  # Linear local variation
            homogeneities.append(float((P / denom_h).sum()))  # Inverse of contrast
            energies.append(float((P * P).sum()))  # Uniformity of distribution
            
            # Correlation (linear dependency between pixels)
            if sig_x > 0 and sig_y > 0:
                correlations.append(float(((I - mu_x) * (J - mu_y) * P).sum() / (sig_x * sig_y)))
            else:
                correlations.append(0.0)  # No variation means no correlation

        texture_features = {
            'glcm_contrast_mean': float(np.mean(contrasts)),
            'glcm_contrast_std': float(np.std(contrasts)),
            'glcm_dissimilarity_mean': float(np.mean(dissimilarities)),
            'glcm_dissimilarity_std': float(np.std(dissimilarities)),
            'glcm_homogeneity_mean': float(np.mean(homogeneities)),
            'glcm_homogeneity_std': float(np.std(homogeneities)),
            'glcm_energy_mean': float(np.mean(energies)),
            'glcm_energy_std': float(np.std(energies)),
            'glcm_correlation_mean': float(np.mean(correlations)),
            'glcm_correlation_std': float(np.std(correlations)),
        }

        return texture_features
    def _granulometry_features(self, region) -> Dict:
        """
        Extract granulometry features using multi-scale morphological analysis.
        
        Granulometry applies morphological opening with increasing disk sizes
        to characterize the size distribution of bright structures within nuclei.
        This helps quantify chromatin granularity and condensation patterns.
        
        Returns:
            Dictionary with spot counts, areas, and area fractions for each
            disk radius used in the opening operations.
        """
        
        nucleus_image = region.intensity_image
        # Ensure we have a 2D image
        if getattr(nucleus_image, 'ndim', 2) == 3:
            nucleus_image = np.mean(nucleus_image, axis=-1)
        nucleus_mask = region.image.astype(bool)
        
        # Apply nucleus mask to focus analysis within the nucleus
        masked_image = nucleus_image.copy()
        masked_image[~nucleus_mask] = 0
        
        granulometry = {}
        
        try:
            # Apply morphological opening with progressively larger disks
            for radius in self.granulometry_radii:
                # Create circular structuring element
                structuring_element = disk(radius)
                
                # Opening removes structures smaller than the disk
                # Remaining bright regions indicate structures >= disk size
                opened = opening(masked_image, structuring_element)
                
                # Identify and count remaining bright structures
                opened_binary = opened > 0
                spots = measure.label(opened_binary)
                n_spots = int(np.max(spots)) if spots.size > 0 else 0
                total_area = int(np.sum(opened_binary))
                
                # Store absolute measurements
                granulometry[f'granulometry_spots_r{radius}'] = n_spots
                granulometry[f'granulometry_area_r{radius}'] = total_area
                
                # Calculate relative area (normalized by nucleus size)
                area_mask = int(nucleus_mask.sum())
                if area_mask > 0:
                    granulometry[f'granulometry_area_fraction_r{radius}'] = total_area / area_mask
                else:
                    granulometry[f'granulometry_area_fraction_r{radius}'] = 0.0
                    
        except Exception as e:
            warnings.warn(f"Granulometry computation failed: {e}")
            # Return zeros on failure
            for radius in self.granulometry_radii:
                granulometry[f'granulometry_spots_r{radius}'] = 0
                granulometry[f'granulometry_area_r{radius}'] = 0
                granulometry[f'granulometry_area_fraction_r{radius}'] = 0.0
        
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
