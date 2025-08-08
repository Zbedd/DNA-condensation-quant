# This module has been integrated into feature_extractor.py
# Import the main functionality:

from .feature_extractor import DNACondensationFeatureExtractor

# For backward compatibility
def extract_morphological_features(*args, **kwargs):
    """Legacy function - use DNACondensationFeatureExtractor instead."""
    extractor = DNACondensationFeatureExtractor()
    return extractor._morphological_features(*args, **kwargs)