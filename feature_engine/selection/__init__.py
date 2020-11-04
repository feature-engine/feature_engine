"""
The module selection includes classes to select features or remove unwanted features.
"""

from .drop_features import DropFeatures
from .drop_constant_features import DropConstantFeatures
from .drop_duplicate_features import DropDuplicateFeatures
from .drop_correlated_features import DropCorrelatedFeatures
from .shuffle_features import ShuffleFeaturesSelector
from .single_feature_performance_selection import SignleFeaturePerformanceSelection

__all__ = [
    "DropFeatures",
    "DropConstantFeatures",
    "DropDuplicateFeatures",
    "DropCorrelatedFeatures",
    "ShuffleFeaturesSelector",
    "SignleFeaturePerformanceSelection",
]
