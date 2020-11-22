"""
The module selection includes classes to select features or remove unwanted features.
"""
from .drop_features import DropFeatures
from .drop_constant_features import DropConstantFeatures
from .drop_duplicate_features import DropDuplicateFeatures
from .drop_correlated_features import DropCorrelatedFeatures
from .shuffle_features import SelectByShuffling
from .single_feature_performance_selection import SelectBySingleFeaturePerformance
from .recursive_feature_elimination import RecursiveFeatureElimination
from .target_mean_selection import SelectByTargetMeanPerformance

__all__ = [
    "DropFeatures",
    "DropConstantFeatures",
    "DropDuplicateFeatures",
    "DropCorrelatedFeatures",
    "SelectByShuffling",
    "SelectBySingleFeaturePerformance",
    "RecursiveFeatureElimination",
    "SelectByTargetMeanPerformance",
]
