"""
The module selection includes classes to select features or remove unwanted features.
"""
from .eliminate_feature_by_feature_importance import RecursiveFeatureElimination
from .drop_features import DropFeatures
from .drop_constant_features import DropConstantFeatures
from .drop_duplicate_features import DropDuplicateFeatures
from .drop_correlated_features import DropCorrelatedFeatures
from .shuffle_features import ShuffleFeaturesSelector

__all__ = [
    "DropFeatures",
    "DropConstantFeatures",
    "DropDuplicateFeatures",
    "DropCorrelatedFeatures",
    "ShuffleFeaturesSelector",
    "RecursiveFeatureElimination"
]
