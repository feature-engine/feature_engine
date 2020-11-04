"""
The module selection includes classes to select features or remove unwanted features.
"""
from .drop_features import DropFeatures
from .drop_constant_features import DropConstantFeatures
from .drop_duplicate_features import DropDuplicateFeatures
from .drop_correlated_features import DropCorrelatedFeatures
from .recursive_feature_elimination import RecursiveFeatureElimination
from .shuffle_features import ShuffleFeaturesSelector
from .feature_selection import FeatureSelection

__all__ = [
    "DropFeatures",
    "DropConstantFeatures",
    "DropDuplicateFeatures",
    "DropCorrelatedFeatures",
    "ShuffleFeaturesSelector",
    "RecursiveFeatureElimination",
    "FeatureSelection",
]
