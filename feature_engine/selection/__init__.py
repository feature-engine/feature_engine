"""
The module selection includes classes to select features, remove unwanted features or shuffle feature values
"""

from .drop_features import DropFeatures
from .drop_constant_features import DropConstantFeatures
from .drop_duplicate_features import DropDuplicateFeatures
from .shuffle_features import ShuffleFeatures

__all__ = ["DropFeatures", "DropConstantFeatures", "DropDuplicateFeatures", "ShuffleFeatures"]
