"""
The module selection includes classes to select features or remove unwanted features.
"""

from .drop_features import DropFeatures
from .drop_constant_features import DropConstantFeatures
from .drop_duplicate_features import DropDuplicateFeatures
from .shuffle_features import ShuffleFeaturesSelector

__all__ = [
    "DropFeatures",
    "DropConstantFeatures",
    "DropDuplicateFeatures",
    "ShuffleFeaturesSelector",
]
