"""
The module selection includes classes to select features or remove unwanted features
"""

from .drop_features import DropFeatures
from .drop_constant_features import DropConstantFeatures

__all__ = [
    'DropFeatures',
    'DropConstantFeatures'
]