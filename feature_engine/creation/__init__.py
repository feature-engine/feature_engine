"""
The module creation includes classes to create new variables by combination of existing
variables in the dataframe.
"""

from .cyclical_features import CyclicalFeatures
from .decision_tree_features import DecisionTreeFeatures
from .math_features import MathFeatures
from .relative_features import RelativeFeatures
from .distance_features import DistanceFeatures

__all__ = [
    "DecisionTreeFeatures",
    "MathFeatures",
    "RelativeFeatures",
    "CyclicalFeatures",
    "DistanceFeatures",
]
