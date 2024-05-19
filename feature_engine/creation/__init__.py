"""
The module creation includes classes to create new variables by combination of existing
variables in the dataframe.
"""
# FIXME: remove in version 1.4
from .cyclical_features import CyclicalFeatures
from .math_features import MathFeatures
from .relative_features import RelativeFeatures
from .decision_tree_features import DecisionTreeFeatures

__all__ = [
    "DecisionTreeFeatures",
    "MathFeatures",
    "RelativeFeatures",
    "CyclicalFeatures",
]
