"""
The module creation includes classes to create new variables by combination of existing
variables in the dataframe.
"""

from .mathematical_combination import MathematicalCombination
from .combine_with_reference_feature import CombineWithReferenceFeature
from .cyclical import CyclicalTransformer

__all__ = [
    "MathematicalCombination",
    "CombineWithReferenceFeature",
    "CyclicalTransformer",
]
