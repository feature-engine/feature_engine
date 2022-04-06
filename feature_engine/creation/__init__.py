"""
The module creation includes classes to create new variables by combination of existing
variables in the dataframe.
"""
# FIXME: remove in version 1.4
from .combine_with_reference_feature import CombineWithReferenceFeature
from .cyclical import CyclicalTransformer
from .cyclical_features import CyclicalFeatures
from .math_features import MathFeatures
from .mathematical_combination import MathematicalCombination
from .relative_features import RelativeFeatures

__all__ = [
    "MathematicalCombination",
    "CombineWithReferenceFeature",
    "CyclicalTransformer",
    "MathFeatures",
    "RelativeFeatures",
    "CyclicalFeatures",
]
