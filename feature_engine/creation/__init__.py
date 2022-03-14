"""
The module creation includes classes to create new variables by combination of existing
variables in the dataframe.
"""
from .cyclical_features import CyclicalFeatures
from .math_features import MathFeatures
from .relative_features import RelativeFeatures

# FIXME: remove in version 1.4
from .combine_with_reference_feature import CombineWithReferenceFeature
from .cyclical import CyclicalTransformer
from .mathematical_combination import MathematicalCombination

__all__ = [
    "MathematicalCombination",
    "CombineWithReferenceFeature",
    "CyclicalTransformer",
    "MathFeatures",
    "RelativeFeatures",
    "CyclicalFeatures",
]
