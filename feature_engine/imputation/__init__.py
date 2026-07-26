"""
The module imputation includes classes to perform missing data imputation
"""

from .arbitrary_number import ArbitraryNumberImputer
from .categorical import CategoricalImputer
from .drop_missing_data import DropMissingData
from .end_tail import EndTailImputer
from .mean_median import MeanMedianImputer, MeanImputer
from .missing_indicator import AddMissingIndicator, MissingIndicator
from .random_sample import RandomSampleImputer

__all__ = [
    "MeanMedianImputer", # deprecated
    "MeanImputer",
    "ArbitraryNumberImputer",
    "CategoricalImputer",
    "EndTailImputer",
    "AddMissingIndicator", # deprecated
    "MissingIndicator",
    "RandomSampleImputer",
    "DropMissingData",
]

