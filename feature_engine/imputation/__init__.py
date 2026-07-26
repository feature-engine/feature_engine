"""
The module imputation includes classes to perform missing data imputation
"""

from .arbitrary_number import ArbitraryNumberImputer, ArbitraryImputer
from .categorical import CategoricalImputer
from .drop_missing_data import DropMissingData
from .end_tail import EndTailImputer
from .mean_median import MeanMedianImputer, MeanImputer
from .missing_indicator import AddMissingIndicator, MissingIndicator
from .random_sample import RandomSampleImputer

__all__ = [
    "MeanMedianImputer",      # deprecated
    "ArbitraryNumberImputer", # deprecated
    "AddMissingIndicator",    # deprecated
    "MeanImputer",
    "ArbitaryImputer",
    "CategoricalImputer",
    "EndTailImputer",
    "MissingIndicator",
    "RandomSampleImputer",
    "DropMissingData",
]

