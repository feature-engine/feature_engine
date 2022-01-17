"""
This module includes a predictor that returns the mean for discretised
and encoded variables.
"""

from .target_mean_regressor import TargetMeanRegressor
from .target_mean_classifier import TargetMeanClassifier

__all__ = [
    "TargetMeanClassifier",
    "TargetMeanRegressor",
]
