"""
This module includes predictor classes.
"""

from .target_mean_regressor import TargetMeanRegressor
from .target_mean_classifier import TargetMeanClassifier

__all__ = [
    "TargetMeanClassifier",
    "TargetMeanRegressor",
]
