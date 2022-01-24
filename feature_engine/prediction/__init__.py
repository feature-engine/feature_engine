"""
This module includes predictor classes.
"""

from .target_mean_classifier import TargetMeanClassifier
from .target_mean_regressor import TargetMeanRegressor

__all__ = [
    "TargetMeanClassifier",
    "TargetMeanRegressor",
]
