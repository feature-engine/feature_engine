"""
The module selection includes classes to select features or remove unwanted features.
"""
from .drop_constant_features import DropConstantFeatures
from .drop_correlated_features import DropCorrelatedFeatures
from .drop_duplicate_features import DropDuplicateFeatures
from .drop_features import DropFeatures
from .drop_psi_features import DropHighPSIFeatures
from .information_value import SelectByInformationValue
from .probe_feature_selection import ProbeFeatureSelection
from .recursive_feature_addition import RecursiveFeatureAddition
from .recursive_feature_elimination import RecursiveFeatureElimination
from .shuffle_features import SelectByShuffling
from .single_feature_performance import SelectBySingleFeaturePerformance
from .smart_correlation_selection import SmartCorrelatedSelection
from .target_mean_selection import SelectByTargetMeanPerformance
from .mrmr import MRMR

__all__ = [
    "DropFeatures",
    "DropConstantFeatures",
    "DropDuplicateFeatures",
    "DropCorrelatedFeatures",
    "DropHighPSIFeatures",
    "SmartCorrelatedSelection",
    "SelectByShuffling",
    "SelectBySingleFeaturePerformance",
    "RecursiveFeatureAddition",
    "RecursiveFeatureElimination",
    "SelectByTargetMeanPerformance",
    "SelectByInformationValue",
    "ProbeFeatureSelection",
    "MRMR",
]
