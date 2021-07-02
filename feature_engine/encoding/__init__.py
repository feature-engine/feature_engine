"""
The module encoding includes classes to transform categorical variables into numerical.
"""

from .count_frequency import CountFrequencyEncoder
from .decision_tree import DecisionTreeEncoder
from .mean_encoding import MeanEncoder
from .one_hot import OneHotEncoder
from .ordinal import OrdinalEncoder
from .probability_ratio import PRatioEncoder
from .rare_label import RareLabelEncoder
from .woe import WoEEncoder

__all__ = [
    "CountFrequencyEncoder",
    "DecisionTreeEncoder",
    "MeanEncoder",
    "OneHotEncoder",
    "OrdinalEncoder",
    "RareLabelEncoder",
    "WoEEncoder",
    "PRatioEncoder",
]
