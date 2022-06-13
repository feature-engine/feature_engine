"""
The module discretisation includes classes to sort continuous variables into bins or
intervals.
"""

from .arbitrary import ArbitraryDiscretiser
from .decision_tree import DecisionTreeDiscretiser
from .equal_frequency import EqualFrequencyDiscretiser
from .equal_width import EqualWidthDiscretiser
from .chi_merge import ChiMergeDiscretiser

__all__ = [
    "DecisionTreeDiscretiser",
    "EqualFrequencyDiscretiser",
    "EqualWidthDiscretiser",
    "ArbitraryDiscretiser",
    "ChiMergeDiscretiser",
]
