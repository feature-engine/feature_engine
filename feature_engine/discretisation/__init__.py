"""
The module discretisation includes classes to sort continuous variables into bins or
intervals.
"""

from .decision_tree import DecisionTreeDiscretiser
from .equal_frequency import EqualFrequencyDiscretiser
from .equal_width import EqualWidthDiscretiser
from .arbitrary import ArbitraryDiscretiser

__all__ = [
    "DecisionTreeDiscretiser",
    "EqualFrequencyDiscretiser",
    "EqualWidthDiscretiser",
    "ArbitraryDiscretiser",
]
