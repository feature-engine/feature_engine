"""
The module discretisation includes classes to sort continuous variables into bins or
intervals.
"""

from .arbitrary import ArbitraryDiscretiser
from .decision_tree import DecisionTreeDiscretiser
from .equal_frequency import EqualFrequencyDiscretiser
from .equal_width import EqualWidthDiscretiser
from .geometric_width import GeometricWidthDiscretiser

__all__ = [
    "DecisionTreeDiscretiser",
    "EqualFrequencyDiscretiser",
    "EqualWidthDiscretiser",
    "ArbitraryDiscretiser",
    "GeometricWidthDiscretiser",
]
