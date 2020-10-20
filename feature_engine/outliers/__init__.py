"""
The module outliers includes classes to remove or cap outliers.
"""

from .winsorizer import Winsorizer
from .artbitrary import ArbitraryOutlierCapper
from .trimmer import OutlierTrimmer

__all__ = ["Winsorizer", "ArbitraryOutlierCapper", "OutlierTrimmer"]
