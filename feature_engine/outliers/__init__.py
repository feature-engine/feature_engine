"""
The module outliers includes classes to remove or cap outliers.
"""

from .artbitrary import ArbitraryOutlierCapper
from .winsorizer import Winsorizer
from .trimmer import OutlierTrimmer

__all__ = ["Winsorizer", "ArbitraryOutlierCapper", "OutlierTrimmer"]
