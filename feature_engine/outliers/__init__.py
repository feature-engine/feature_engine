"""
The module outliers includes classes to remove or cap outliers.
"""

from .artbitrary import ArbitraryOutlierCapper
from .trimmer import OutlierTrimmer
from .winsorizer import Winsorizer

__all__ = ["Winsorizer", "ArbitraryOutlierCapper", "OutlierTrimmer"]
