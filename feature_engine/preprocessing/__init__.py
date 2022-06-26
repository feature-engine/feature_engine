"""
The module preprocessing includes classes and functions for general data pre-processing
and transformation.
"""

from .match_categories import MatchCategories
from .match_columns import MatchVariables
from .category_encoder import CategoryEncoder

__all__ = [
    "MatchCategories",
    "MatchVariables",
    "CategoryEncoder"
]
