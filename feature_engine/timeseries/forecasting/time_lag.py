# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import warnings
from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import (
    _find_all_variables,
    _find_or_check_categorical_variables,
    _check_input_parameter_variables,
)


class TimeSeriesLagTransformer(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    periods: int, default=1
        Number of periods to shift. Can be positive or negative.

    freq: str, default=None
        Optional increment to use from the tseries module or time rule.

    axis: int, default=1
        Shift direction. Index is '0'. Columns are '1'.

    keep_original: bool, default=True
        Determines whether the dataframe keeps the original columns that are transformed.

    Attributes
    ----------


    Methods
    -------


    Notes
    -----


    See Also
    --------


    References
    ----------

    """

    def __init__(
        self,
        periods: int = 1,
        freq: str = None,
        axis: int = 0,
        keep_original: bool = True,

    ) -> None:

        if not isinstance(periods, int):
            raise ValueError(
                f"'num_periods' is {num_periods}. The variable must be an integer."
            )

        if axis not in (0, 1):
            raise ValueError(
                f"'axis' is {axis}. The variable must be 0 or 1."
            )

        if not isinstance(keep_original):
            raise ValueError(
                f"'keep_original' is {keep_original}. The variable must be boolean."
            )

        self.periods = periods
        self.freq = freq
        self.axis = axis
        self.keep_original = keep_original

