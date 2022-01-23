# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Union

import pandas as pd
from pandas.tseries.offsets import DateOffset
from pandas.tseries.frequencies import to_offset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from datetime import datetime

from feature_engine.dataframe_checks import (
    _is_dataframe,
)
from feature_engine.variable_manipulation import (
    _find_all_variables,
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

    fill_value: [int, float, str], default=None
        Character to be used to fill missing values.

    keep_original: bool, default=True
        Determines whether the dataframe keeps the original columns
        that are transformed.

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
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        periods: int = 1,
        freq: str = None,
        axis: int = 0,
        fill_value: [int, str] = None,
        keep_original: bool = True,

    ) -> None:

        if not isinstance(periods, int):
            raise ValueError(
                f"'num_periods' is {periods}. The variable must be "
                f"an integer."
            )

        if axis not in (0, 1):
            raise ValueError(
                f"'axis' is {axis}. The variable must be 0 or 1."
            )

        if not isinstance(keep_original, bool):
            raise ValueError(
                f"'keep_original' is {keep_original}. The variable must "
                f"be boolean."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.periods = periods
        self.freq = freq
        self.axis = axis
        self.fill_value = fill_value
        self.keep_original = keep_original

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates lag

        Parameters
        ----------
        df: pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing the variables that that will be lagged.

        Returns
        -------
        df_new: pandas dataframe
            The dataframe comprised of only the transformed variables or
            the original dataframe plus the transformed variables.

        """
        # check if 'df' is a dataframe
        _is_dataframe(df)

        # check if index is a datetime object
        if not isinstance(df.index[0], datetime.datetime):
            raise ValueError(
                "Dataframe's index is not a datetime object. Transformer requires"
                "the index to be a datetime object."
            )

        # check variables
        self.variables_ = _find_all_variables(df, self.variables)

        tmp = df[self.variables_].shift(
            periods=self.periods,
            freq=self.freq,
            axis=self.axis,
            fill_value=self.fill_value
        )

        tmp.columns = self.rename_variables()

        if self.keep_original:
            df = df[self.variables_].merge(
                tmp, left_index=True, right_index=True, how="left"
            )
        else:
            df = tmp.copy()

        return df

    def rename_variables(self):
        """
        Renames variables by adding the time-lag interval.

        Parameters
        ----------

        Returns
        -------
        variables_lag: list
            Names of the variables with the time-lag interval that is
            used in the transformation.

        """

        if self.freq is None:
            lag_str = f"_lag_{self.periods}pds"
        else:
            lag_str = f"_lag_{self.freq}"

        variables_lag = [name + lag_str for name in self.variables_]

        return variables_lag
