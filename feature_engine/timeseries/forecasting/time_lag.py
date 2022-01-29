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
    _find_or_check_numerical_variables,
    _check_input_parameter_variables,
)


class TimeSeriesLagTransformer(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    variables: list, default=None
        The list of numerical variables to transform. If None, the transformer
        will automatically select all numerical variables.

    periods: int, default=1
        Number of periods to shift. Can be positive or negative.

    freq: str, default=None
        Optional increment to use from the tseries module or time rule.

    drop_original: bool, default=True
        Determines whether the dataframe keeps the original columns
        that are transformed.

    Attributes
    ----------

    variables_numerical_:
        The group of numerical input variables.

    n_features_in_:
        The number of features in the dataframe.

    Methods
    -------
    fit:
        Obtain attributes and determine whether the dataframe is eligible to
        be transformed.

    transform:
        Perform the time-lag transformation on the provided dataframe.

    rename_variables:
        Create names for the new time-lagged variables.

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
        drop_original: bool = False,

    ) -> None:

        if not isinstance(drop_original, bool):
            raise ValueError(
                f"'drop_original' is {drop_original}. The variable must "
                f"be boolean."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.periods = periods
        self.freq = freq
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame) -> None:
        """
        Creates lag

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
           The dataframe containing the variables that that will be lagged.

        Returns
        -------

        """
        # check if 'X' is a dataframe
        _is_dataframe(X)

        # check variables
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Creates lag

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing the variables that that will be lagged.

        Returns
        -------
        X_new: pandas dataframe
            The dataframe comprised of only the transformed variables or
            the original dataframe plus the transformed variables.

        """
        # check if 'X' is a dataframe
        _is_dataframe(X)

        tmp = X[self.variables_].shift(
            periods=self.periods,
            freq=self.freq,
            axis=0,
        )

        tmp.columns = self.rename_variables()
        X = X.merge(
            tmp, left_index=True, right_index=True, how="left"
        )

        if self.drop_original:
            X = X.drop(self.variables_, axis=1)

            return X

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
            lag_str = f"_lag_{self.periods}"
        else:
            lag_str = f"_lag_{self.freq}"

        variables_lag = [name + lag_str for name in self.variables_]

        return variables_lag
