# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine.dataframe_checks import (
    _is_dataframe,
)
from feature_engine.variable_manipulation import (
    _find_or_check_numerical_variables,
    _check_input_parameter_variables,
)
from feature_engine.docstrings import (
    Substitution,
    _variables_numerical_docstring,
    _drop_original_docstring,
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _n_features_in_docstring,
)

@Substitution(
    variables = _variables_numerical_docstring,
    drop_original = _drop_original_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class LagFeatures(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    {variables}

    periods: int, default=1
        Number of periods to shift. Can be positive or negative.

    freq: str, default=None
        Offset to use from the tseries module or time rule. See parameter `freq` in
        `pandas.shift`. It is the same functionality.

    {drop_original}

    Attributes
    ----------
    variables_:
        The group of variables that will be lagged.

    {n_features_in_}

    Methods
    -------
    {fit}

    transform:
        Add the lagged features.

    {fit_transform}

    Notes
    -----


    See Also
    --------
    pandas.shift()

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
        # Prevents True and False passing as 1 and 0.
        if isinstance(periods, bool):
            raise ValueError(
                f"'periods' is {periods}. The variable must be "
                f"an integer."
            )
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
        Identifies the numerical variables to be transformed.

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

        tmp.columns = self._rename_variables()
        X = X.merge(
            tmp, left_index=True, right_index=True, how="left"
        )

        if self.drop_original:
            X = X.drop(self.variables_, axis=1)

        return X

    def _rename_variables(self):
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

        variables_lag = [str(name) + lag_str for name in self.variables_]

        return variables_lag
