# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Union, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_input_matches_training_df,
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
    _missing_values_docstring,
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _n_features_in_docstring,
)


@Substitution(
    variables=_variables_numerical_docstring,
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class LagFeatures(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    {variables}

    periods: int, list of ints, default=1
        Number of periods to shift. Can be positive or negative. If list, features will
        be created for each one of the periods.

    freq: str, list of str, default=None
        Offset to use from the tseries module or time rule. See parameter `freq` in
        `pandas.shift`. It is the same functionality. If list, features will
        be created for each one of the frequency values in the list.

    {missing_values}

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
            missing_values: str = 'raise',
            drop_original: bool = False,

    ) -> None:
        # Prevents True and False passing as 1 and 0.
        if not isinstance(periods, (int, list)):
            raise ValueError(
                f"`periods` must be an integer or a list of integers. Got {periods} "
                f"instead."
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'."
                             f"Got {missing_values} instead.")

        if not isinstance(drop_original, bool):
            raise ValueError(
                "drop_original takes only boolean values True and False. "
                f"Got {drop_original} instead."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.periods = periods
        self.freq = freq
        self.missing_values = missing_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas Series, default=None
            y is not needed in this imputation. You can pass None or y.
        """
        # check if 'X' is a dataframe
        _is_dataframe(X)

        # check variables
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds lag features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe, shape = [n_samples, n_features + n_operations]
            The dataframe with the original variables plus the new variables.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check if 'X' is a dataframe
        _is_dataframe(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables_to_combine)
            _check_contains_inf(X, self.variables_to_combine)


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
