from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.variable_manipulation import _find_or_check_numerical_variables


class BaseCreation(BaseEstimator, TransformerMixin):
    """Shared set-up, checks and methods across creation transformers."""

    _transform_docstring = """transform:
            Create new features.
        """.rstrip()

    def __init__(
        self,
        missing_values: str = "raise",
        drop_original: bool = False,
    ) -> None:

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        if not isinstance(drop_original, bool):
            raise TypeError(
                "drop_original takes only boolean values True and False. "
                f"Got {drop_original} instead."
            )

        self.missing_values = missing_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Common set-up of creation transformers.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: pandas Series, or np.array. Defaults to None.
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = _is_dataframe(X)

        # check variables are numerical
        self.variables: List[Union[str, int]] = _find_or_check_numerical_variables(X, self.variables)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables)
            _check_contains_inf(X, self.variables)

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Common input and transformer checks.

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

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_input_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na
        if self.missing_values == "raise":
            _check_contains_na(X, self.variables)
            _check_contains_inf(X, self.variables)

        # reorder variables to match train set
        X = X[self.feature_names_in_]

        return X
