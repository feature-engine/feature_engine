import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import List, Optional, Union

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)


class BaseOutlier(BaseEstimator, TransformerMixin):
    """shared set-up checks and methods across outlier transformers"""

    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
        """Checks that the input is a dataframe and of the same size than the one used
        in the fit method. Checks absence of NA.

        Parameters
        ----------
        X: Pandas DataFrame

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            If the dataframe is not of same size as that used in fit()

        Returns
        -------
        X: Pandas DataFrame
            The same dataframe entered by the user.
        """
        # check if class was fitted
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # Check that the dataframe contains the same number of columns
        # than the dataframe used to fit the imputer.
        _check_input_matches_training_df(X, self.n_features_in_)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cap the variable values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the capped variables.
        """

        # check if class was fitted
        X = self._check_transform_input_and_state(X)

        # replace outliers
        for feature in self.right_tail_caps_.keys():
            X[feature] = np.where(
                X[feature] > self.right_tail_caps_[feature],
                self.right_tail_caps_[feature],
                X[feature],
            )

        for feature in self.left_tail_caps_.keys():
            X[feature] = np.where(
                X[feature] < self.left_tail_caps_[feature],
                self.left_tail_caps_[feature],
                X[feature],
            )

        return X

    def _more_tags(self):
        return _return_tags()


class WinsorizerBase(BaseOutlier):
    def __init__(
        self,
        capping_method: str = "gaussian",
        tail: str = "right",
        fold: Union[int, float] = 3,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
    ) -> None:

        if capping_method not in ["gaussian", "iqr", "quantiles"]:
            raise ValueError(
                "capping_method takes only values 'gaussian', 'iqr' or 'quantiles'"
            )

        if tail not in ["right", "left", "both"]:
            raise ValueError("tail takes only values 'right', 'left' or 'both'")

        if fold <= 0:
            raise ValueError("fold takes only positive numbers")

        if capping_method == "quantiles" and fold > 0.2:
            raise ValueError(
                "with capping_method ='quantiles', fold takes values between 0 and "
                "0.20 only."
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'")

        self.capping_method = capping_method
        self.tail = tail
        self.fold = fold
        self.variables = _check_input_parameter_variables(variables)
        self.missing_values = missing_values

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the values that should be used to replace outliers.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : pandas Series, default=None
            y is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        self.right_tail_caps_ = {}
        self.left_tail_caps_ = {}

        # estimate the end values
        if self.tail in ["right", "both"]:
            if self.capping_method == "gaussian":
                self.right_tail_caps_ = (
                    X[self.variables_].mean() + self.fold * X[self.variables_].std()
                ).to_dict()

            elif self.capping_method == "iqr":
                IQR = X[self.variables_].quantile(0.75) - X[self.variables_].quantile(
                    0.25
                )
                self.right_tail_caps_ = (
                    X[self.variables_].quantile(0.75) + (IQR * self.fold)
                ).to_dict()

            elif self.capping_method == "quantiles":
                self.right_tail_caps_ = (
                    X[self.variables_].quantile(1 - self.fold).to_dict()
                )

        if self.tail in ["left", "both"]:
            if self.capping_method == "gaussian":
                self.left_tail_caps_ = (
                    X[self.variables_].mean() - self.fold * X[self.variables_].std()
                ).to_dict()

            elif self.capping_method == "iqr":
                IQR = X[self.variables_].quantile(0.75) - X[self.variables_].quantile(
                    0.25
                )
                self.left_tail_caps_ = (
                    X[self.variables_].quantile(0.25) - (IQR * self.fold)
                ).to_dict()

            elif self.capping_method == "quantiles":
                self.left_tail_caps_ = X[self.variables_].quantile(self.fold).to_dict()

        self.n_features_in_ = X.shape[1]

        return self
