import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)


class BaseOutlier(BaseEstimator, TransformerMixin):
    """
    Base class for outliers module.
    Other classes inherits from it.

    Methods:
        transform(): Apply the transformation to the data.
    """

    def _check_transform_input_and_state(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Checks if class was fitted along with DataFrame type existence check.
        Then checks if there are NA values and lastly checks if the DataFrame
        contains the same number of columns than the dataframe used to fit the imputer.

        Args:
            X: Pandas dataframe of shape = [n_samples, n_features]

        Returns:
            Pandas DataFrame after performing all the required checks.

        """

        check_is_fitted(self)

        X = _is_dataframe(X)

        if self.missing_values == "raise":
            _check_contains_na(X, self.variables)

        _check_input_matches_training_df(X, self.input_shape_[1])

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Caps the variable values, that is, censors outliers.

        Args:
            X: Pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns:
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
