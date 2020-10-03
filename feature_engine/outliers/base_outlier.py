import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
    _check_contains_na,
)


class BaseOutlier(BaseEstimator, TransformerMixin):

    def _check_transform_input_and_state(self, X):
        # check if class was fitted
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        if self.missing_values == 'raise':
            # check if dataset contains na
            _check_contains_na(X, self.variables)

        # Check that the dataframe contains the same number of columns
        # than the dataframe used to fit the imputer.
        _check_input_matches_training_df(X, self.input_shape_[1])

        return X

    def transform(self, X):
        """
        Caps the variable values, that is, censors outliers.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the capped variables.
        """

        # check if class was fitted
        X = self._check_transform_input_and_state(X)

        # replace outliers
        for feature in self.right_tail_caps_.keys():
            X[feature] = np.where(X[feature] > self.right_tail_caps_[feature],
                                  self.right_tail_caps_[feature], X[feature])

        for feature in self.left_tail_caps_.keys():
            X[feature] = np.where(X[feature] < self.left_tail_caps_[feature],
                                  self.left_tail_caps_[feature], X[feature])

        return X
