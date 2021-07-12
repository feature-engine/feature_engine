from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _is_dataframe,
)


class SimilarColumns(BaseEstimator, TransformerMixin):
    """Ensure that similar columns are in test and train dataset.


    Parameters
    ----
    None

    """

    def __init__(self, impute_with: Any = np.nan, missing_values: str = "ignore"):
        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'.")

        self.impute_with = impute_with
        self.missing_values = missing_values

    def _check_input(self, X: pd.DataFrame):
        X = _is_dataframe(X)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        return X

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit columns schema

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe

        y: None
            y is not needed for this transformer. You can pass y or None.

        """
        X = self._check_input(X)

        self.variables_ = X.columns
        return self

    def transform(self, X: pd.DataFrame, **transform_params) -> pd.DataFrame:
        """Drops the variable that are not in the fitted dataframe and returns
        a new dataframe with the remaining subset of variables.

        If a column is in train but not in test, then the column will be created in
        test dataset with missing value everywhere.

        If a column is in test but not in train, it will be dropped.

        Parameters
        ----------
        X: pandas dataframe
            The input dataframe from which features will be dropped

        Returns
        -------
        X_transformed: pandas dataframe of shape =
             [n_samples, n_features - len(features_to_drop)]

                The transformed dataframe with the same columns
                (in same order) than the fitted dataframe

        """
        check_is_fitted(self)

        X = self._check_input(X)

        for col in self.variables_:
            if col not in X.columns:
                X[col] = self.impute_with
        # reorder columns
        X = X.loc[:, self.variables_]
        return X
