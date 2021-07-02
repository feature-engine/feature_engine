from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class SimilarColumns(BaseEstimator, TransformerMixin):
    """Ensure that similar columns are in test and train dataset.


    Parameters
    ----
    None

    """

    def __init__(self, impute_with: Any = np.nan):
        self.impute_with = impute_with
        self.col = None

    def fit(self, df: pd.DataFrame, y: pd.Series = None, **fit_params):
        """Fit columns schema

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe

        y: None
            y is not needed for this transformer. You can pass y or None.

        """

        self.col = df.columns
        return self

    def transform(self, X: pd.DataFrame, **transform_params) -> pd.DataFrame:
        """Drops the variable that are not in the fitted dataframe and returns a new dataframe with the remaining subset of variables.

        If a column is in train but not in test, then the column will be created in
        test dataset with missing value everywhere.

        If a column is in test but not in train, it will be dropped.

        Parameters
        ----------
        X: pandas dataframe
            The input dataframe from which features will be dropped

        Returns
        -------
        X_transformed: pandas dataframe of shape = [n_samples, n_features - len(features_to_drop)]
            The transformed dataframe with the same columns (in same order) than the fitted dataframe

        """

        for col in self.col:
            if col not in X.columns:
                X[col] = self.impute_with
        # reorder columns
        X = X.loc[:, self.col]
        return X
