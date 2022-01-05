# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TargetMeanPredictor(BaseEstimator):
    """


    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit predictor per variables.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : pandas series.
            The target variable.
        """
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The inputs uses to derive the predictions.

        Return
        -------
        y : pandas series of (n_samples,)
            Mean target values.

        """
        pass