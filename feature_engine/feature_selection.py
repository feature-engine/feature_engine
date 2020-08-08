# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.variable_manipulation import _define_variables


class FeatureEliminator(BaseEstimator, TransformerMixin):
    """
    The FeatureEliminator() drops the list of variable(s) as provided by the user
    from the dataframe and returns the subset of original dataframe with remaining
    variables.

    Parameters
    ----------

    features_to_drop : str or list, default=None
        Desired variable/s to be dropped from the dataframe

    """

    def __init__(self, features_to_drop=None):

        if not any(isinstance(features_to_drop, t) for t in [str, list]):
            raise ValueError("features_to_drop must be a string or list object")

        self.features = _define_variables(features_to_drop)

    def fit(self, X, y=None):
        """
        Verifies that the passed input X if of the type pandas dataframe

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe on which the feature elimination has to be performed
        y: None
            y is not needed for this transformer

        """
        # check input dataframe
        X = _is_dataframe(X)
        return self

    def transform(self, X):
        """
        Drops the variable or list of variables provided from the original dataframe
        and returns the dataframe with subset of variables.

        Parameters
        ----------
        X: pandas dataframe
            The input dataframe on which the feature elimination has to be performed

        Returns
        -------
        X_transformed: pandas dataframe of shape = [n_samples, n_features - len(features_to_drop)]
            The transformed dataframe with subset of variables.

        """
        return X.drop(columns=self.features).copy()


