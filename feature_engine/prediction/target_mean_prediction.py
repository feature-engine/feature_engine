# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.discretisation import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser
)
from feature_engine.encoding import MeanEncoder

class TargetMeanPredictor(ClassifierMixin, RegressorMixin):
    """

    Parameters
    ----------


    Attributes
    ----------


    Methods
    -------


    Notes
    -----


    See Also
    --------


    References
    ----------


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

        y : pandas series of shape = [n_samples,]
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