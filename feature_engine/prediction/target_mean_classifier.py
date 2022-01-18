# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Union

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_input_matches_training_df,
    _is_dataframe,
)

from feature_engine.prediction.base_predictor import BaseTargetMeanEstimator


class TargetMeanClassifier(BaseTargetMeanEstimator, ClassifierMixin):
    """

    Parameters
    ----------
    variables: list, default=None
        The list of input variables. If None, the estimator will evaluate will use all
        variables as input fetures.

    bins: int, default=5
        If the dataset contains numerical variables, the number of bins into which
        the values will be sorted.

    strategy: str, default='equal_width'
        Whether the bins should of equal width ('equal_width') or equal frequency
        ('equal_frequency').

    Attributes
    ----------
    variables_:
    The group of variables that will be transformed.

    pipeline:
        An assembly of a dicretiser and/or encoder that transforms the data.

    Methods
    -------
    predict:
        Returns the mean of the labels of the corresponding (discretised) bin
        or category.

    predict_proba:


    Notes
    -----


    See Also
    --------


    References
    ----------


    """

    def __init__(
            self,
            variables: Union[None, int, str, List[Union[str, int]]] = None,
            bins: int = 5,
            strategy: str = "equal_width",
    ):

        BaseTargetMeanEstimator.__init__(
            self,
            variables,
            bins,
            strategy,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, ]
            The input series which must have the same name as one of the features in the
            dataframe that was used to fit the predictor.

        Return
        -------
        predictions: pandas series of shape = [n_samples, ]
            Values are the mean values associated with the corresponding encoded or
            discretised bin.

        """
        # check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        _is_dataframe(X)

        # Check input data contains same number of columns as df used to fit
        _check_input_matches_training_df(X, self.n_features_in_)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, ]
            The input series which must have the same name as one of the features in the
            dataframe that was used to fit the predictor.

        Return
        -------
        prob_predictions: pandas series of shape = [n_samples, ]


        """
        pass
