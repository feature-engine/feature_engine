# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _check_input_matches_training_df,
    _is_dataframe,
)
from feature_engine.prediction.base_predictor import BaseTargetMeanEstimator


class TargetMeanRegressor(BaseTargetMeanEstimator, RegressorMixin):
    """
    The TargetMeanRegressor outputs a target estimation based on the mean target value
    per category or bin, across a group of variables.

    First, it calculates the mean target value per category or per bin for each
    variable. The final estimation, is the average of the target mean values across
    variables.

    The TargetMeanRegressor takes both numerical and categorical variables as input. If
    variables are numerical, the values are first sorted into bins of equal-width or
    equal-frequency. Then, the mean target value is estimated for each bin. If the
    variable is categorical, the mean target value is estimated for each category.

    Parameters
    ----------
    variables: list, default=None
        The list of input variables. If None, the estimator will use all variables as
        input features (except datetime).

    bins: int, default=5
        If the dataset contains numerical variables, the number of bins into which
        the values will be sorted.

    strategy: str, default='equal_width'
        Whether the bins should of equal width ('equal_width') or equal frequency
        ('equal_frequency').

    Attributes
    ----------
    variables_categorical_:
        The group of categorical input variables that will be used for prediction.

    variables_numerical_:
        The group of numerical input variables that will be used for prediction.

    pipeline_:
        A Sickit-learn Pipeline with a dicretiser and/or encoder. Used to determine the
        mean target value per category or bin, per variable.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Learn the mean target value per category or per bin, for each variable.
    predict:
        Predict using the average of the target mean value across variables.
    score:
        Return the coefficient of determination of the prediction.

    See Also
    --------
    feature_engine.encoding.MeanEncoder
    feature_engine.discretisation.EqualWidthDiscretiser
    feature_engine.discretisation.EqualFrequencyDiscretiser

    References
    ----------
    Adapted from:

    .. [1] Miller, et al. "Predicting customer behaviour: The University of Melbourne’s
        KDD Cup report". JMLR Workshop and Conference Proceeding. KDD 2009
        http://proceedings.mlr.press/v7/miller09/miller09.pdf
    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn mean target value per category or bin.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : pandas series of shape = [n_samples,]
            The target variable.
        """

        if type_of_target(y) == "binary":
            raise ValueError(
                "Trying to fit a regression to a binary target is not "
                "allowed by this transformer. Check the target values "
                "or set regression to False."
            )

        super().fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict using the average of the target mean value across variables.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, ]
            The input samples.

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

        # transform dataframe
        X_tr = self.pipeline.transform(X)

        # calculate the average for each observation
        predictions = X_tr[
            self.variables_numerical_ + self.variables_categorical_
        ].mean(axis=1)

        return predictions
