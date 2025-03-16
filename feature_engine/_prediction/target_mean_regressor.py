import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.utils.multiclass import type_of_target

from feature_engine._prediction.base_predictor import BaseTargetMeanEstimator


class TargetMeanRegressor(RegressorMixin, BaseTargetMeanEstimator):
    """
    The TargetMeanRegressor() outputs a target estimation based on the mean target
    value per category or bin, across a group of categorical or numerical variables.

    First, TargetMeanRegressor() calculates the mean target value per category or per
    bin for each variable. The final estimation is the average of the target mean
    values across variables.

    The TargetMeanRegressor() takes both numerical and categorical variables as input.
    For numerical variables, the values are first sorted into bins of equal-width or
    equal-frequency. Then, the mean target value is estimated for each bin. If the
    variables are categorical, the mean target value is estimated for each category.
    Finally, the estimator takes the average of the mean target value across all
    input variables.

    More details in the :ref:`User Guide <targetmeanregressor>`.


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

    binner_dict_:
         Dictionary with the interval limits per numerical variable.

    encoder_dict_:
        Dictionary with the mean target value per category or interval, per variable.

    n_features_in_:
        The number of features in the train set used in fit.

    feature_names_in_:
        List with the names of features seen during `fit`.

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
        Learn the mean target value per category or bin.

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
                "allowed by this transformer. "
            )

        return super().fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the average of the target mean value across variables.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, ]
            The input samples.

        Return
        -------
        y_pred: ndarray of shape (n_samples,)
            Returns predicted values.
        """
        return self._predict(X)
