# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause
import warnings

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from feature_engine.prediction.base_predictor import BaseTargetMeanEstimator


class TargetMeanClassifier(BaseTargetMeanEstimator, ClassifierMixin):
    """
    The TargetMeanClassifier estimates the target value based on the mean target value
    per category or bin, across a group of variables.

    The TargetMeanClassifier takes both numerical and categorical variables as input.
    For numerical variables, the values are first sorted into bins of equal-width or
    equal-frequency. Then, the mean target value is estimated for each bin. If the
    variables are categorical, the mean target value is estimated for each category.
    Finally, the estimator takes the average of the mean target value across the
    input variables and determines the class based on a threshold of 0.5.

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

    classes_:
        A list of class labels known to the classifier.

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
        Predict class labels for samples in X.

    predict_log_proba:
        Predict logarithm of probability estimates.

    predict_proba:
        Proxy for Probability estimates based of the average of the target mean value
         across variables.

    score:
        Return the mean accuracy on the given test data and labels.

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
        # check that y is binary
        self.classes_ = list(y.unique())

        if len(self.classes_) > 2:
            raise ValueError(
                "This encoder is designed for binary classification only. The target "
                "has more than 2 unique values."
            )

        # if target has values other than 0 and 1, we need to remap the values,
        # to be able to compute meaningful averages.
        if any(x for x in y.unique() if x not in [0, 1]):
            y = np.where(y == y.unique()[0], 0, 1)

        super().fit(X, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Proxy for probability estimates based of the average of the target mean value
        across variables.

        The returned estimates for all classes are ordered by the label of classes.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Return
        -------
        T: array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model, where
            classes are ordered as they are in self.classes_.
        """
        prob = super()._predict(X)

        # TODO: check that this outputs a numpy array with 2 columns and
        # the second column is the prob of class 1.
        return np.vstack([1 - prob, prob]).T

    def predict_log_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the label of classes.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Return
        -------
        T: array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the model,
            where classes are ordered as they are in self.classes_..

        """
        # TODO: check that this outputs a numpy array with 2 columns
        return np.log(self.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input series which must have the same name as one of the features in the
            dataframe that was used to fit the predictor.

        Return
        -------
        y_pred: ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        # TODO: check that this is a numpy array
        return np.where(self.predict_proba(X) > 0.5, 1, 0)
