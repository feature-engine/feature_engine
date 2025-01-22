import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, unique_labels

from feature_engine._prediction.base_predictor import BaseTargetMeanEstimator


class TargetMeanClassifier(ClassifierMixin, BaseTargetMeanEstimator):
    """
    The TargetMeanClassifier() estimates target values based on the average of the mean
    target value per category or bin of a group of categorical and numerical variables.

    The TargetMeanClassifier() first sorts numerical variables into bins of equal-width
    or equal-frequency. Then, the mean target value is estimated for each bin. If the
    variables are categorical, the mean target value is estimated for each category.

    Finally, the estimator takes the average of the mean target value per observation
    across the input variables. This average of the target value per observation is
    used as proxy for probability estimates. The class is then determined based on a
    threshold of 0.5.

    TargetMeanClassifier() works like any Scikit-learn classifier. At the moment, it
    only works for binary classification.

    More details in the :ref:`User Guide <targetmeanclassifier>`.

    Parameters
    ----------
    variables: list, default=None
        The list of input variables. If None, the estimator will use all variables as
        input features (except datetime).

    bins: int, default=5
        If the dataset contains numerical variables, the number of bins into which
        the values will be sorted.

    strategy: str, default='equal_width'
        Whether the bins should be of equal width ('equal_width') or equal frequency
        ('equal_frequency').

    Attributes
    ----------
    variables_categorical_:
        The group of categorical input variables that will be used for prediction.

    variables_numerical_:
        The group of numerical input variables that will be used for prediction.

    classes_:
        A list of class labels known to the classifier.

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
        Learn the mean target value per category or per bin, per variable.

    predict:
        Predict class labels for samples in X.

    predict_log_proba:
        Predict logarithm of probability estimates.

    predict_proba:
        Proxy for probability estimates based of the average of the target mean value
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
        check_classification_targets(y)

        self.classes_ = unique_labels(y)

        # check that y is binary
        if len(self.classes_) > 2:
            raise NotImplementedError(
                "This classifier is designed for binary classification only. "
                "The target has more than 2 unique values."
            )

        # if target has values other than 0 and 1, we need to remap the values,
        # to be able to compute meaningful averages.
        if any(x for x in self.classes_ if x not in [0, 1]):
            y = np.where(y == unique_labels(y)[0], 0, 1)

        return super().fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for X.

        Proxy for probability estimates based of the average of the target mean value
        across variables.

        The returned estimates for all classes are ordered by the label of classes.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Return
        -------
        p: array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model, where
            classes are ordered as they are in self.classes_.
        """
        prob = self._predict(X)
        return np.vstack([1 - prob, prob]).T

    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class log-probabilities for X.

        The returned estimates for all classes are ordered by the label of classes.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Return
        -------
        p: array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the model,
            where classes are ordered as they are in self.classes_.
        """
        return np.log(self.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class for X.

        Class 1 is returned when the class probability is bigger than 0.5.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Return
        -------
        y_pred: ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        y_pred = np.where(self._predict(X) > 0.5, self.classes_[1], self.classes_[0])
        return y_pred

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
