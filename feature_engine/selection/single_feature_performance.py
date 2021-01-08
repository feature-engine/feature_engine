from typing import List, Union
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)
from feature_engine.selection.base_selector import BaseSelector

Variables = Union[None, int, str, List[Union[str, int]]]


class SelectBySingleFeaturePerformance(BaseSelector):
    """
    SelectBySingleFeaturePerformance() selects features based on the performance
    obtained from a machine learning model trained utilising a single feature. In other
    words, it trains a machine learning model for every single feature, utilising that
    individual feature, then determines each model performance. If the performance of
    the model based on the single feature is greater than a user specified threshold,
    then the feature is retained, otherwise removed.

    The models are trained on the individual features using cross-validation.
    The performance metric to evaluate and the machine learning model to train are
    specified by the user.

    Parameters
    ----------
    variables : str or list, default=None
        The list of variable(s) to be evaluated.
        If None, the transformer will evaluate all numerical variables in the dataset.

    estimator : object, default = RandomForestClassifier()
        A Scikit-learn estimator for regression or classification.

    scoring : str, default='roc_auc'
        Desired metric to optimise the performance for the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    threshold : float, int, default = None
        The value that defines if a feature will be kept or removed.

        The r2varies between 0 and 1. So a threshold needs to be set-up within
        these boundaries.

        The roc-auc varies between 0.5 and 1. So a threshold needs to be set-up within
        these boundaries.

        For metrics like the mean_square_error and the root_mean_square_error the
        threshold will be a big number.

        The threshold can be specified by the user. If None, it will be automatically
        set to the mean performance value of all features.

    cv : int, default=3
        Desired number of cross-validation fold to be used to fit the estimator.

    Attributes
    ----------
    features_to_drop_:
        List with the features to remove from the dataset.

    feature_performance_:
        Dictionary with the single feature model performance per feature.

    Methods
    -------
    fit:
        Find the important features.
    transform:
        Reduce X to the selected features.
    fit_transform:
        Fit to data, then transform it.
    """

    def __init__(
        self,
        estimator=RandomForestClassifier(),
        scoring: str = "roc_auc",
        cv: int = 3,
        threshold: Union[int, float] = None,
        variables: Variables = None,
    ):

        if not isinstance(cv, int) or cv < 1:
            raise ValueError("cv can only take positive integers bigger than 1")

        if threshold:
            if not isinstance(threshold, (int, float)):
                raise ValueError("threshold can only be integer, float or None")

            if scoring == "roc_auc" and (threshold < 0.5 or threshold > 1):
                raise ValueError(
                    "roc-auc score should vary between 0.5 and 1. Pick a "
                    "threshold within this interval."
                )

            if scoring == "r2" and (threshold < 0 or threshold > 1):
                raise ValueError(
                    "r2 takes values between -1 and 1. To select features the "
                    "transformer considers the absolute value. Pick a threshold within "
                    "0 and 1."
                )

        self.variables = _check_input_parameter_variables(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Select features.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
           The input dataframe

        y : array-like of shape (n_samples)
           Target variable. Required to train the estimator.

        Returns
        -------
        self
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find numerical variables or check variables entered by user
        self.variables = _find_or_check_numerical_variables(X, self.variables)

        self.feature_performance_ = {}

        # train a model for every feature and store the performance
        for feature in self.variables:
            model = cross_validate(
                self.estimator,
                X[feature].to_frame(),
                y,
                cv=self.cv,
                return_estimator=False,
                scoring=self.scoring,
            )

            self.feature_performance_[feature] = model["test_score"].mean()

        # select features
        if not self.threshold:
            threshold = pd.Series(self.feature_performance_).mean()
        else:
            threshold = self.threshold

        self.features_to_drop_ = [
            f
            for f in self.feature_performance_.keys()
            if self.feature_performance_[f] < threshold
            ]

        # check we are not dropping all the columns in the df
        if len(self.features_to_drop_) == len(X.columns):
            warnings.warn("All features will be dropped, try changing the threshold.")

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseSelector.transform.__doc__
