from typing import List, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
)
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)

Variables = Union[None, int, str, List[Union[str, int]]]


class SelectBySingleFeaturePerformance(BaseEstimator, TransformerMixin):
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

    threshold : float, int, default = 0.5
        The value that defines if a feature will be kept or removed. Note that for
        metrics like roc-auc, r2_score and accuracy, the thresholds will be floats
        between 0 and 1. For metrics like the mean_square_error and the
        root_mean_square_error the threshold will be a big number.
        The threshold must be defined by the user.

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
        threshold: Union[int, float] = 0.5,
        variables: Variables = None,
    ):

        if not isinstance(cv, int) or cv < 1:
            raise ValueError("cv can only take positive integers bigger than 1")

        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold can only be integer or float")

        if scoring == "roc_auc" and (threshold < 0.5 or threshold > 1):
            raise ValueError(
                "roc-auc score should vary between 0.5 and 1. Pick a "
                "threshold within this interval."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find the important features.

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
        self.features_to_drop_ = []

        # train a model for every feature
        for feature in self.variables:
            model = cross_validate(
                self.estimator,
                X[feature].to_frame(),
                y,
                cv=self.cv,
                return_estimator=False,
                scoring=self.scoring,
            )

            if model["test_score"].mean() < self.threshold:
                self.features_to_drop_.append(feature)

            self.feature_performance_[feature] = model["test_score"].mean()

        # check we are not dropping all the columns in the df
        if len(self.features_to_drop_) == len(X.columns):
            raise ValueError(
                "All features will be dropped, try changing the threshold."
            )

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame):
        """
        Return dataframe with selected features.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features].
            The input dataframe from which feature values will be train.

        Returns
        -------
        X_transformed: pandas dataframe
            of shape = [n_samples, selected_features]
            Pandas dataframe with the selected features.
        """

        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = _is_dataframe(X)

        # check if number of columns in test dataset matches to train dataset
        _check_input_matches_training_df(X, self.input_shape_[1])

        return X.drop(columns=self.features_to_drop_)
