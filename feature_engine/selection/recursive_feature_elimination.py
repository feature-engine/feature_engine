from typing import List, Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.selection.base_selector import get_feature_importances
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)
from feature_engine.selection.base_selector import BaseSelector

Variables = Union[None, int, str, List[Union[str, int]]]


class RecursiveFeatureElimination(BaseSelector):
    """
    RecursiveFeatureElimination selects features following a recursive process.

    The process is as follows:

    1. Train an estimator using all the features.

    2. Rank the features according to their importance, derived from the estimator.

    3. Remove one feature -the least important- and fit a new estimator with the
    remaining features.

    4. Calculate the performance of the new estimator.

    5. Calculate the difference in performance between the new and the original
    estimator.

    6. If the performance drops beyond the threshold, then that feature is important
    and will be kept. Otherwise, that feature is removed.

    7. Repeat steps 3-6 until all features have been evaluated.

    Model training and performance calculation are done with cross-validation.

    Parameters
    ----------
    variables : str or list, default=None
        The list of variable to be evaluated. If None, the transformer will evaluate
        all numerical features in the dataset.

    estimator : object, default = RandomForestClassifier()
        A Scikit-learn estimator for regression or classification.
        The estimator must have either a `feature_importances` or `coef_` attribute
        after fitting.

    scoring : str, default='roc_auc'
        Desired metric to optimise the performance of the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    threshold : float, int, default = 0.01
        The value that defines if a feature will be kept or removed. Note that for
        metrics like roc-auc, r2_score and accuracy, the thresholds will be floats
        between 0 and 1. For metrics like the mean_square_error and the
        root_mean_square_error the threshold will be a big number.
        The threshold must be defined by the user. Bigger thresholds will select less
        features.

    cv : int, default=3
        Cross-validation fold to be used to fit the estimator.

    Attributes
    ----------
    initial_model_performance_ :
        Performance of the model trained using the original dataset.

    feature_importances_ :
        Pandas Series with the feature importance

    performance_drifts_:
        Dictionary with the performance drift per examined feature.

    features_to_drop_:
        List with the features to remove from the dataset.

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
        threshold: Union[int, float] = 0.01,
        variables: Variables = None,
    ):

        if not isinstance(cv, int) or cv < 1:
            raise ValueError("cv can only take positive integers bigger than 1")

        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold can only be integer or float")

        self.variables = _check_input_parameter_variables(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Find the important features. Note that the selector trains various models at
        each round of selection, so it might take a while.

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

        # train model with all features and cross-validation
        model = cross_validate(
            self.estimator,
            X[self.variables],
            y,
            cv=self.cv,
            scoring=self.scoring,
            return_estimator=True,
        )

        # store initial model performance
        self.initial_model_performance_ = model["test_score"].mean()

        # Initialize a dataframe that will contain the list of the feature/coeff
        # importance for each cross validation fold
        feature_importances_cv = pd.DataFrame()

        # Populate the feature_importances_cv dataframe with columns containing
        # the feature importance values for each model returned by the cross
        # validation.
        # There are as many columns as folds.
        for m in model["estimator"]:

            feature_importances_cv[m] = get_feature_importances(m)

        # Add the variables as index to feature_importances_cv
        feature_importances_cv.index = self.variables

        # Aggregate the feature importance returned in each fold
        self.feature_importances_ = feature_importances_cv.mean(axis=1)

        # Sort the feature importance values
        self.feature_importances_.sort_values(ascending=True, inplace=True)

        # to collect selected features
        _selected_features = []

        # temporary copy where we will remove features recursively
        X_tmp = X[self.variables].copy()

        # we need to update the performance as we remove features
        baseline_model_performance = self.initial_model_performance_

        # dict to collect features and their performance_drift after shuffling
        self.performance_drifts_ = {}

        # evaluate every feature, starting from the least important
        # remember that feature_importances_ is ordered already
        for feature in list(self.feature_importances_.index):

            # remove feature and train new model
            model_tmp = cross_validate(
                self.estimator,
                X_tmp.drop(columns=feature),
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_estimator=False,
            )

            # assign new model performance
            model_tmp_performance = model_tmp["test_score"].mean()

            # Calculate performance drift
            performance_drift = baseline_model_performance - model_tmp_performance

            # Save feature and performance drift
            self.performance_drifts_[feature] = performance_drift

            if performance_drift > self.threshold:

                _selected_features.append(feature)

            else:
                # remove feature and adjust initial performance
                X_tmp = X_tmp.drop(columns=feature)

                baseline_model = cross_validate(
                    self.estimator,
                    X_tmp,
                    y,
                    cv=self.cv,
                    return_estimator=False,
                    scoring=self.scoring,
                )

                # store initial model performance
                baseline_model_performance = baseline_model["test_score"].mean()

        self.features_to_drop_ = [
            f for f in self.variables if f not in _selected_features
        ]

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseSelector.transform.__doc__
