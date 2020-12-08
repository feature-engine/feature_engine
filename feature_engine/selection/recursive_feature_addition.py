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
from feature_engine.selection.base_selector import get_feature_importances
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)

Variables = Union[None, int, str, List[Union[str, int]]]


class RecursiveFeatureAddition(BaseEstimator, TransformerMixin):
    """

    RecursiveFeatureAddition selects features. #WIP


    Model training and performance calculation are done with cross-validation.

    Parameters
    ----------

    variables : str or list, default=None
        The list of variable to be evaluated. If None, the transformer will evaluate
        all numerical features in the dataset.

    estimator: object, default = RandomForestClassifier()
        A Scikit-learn estimator for regression or classification.
        The estimator must have either a feature_importances or coef attribute
        after fitting.

    scoring: str, default='roc_auc'
        Desired metric to optimise the performance of the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    threshold: float, int, default = 0.01
        The value that defines if a feature will be kept or removed. Note that for
        metrics like roc-auc, r2_score and accuracy, the thresholds will be floats
        between 0 and 1. For metrics like the mean_square_error and the
        root_mean_square_error the threshold will be a big number.
        The threshold must be defined by the user. Bigger thresholds will retain less
        features.

    cv : int, default=3
        Cross-validation fold to be used to fit the estimator.

    Attributes
    ----------

    initial_model_performance_: float
        performance of the model built using the original dataset with all the features.

    feature_importances_: pandas series
        A pandas Series containing the feature names in the axis, and the performance
        derived from the model trained on the entire dataset, as values. The Series is
        ordered from least important to most important feature.

    performance_drifts_: dict
        A dictionary containing the feature, and the change in performance incurred
        when training a model without that feature.

    selected_features_: list
        The selected features by the recursive feature addition selector

    Methods
    -------

    fit: finds important features

    transform: removes non-important / non-selected features

    fit_transform: finds and removes non-important features

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

        Args
        ----

        X: pandas dataframe of shape = [n_samples, n_features]
           The input dataframe

        y: array-like of shape (n_samples)
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

        # Sort the feature importance values descreasingly
        self.feature_importances_.sort_values(ascending=False, inplace=True)

        # Extract most important feature from the ordered list of features
        first_most_important_feature = list(self.feature_importances_.index)[0]

        # Run baseline model using only the most important feature
        baseline_model = cross_validate(
            self.estimator,
            X[first_most_important_feature].to_frame(),
            y,
            cv=self.cv,
            scoring=self.scoring,
            return_estimator=True,
        )

        # Save baseline model performance
        baseline_model_performance = baseline_model["test_score"].mean()

        # list to collect selected features
        # It is initialized with the most important feature
        self.selected_features_ = [first_most_important_feature]

        # dict to collect features and their performance_drift
        # It is initialized with the performance drift of
        # the most important feature
        self.performance_drifts_ = {
            first_most_important_feature: baseline_model_performance -
            self.initial_model_performance_
        }

        # loop over the ordered list of features by feature importance starting
        # from the second element in the list.
        # Train the estimator with
        for feature in list(self.feature_importances_.index)[1:]:

            # Add feature and train new model
            model_tmp = cross_validate(
                self.estimator,
                X[self.selected_features_ + [feature]],
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_estimator=True,
            )

            # assign new model performance
            model_tmp_performance = model_tmp["test_score"].mean()

            # Calculate performance drift
            performance_drift = model_tmp_performance - baseline_model_performance

            # Save feature and performance drift
            self.performance_drifts_[feature] = performance_drift

            # If new performance model is
            if performance_drift > self.threshold:

                # add feature to the list of selected features
                self.selected_features_.append(feature)

                # Update new baseline model performance
                baseline_model_performance = model_tmp_performance

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame):
        """
        Removes non-selected features. That is, features which did not cause a big
        estimator performance increase when added to the dataset.

        Args
        ----

        X: pandas dataframe of shape = [n_samples, n_features].
            The input dataframe from which features will be selected.

        Returns
        -------

        X_transformed: pandas dataframe
            of shape = [n_samples, n_selected_features]
            Pandas dataframe with the selected features.
        """

        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = _is_dataframe(X)

        # check if number of columns in test dataset matches to train dataset
        _check_input_matches_training_df(X, self.input_shape_[1])

        # return the dataframe with the selected features
        return X[self.selected_features_]
