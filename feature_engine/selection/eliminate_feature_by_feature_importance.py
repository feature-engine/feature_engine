# WORK IN PROGRESS
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
    _define_variables,
    _find_numerical_variables,
)


def get_feature_importances(estimator, norm_order=1):
    """Retrieve feature importances from a fitted estimator"""

    importances = getattr(estimator, "feature_importances_", None)

    coef_ = getattr(estimator, "coef_", None)

    if importances is None and coef_ is not None:
        importances = np.abs(coef_)

    return list(importances)


class RecursiveFeatureElimination(BaseEstimator, TransformerMixin):
    """

    Model training and performance calculation are done with cross-validation.

    Parameters
    ----------

    variables : str or list, default=None
        The list of variable(s) to be shuffled from the dataframe.
        If None, the transformer will shuffle all numerical variables in the dataset.

    estimator: object, default = RandomForestClassifier()
        A Scikit-learn estimator for regression or classification.
        The estimator must have either a feature_importances_ or coef_ attribute
        after fitting.

    scoring: str, default='roc_auc'
        Desired metric to optimise the performance for the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    threshold: float, int, default = 0.01
        The value that defines if a feature will be kept or removed. Note that for
        metrics like roc-auc, r2_score and accuracy, the thresholds will be floats
        between 0 and 1. For metrics like the mean_square_error and the
        root_mean_square_error the threshold will be a big number.
        The threshold must be defined by the user.

    cv : int, default=3
        Desired number of cross-validation fold to be used to fit the estimator.

    Attributes
    ----------


    Methods
    -------

    fit: finds important features

    transform: removes non-important / non-selected features

    fit_transform: finds and removes non-important features

    """

    def __init__(
        self,
        estimator=RandomForestClassifier(),
        scoring="roc_auc",
        cv=3,
        threshold=0.01,
        variables=None
    ):

        if not isinstance(cv, int) or cv < 1:
            raise ValueError("cv can only take positive integers bigger than 1")

        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold can only be integer or float")

        self.variables = _define_variables(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv

    def fit(self, X, y):
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
        self.variables = _find_numerical_variables(X, self.variables)

        # train model with all features and cross-validation
        model = cross_validate(
            self.estimator,
            X,
            y,
            cv=self.cv,
            scoring=self.scoring,
            return_estimator=True
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

            features_importance_ls = get_feature_importances(m)
            feature_importances_cv[m] = features_importance_ls

        # Add the X variables as index to feature_importances_cv
        feature_importances_cv.index = self.variables

        # Apply absolute value function to entire feature_importances_cv dataframe.
        # This is done specificially for the linear estimators since large negative
        # coefficients signify important features.
        feature_importances_cv = feature_importances_cv.abs()

        # Aggregated the feature importance returned in each fold by applying mean
        feature_importances_agg = feature_importances_cv.mean(axis=1)

        # Sort the feature importance values
        feature_importances_agg.sort_values(ascending=True, inplace=True)

        # Store the feature importance series in a attribute
        self.feature_importances_ = feature_importances_agg

        # Extract the ordered feature list by importance and store it in
        # the attribute self.ordered_features_by_importance_
        self.ordered_features_by_importance_ = list(feature_importances_agg.index)
        # list to collect selected features
        self.selected_features_ = []

        X_tmp = X.copy()

        baseline_model_performance = self.initial_model_performance_

        # dict to collect features and their performance_drift after shuffling
        self.performance_drifts_ = {}

        for feature in self.ordered_features_by_importance_:

            # train model with new feature list
            model_tmp = cross_validate(
                self.estimator,
                X_tmp.drop(columns=feature),
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_estimator=True
            )

            # store new model performance
            model_tmp_performance = model_tmp["test_score"].mean()

            # Calculate performance drift
            performance_drift = baseline_model_performance - model_tmp_performance

            # Save feature and performance drift
            self.performance_drifts_[feature] = performance_drift

            if performance_drift > self.threshold:

                self.selected_features_.append(feature)

            else:

                X_tmp = X_tmp.drop(columns=feature)
                baseline_model = cross_validate(
                    self.estimator,
                    X_tmp,
                    y,
                    cv=self.cv,
                    return_estimator=True,
                    scoring=self.scoring
                )

                # store initial model performance
                baseline_model_performance = baseline_model["test_score"].mean()

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """

        Args
        ----

        X: pandas dataframe of shape = [n_samples, n_features].
            The input dataframe from which feature values will be shuffled.


        Returns
        -------

        X_transformed: pandas dataframe
            of shape = [n_samples, n_features - len(dropped features)]
            Pandas dataframe with the selected features.
        """

        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = _is_dataframe(X)

        # check if number of columns in test dataset matches to train dataset
        _check_input_matches_training_df(X, self.input_shape_[1])

        return X[self.selected_features_]
