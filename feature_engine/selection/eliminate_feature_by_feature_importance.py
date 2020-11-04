import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.utils.validation import check_is_fitted

from feature_engine.selection.base_selector import get_feature_importances
from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
)
from feature_engine.variable_manipulation import (
    _define_variables,
    _find_numerical_variables,
)


class RecursiveFeatureElimination(BaseEstimator, TransformerMixin):
    """
    
    RecursiveFeatureElimination selects features following a recursive process.
    
    The process is as follow:
    
    1) Rank the features according to their importance derived from the estimator.

    2) Remove one feature -the least important- and fit the estimator again 
    utilising the remaining features.

    3) Calculate the performance of the estimator.

    4) If the estimator performance drops beyond the indicated threshold, then
    that feature is important and should be kept.
    Otherwise, that feature is removed.

    5) Repeat steps 2-4 until all features have been evaluated.
    
    
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
    
    initial_model_performance_: float
        performance of the model built using the original dataset.

    feature_importances_: pandas series
        The index contains feature while values represent the feature importances.
        The series are ordered from least importance to most important feature.

    performance_drifts_: dict
        A dictionary containing the feature, performance drift pairs, after
        the recursive feature elimination.


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

            feature_importances_cv[m] = get_feature_importances(m)

        # Add the X variables as index to feature_importances_cv
        feature_importances_cv.index = self.variables

        # Aggregated the feature importance returned in each fold by applying mean
        feature_importances_agg = feature_importances_cv.mean(axis=1)

        # Sort the feature importance values
        feature_importances_agg.sort_values(ascending=True, inplace=True)

        # Store the feature importance series in a attribute
        self.feature_importances_ = feature_importances_agg

        # Extract the ordered feature list by importance and store it
        ordered_features_by_importance_ = list(feature_importances_agg.index)
        # list to collect selected features
        self.selected_features_ = []

        X_tmp = X.copy()

        baseline_model_performance = self.initial_model_performance_

        # dict to collect features and their performance_drift after shuffling
        self.performance_drifts_ = {}

        for feature in ordered_features_by_importance_:

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
        Removes non-selected features. That is, features when dropped, did not
        decrease the machine learning model performance beyond the indicated threshold.

        Args
        ----

        X: pandas dataframe of shape = [n_samples, n_features].
            The input dataframe from which features will be selected.

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

        # return the dataframe with the selected features
        return X[self.selected_features_]
