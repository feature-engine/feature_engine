import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer
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


class FeatureSelection(BaseEstimator, TransformerMixin):
    """

    FeatureSelection train a machine learning model using the cross_validate function from sklearn.

    Cross_validate takes any machine learning model available in sklearn as input, 
    any metric available in sklearn as input, 
    a cross validation fold and it returns trained machine lerning model if return_estimator=True. 
    and the model performance.

    Parameters
    ----------

    variables : str or list, default=None
        The list of variable(s) to be shuffled from the dataframe.
        If None, the transformer will shuffle all numerical variables in the dataset.

    estimator: object, default = RandomForestClassifier()
        A Scikit-learn estimator for regression or classification.

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

    initial_model_performance_: float,
        performance of the model built using the original dataset.

    selected_features_: list
        The selected features.

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
        threshold=0.5,
        variables=None,
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

        # reset the index
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # find numerical variables or check variables entered by user
        self.variables = _find_numerical_variables(X, self.variables)

        # list to collect selected features
        self.selected_features_ = []

        # train model with all features and cross-validation
        for feature in self.variables:
            model = cross_validate(
                self.estimator,
                X[feature].to_frame(),
                y,
                cv=self.cv,
                return_estimator=True,
                scoring=self.scoring,
            )

            if model["test_score"].mean() > self.threshold:
                self.selected_features_.append(feature)

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Removes non-selected features. That is, features which shuffling did not
        decrease the machine learning model performance beyond the indicated threshold.

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

        # reset the index
        X = X.reset_index(drop=True)

        # check if number of columns in test dataset matches to train dataset
        _check_input_matches_training_df(X, self.input_shape_[1])

        return X[self.selected_features_]
