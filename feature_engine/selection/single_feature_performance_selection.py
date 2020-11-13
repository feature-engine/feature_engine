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


class SelectBySingleFeaturePerformance(BaseEstimator, TransformerMixin):
    """

    SelectBySingleFeaturePerformance selects features based on the performance obtained
    from a machine learning model trained utilising a single feature. In other words,
    it trains a machine learning model for every single feature, utilising that
    individual feature, then determines each model performance. If the performance of
    the model based on the single feature is greater than a user specified threshold,
    then the feature is retained, otherwise removed.

    The models trained on the individual features are trained using cross-validation.
    The performance metric to evaluate and the machine learning model to train are
    specified by the user.

    Parameters
    ----------

    variables : str or list, default=None
        The list of variable(s) to be evaluated.
        If None, the transformer will evaluate all numerical variables in the dataset.

    estimator: object, default = RandomForestClassifier()
        A Scikit-learn estimator for regression or classification.

    scoring: str, default='roc_auc'
        Desired metric to optimise the performance for the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    threshold: float, int, default = 0.5

        The value that defines if a feature will be kept or removed. Note that for
        metrics like roc-auc, r2_score and accuracy, the thresholds will be floats
        between 0 and 1. For metrics like the mean_square_error and the
        root_mean_square_error the threshold will be a big number.
        The threshold must be defined by the user.

    cv : int, default=3
        Desired number of cross-validation fold to be used to fit the estimator.

    Attributes
    ----------

    selected_features_: list
        The selected features.

    feature_performance_: dict
        A dictionary containing the feature name as key and the performance of the
        model trained on each feature as value.

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

        # find numerical variables or check variables entered by user
        self.variables = _find_numerical_variables(X, self.variables)

        # list to collect selected features
        self.selected_features_ = []

        self.feature_performance_ = {}

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

            if model["test_score"].mean() > self.threshold:
                self.selected_features_.append(feature)

            self.feature_performance_[feature] = model["test_score"].mean()

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        
        Removes non-selected features.

        Args
        ----

        X: pandas dataframe of shape = [n_samples, n_features].
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

        return X[self.selected_features_]
