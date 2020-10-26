from feature_engine.variable_manipulation import (
    _define_variables,
    _find_all_variables
)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.utils.validation import check_is_fitted


class ShuffleFeatures(BaseEstimator, TransformerMixin):

    """

    ShuffleFeatures reorganizes the values inside each feature, one feature
    at the time, from a dataframe and determines how that permutation affects
    the performance metric of the machine learning algorithm.

    If the variables are important, a random permutation of their values will
    decrease dramatically any of these metrics. Contrarily, the permutation of
    values should have little to no effect on the model performance metric we
    are assessing.


    Parameters
    ----------

    variables : str or list, default=None
        The list of variable(s) to be shuffled from the dataframe.
        If None, the transformer will shuffle all variables in the dataset.

    estimator: object, default = RandomForestClassifier()
        estimator object implementing ‘fit’
        The object to use to fit the data.

    scoring: str, default='neg_mean_squared_error'
        Desired metric to optimise the performance for the tree. Comes from
        sklearn metrics. See DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    threshold: float
        the value that defines if a feature will be kept or removed.


    cv : int, default=3
        Desired number of cross-validation fold to be used to fit the decision
        tree.

    """

    def __init__(
        self,
        estimator=RandomForestClassifier(),
        scoring="neg_mean_squared_error",
        cv=3,
        threshold=0.01,
        variables=None,
    ):

        if not isinstance(cv, int) or cv < 0:
            raise ValueError("cv can only take only positive integers")

        self.variables = _define_variables(variables)
        self.estimator = estimator
        self.scoring = scoring
        self.threshold = threshold
        self.cv = cv

    def fit(self, X, y):
        """

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe

        y: array-like of shape (n_samples)
            Target variable. Required to train the estimator.


        Attributes
        ----------

        shuffled_features_: dict
            The shuffled features values

        """

        # check input dataframe
        X = _is_dataframe(X)

        # find all variables or check those entered are in the dataframe
        self.variables = _find_all_variables(X, self.variables)

        # Fit machine learning model with the input estimator if provided.
        # If the estimator is not provided, default to random tree model
        # depending on value of self.regression

        model_scores = cross_validate(
            self.estimator,
            X,
            y,
            cv=self.cv,
            return_estimator=True,
            scoring=self.scoring,
        )
        model_performance = model_scores["test_score"].mean()

        # dict to collect features and their performance_drift
        self.performance_drifts_ = {}

        # list to collect selected features
        self.selected_features_ = []

        # shuffle features and save feature performance drift into a dict
        for feature in self.variables:

            #  Create a copy of X
            X_shuffled = X.copy()

            # shuffle individual feature
            X_shuffled[feature] = (
                X_shuffled[feature].sample(frac=1).reset_index(drop=True)
            )

            # fit the estimator with the new data containing the shuffled feature
            shuffled_model_scores = cross_validate(
                self.estimator,
                X_shuffled,
                y,
                cv=self.cv,
                return_estimator=True,
                scoring=self.scoring,
            )
            # calculate the model performance for the new data containing the shuffled feature
            shuffled_model_performance = shuffled_model_scores["test_score"].mean()

            # Calculate drift in model performance after the feature has
            # been shuffled.
            drift = model_performance - shuffled_model_performance

            # Save feature and its performance drift in the
            # features_performance_drifts_ attribute.
            self.performance_drifts_[feature] = drift

            # save the selected features to keep in attribute "selected_features_"
            if drift > self.threshold:
                self.selected_features_.append(feature)

        return self

    def transform(self, X):
        """

        Updates the X dataframe with the new shuffled features.

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features].
            The input dataframe from which feature values will be shuffled.


        Returns
        -------

        X_transformed: pandas dataframe of shape = [n_samples, n_features - len(dropped features)]
            Pandas dataframe with the selected features
        """

        # check if fit is performed prior to transform
        check_is_fitted(self)

        # check if input is a dataframe
        X = _is_dataframe(X)

        # Create a list of the features to be dropped depending on the threshold value
        columns_to_drop = [
            feature
            for (feature, drift) in self.performance_drifts_.items()
            if drift <= self.threshold
        ]

        return X.drop(columns=columns_to_drop)
