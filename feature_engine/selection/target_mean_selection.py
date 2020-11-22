from typing import List, Union

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
    _check_contains_na,
)

from feature_engine.discretisation import (
    EqualWidthDiscretiser,
    EqualFrequencyDiscretiser,
)

from feature_engine.encoding import MeanEncoder

from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_categorical_variables,
    _find_or_check_numerical_variables,
)

Variables = Union[None, int, str, List[Union[str, int]]]


class SelectByTargetMeanPerformance(BaseEstimator, TransformerMixin):
    """
    SelectByTargetMeanPerformance selects features by using the mean value of the
    target per category or bin as proxy of target estimation, by determining its
    performance.

    Works with both numerical and categorical variables.

    The transformer works as follows:

    1) Separates the training set into train and test sets.

    Then, for each categorical variable:

    2) Determine the mean value of the target for each category of the
    variable using the train set (equivalent of Target mean encoding)

    3) Replaces the categories in the test set, by the target mean values
    determined from the train set

    4) Using the encoded variable calculates the roc-auc or r2

    5) Selects the features which roc-auc or r2 is bigger than the indicated
    threshold

    For each numerical variable:

    2) Discretize the variable into intervals of equal width or equal frequency
    (uses the discretizers of Feature-engine)

    3) Determine the mean value of the target for each interval of the
    variable using the train set (equivalent of Target mean encoding)

    3) Replaces the intervals in the test set, by the target mean values
    determined from the train set

    4) Using the encoded variable calculates the roc-auc or r2

    5) Selects the features which roc-auc or r2 is bigger than the indicated
    threshold

    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        variables in the dataset.

    scoring: string, default='roc_auc_score'
        This indicates the metrics score to perform the feature selection.
        The current implementation supports 'roc_auc_score' and 'r2_score'.

    threshold: float, default = 0.5
        The performance threshold above which a feature will be selected.

    bins: int, default = 5
        If the dataset contains numerical variables, the number of bins into which
        the values will be sorted.

    strategy: str, default = equal_width
        whether to create the bins for discretisation of numerical variables of
        equal width or equal frequency.

    test_size: float, default=0.3
        The test size setting of the data in the train_test_split method.

    random_state: int, default=0
        The random state setting in the train_test_split method.

    Attributes
    ----------
    selected_features_: list
        The selected features.

    feature_performance_: dict
        A dictionary containing the feature name as key and the performance of the
        model trained on each feature as value.

    """

    def __init__(
        self,
        variables: Variables = None,
        scoring: str = "roc_auc_score",
        threshold: float = 0.5,
        bins: int = 5,
        strategy: str = "equal_width",
        test_size: float = 0.3,
        random_state: int = None,
    ):

        if scoring not in ["roc_auc_score", "r2_score"]:
            raise ValueError(
                "At the moment, the selector can evaluate only the "
                "roc_auc and r2 scores. Please enter either "
                "'roc_auc_score' or 'r2_score' for the parameter "
                "'scoring'"
            )

        if not isinstance(threshold, float):
            raise ValueError("threshold can only take float")

        if scoring == "roc_auc_score" and (threshold < 0.5 or threshold > 1):
            raise ValueError(
                "roc-auc score should vary between 0.5 and 1. Pick a "
                "threshold within this interval."
            )

        if scoring == "r2_score" and (threshold < 0 or threshold > 1):
            raise ValueError(
                "r2 score should vary between 0 and 1. Pick a "
                "threshold within this interval."
            )

        if not isinstance(bins, int):
            raise TypeError("'bins' takes only integers")

        if strategy not in ["equal_width", "equal_frequency"]:
            raise ValueError(
                "'strategy' takes boolean values 'equal_width' and "
                "'equal_frequency'."
            )

        if not isinstance(test_size, float) or test_size >= 1 or test_size <= 0:
            raise ValueError("'test_size' takes floats between 0 and 1")

        if not isinstance(random_state, int):
            raise TypeError("'random_state' takes only integers")

        self.variables = _check_input_parameter_variables(variables)
        self.scoring = scoring
        self.threshold = threshold
        self.bins = bins
        self.strategy = strategy
        self.test_size = test_size
        self.random_state = random_state

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

        # check if df contains na
        _check_contains_na(X, self.variables)

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X[self.variables],
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # find or check for categorical and numerical variables
        self.variables_categorical_ = _find_or_check_categorical_variables(
            X, self.variables
        )
        self.variables_numerical_ = _find_or_check_numerical_variables(
            X, self.variables
        )

        # encode variables with mean of target
        if len(self.variables_categorical_) > 0:
            _pipeline_categorical = self._make_categorical_pipeline()
            _pipeline_categorical.fit(X_train, y_train)
            X_test = _pipeline_categorical.transform(X_test)

        if len(self.variables_numerical_) > 0:
            _pipeline_numerical = self._make_numerical_pipeline()
            _pipeline_numerical.fit(X_train, y_train)
            X_test = _pipeline_numerical.transform(X_test)

        # select features
        if self.scoring == "roc_auc_score":
            self.selected_features_ = [
                f
                for f in self.variables
                if roc_auc_score(y_test, X_test[f]) > self.threshold
            ]

            self.feature_performance_ = {
                f: roc_auc_score(y_test, X_test[f]) for f in self.variables
            }

        else:
            self.selected_features_ = [
                f
                for f in self.variables
                if r2_score(y_test, X_test[f]) > self.threshold
            ]

            self.feature_performance_ = {
                f: r2_score(y_test, X_test[f]) for f in self.variables
            }

        self.input_shape_ = X.shape

        return self

    def _make_numerical_pipeline(self):

        # initialize categorical encoder
        if self.strategy == "equal_width":
            discretizer = EqualWidthDiscretiser(
                bins=self.bins, variables=self.variables_numerical_, return_object=True
            )
        else:
            discretizer = EqualFrequencyDiscretiser(
                q=self.bins, variables=self.variables_numerical_, return_object=True
            )

        encoder = MeanEncoder(variables=self.variables_numerical_)

        _pipeline_numerical = Pipeline(
            [
                ("discretization", discretizer),
                ("encoder", encoder),
            ]
        )

        return _pipeline_numerical

    def _make_categorical_pipeline(self):

        _pipeline_categorical = MeanEncoder(variables=self.variables_categorical_)

        return _pipeline_categorical

    def transform(self, X: pd.DataFrame):
        """
        Removes non-selected features.

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

        _check_contains_na(X, self.variables)

        return X[self.selected_features_]
