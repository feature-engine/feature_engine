from typing import List, Union

import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_contains_na,
)

from feature_engine.discretisation import (
    EqualWidthDiscretiser,
    EqualFrequencyDiscretiser,
)

from feature_engine.encoding import MeanEncoder

from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
)

from feature_engine.selection.base_selector import BaseSelector

Variables = Union[None, int, str, List[Union[str, int]]]


class SelectByTargetMeanPerformance(BaseSelector):
    """
    SelectByTargetMeanPerformance() selects features by using the mean value of the
    target per category or bin, if the variable is numerical, as proxy of target
    estimation, by determining its performance.

    Works with both numerical and categorical variables.

    The transformer works as follows:

    1. Separates the training set into train and test sets.

    Then, for each categorical variable:

    2. Determine the mean value of the target for each category of the variable using
    the train set (equivalent of Target mean encoding)

    3. Replaces the categories in the test set, by the target mean values determined
    from the train set

    4. Using the encoded variable calculates the roc-auc or r2

    5. Selects the features which roc-auc or r2 is bigger than the indicated
    threshold

    For each numerical variable:

    2. Discretize the variable into intervals of equal width or equal frequency
    (uses the discretizers of Feature-engine)

    3. Determine the mean value of the target for each interval of the
    variable using the train set (equivalent of Target mean encoding)

    4. Replaces the intervals in the test set, by the target mean values
    determined from the train set

    5. Using the encoded variable calculates the roc-auc or r2

    6. Selects the features which roc-auc or r2 is bigger than the indicated
    threshold

    Parameters
    ----------
    variables : list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        variables in the dataset.

    scoring : string, default='roc_auc_score'
        This indicates the metrics score to perform the feature selection.
        The current implementation supports 'roc_auc_score' and 'r2_score'.

    threshold : float, default = None
        The performance threshold above which a feature will be selected.

    bins : int, default = 5
        If the dataset contains numerical variables, the number of bins into which
        the values will be sorted.

    strategy : str, default = equal_width
        whether to create the bins for discretization of numerical variables of
        equal width or equal frequency.

    cv : int, default=3
        Desired number of cross-validation fold to be used to fit the estimator.

    random_state : int, default=0
        The random state setting in the train_test_split method.

    Attributes
    ----------
    features_to_drop_:
        List with the features to remove from the dataset.

    feature_performance_:
        Dictionary with the performance proxy per feature.

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
        variables: Variables = None,
        scoring: str = "roc_auc_score",
        threshold: float = 0.5,
        bins: int = 5,
        strategy: str = "equal_width",
        cv: int = 3,
        random_state: int = None,
    ):

        if scoring not in ["roc_auc_score", "r2_score"]:
            raise ValueError(
                "At the moment, the selector can evaluate only the "
                "roc_auc and r2 scores. Please enter either "
                "'roc_auc_score' or 'r2_score' for the parameter "
                "'scoring'"
            )

        if threshold and not isinstance(threshold, (int, float)):
            raise ValueError("threshold can only take integer or float")

        if not isinstance(bins, int):
            raise TypeError("'bins' takes only integers")

        if strategy not in ["equal_width", "equal_frequency"]:
            raise ValueError(
                "'strategy' takes boolean values 'equal_width' and "
                "'equal_frequency'."
            )

        if not isinstance(cv, int) or cv <= 1:
            raise ValueError("cv takes integers bigger than 1")

        if random_state and not isinstance(random_state, int):
            raise TypeError("'random_state' takes only integers")

        self.variables = _check_input_parameter_variables(variables)
        self.scoring = scoring
        self.threshold = threshold
        self.bins = bins
        self.strategy = strategy
        self.cv = cv
        self.random_state = random_state

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

        # check variables
        self.variables = _find_all_variables(X, self.variables)

        # check if df contains na
        _check_contains_na(X, self.variables)

        self.input_shape_ = X.shape

        # limit df to variables to smooth code below
        X = X[self.variables].copy()

        # find categorical and numerical variables
        self.variables_categorical_ = list(X.select_dtypes(include="O").columns)
        self.variables_numerical_ = list(
            X.select_dtypes(include=["float", "integer"]).columns
        )

        # obtain cross-validation indeces
        skf = KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        skf.get_n_splits(X, y)

        if self.variables_categorical_ and self.variables_numerical_:
            _pipeline = self._make_combined_pipeline()

        elif self.variables_categorical_:
            _pipeline = self._make_categorical_pipeline()

        else:
            _pipeline = self._make_numerical_pipeline()

        # obtain feature performance with cross-validation
        feature_importances_cv = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            _pipeline.fit(X_train, y_train)

            X_test = _pipeline.transform(X_test)

            if self.scoring == "roc_auc_score":
                tmp_split = {
                    f: roc_auc_score(y_test, X_test[f]) for f in self.variables
                }
            else:
                tmp_split = {f: r2_score(y_test, X_test[f]) for f in self.variables}

            feature_importances_cv.append(pd.Series(tmp_split))

        feature_importances_cv = pd.concat(feature_importances_cv, axis=1)

        self.feature_performance_ = feature_importances_cv.mean(  # type: ignore
            axis=1
        ).to_dict()

        # select features
        if not self.threshold:
            threshold = pd.Series(self.feature_performance_).mean()
        else:
            threshold = self.threshold

        self.features_to_drop_ = [
            f
            for f in self.variables
            if self.feature_performance_[f] < threshold
        ]

        return self

    def _make_numerical_pipeline(self):

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

        return MeanEncoder(variables=self.variables_categorical_)

    def _make_combined_pipeline(self):

        if self.strategy == "equal_width":
            discretizer = EqualWidthDiscretiser(
                bins=self.bins, variables=self.variables_numerical_, return_object=True
            )
        else:
            discretizer = EqualFrequencyDiscretiser(
                q=self.bins, variables=self.variables_numerical_, return_object=True
            )

        encoder_num = MeanEncoder(variables=self.variables_numerical_)
        encoder_cat = MeanEncoder(variables=self.variables_categorical_)

        _pipeline_combined = Pipeline(
            [
                ("discretization", discretizer),
                ("encoder_num", encoder_num),
                ("encoder_cat", encoder_cat),
            ]
        )

        return _pipeline_combined

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseSelector.transform.__doc__
