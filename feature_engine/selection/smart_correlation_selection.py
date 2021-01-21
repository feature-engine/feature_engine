from typing import List, Union

import pandas as pd
from sklearn.model_selection import cross_validate

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_contains_na,
)
from feature_engine.variable_manipulation import (
    _find_or_check_numerical_variables,
    _check_input_parameter_variables,
)
from feature_engine.selection.base_selector import BaseSelector

Variables = Union[None, int, str, List[Union[str, int]]]


class SmartCorrelatedSelection(BaseSelector):
    """
    SmartCorrelatedSelection() finds groups of correlated features and then selects,
    from each group, a feature following certain criteria:

    - Feature with least missing values
    - Feature with most unique values
    - Feature with highest variance
    - Best performing feature according to estimator entered by user

    SmartCorrelatedSelection() returns a dataframe containing from each group of
    correlated features, the selected variable, plus all original features that were
    not correlated to any other.

    Correlation is calculated with `pandas.corr()`.

    SmartCorrelatedSelection() works only with numerical variables. Categorical
    variables will need to be encoded to numerical or will be excluded from the
    analysis.

    Parameters
    ----------
    variables : list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        numerical variables in the dataset.

    method : string, default='pearson'
        Can take 'pearson', 'spearman' or'kendall'. It refers to the correlation method
        to be used to identify the correlated features.

        - pearson : standard correlation coefficient
        - kendall : Kendall Tau correlation coefficient
        - spearman : Spearman rank correlation

    threshold : float, default=0.8
        The correlation threshold above which a feature will be deemed correlated with
        another one and removed from the dataset.

    missing_values : str, default=ignore
        Takes values 'raise' and 'ignore'. Whether the missing values should be raised
        as error or ignored when determining correlation.

    selection_method : str, default= "missing_values"
        Takes the values "missing_values", "cardinality", "variance" and
        "model_performance".

        "missing_values": keeps the feature from the correlated group with least
        missing observations

        "cardinality": keeps the feature from the correlated group with the highest
        cardinality.

        "variance": keeps the feature from the correlated group with the highest
        variance.

        "model_performance": trains a machine learning model using the correlated
        feature group and retains the feature with the highest importance.

    estimator : object, default = None
        A Scikit-learn estimator for regression or classification.

    scoring : str, default='roc_auc'
        Desired metric to optimise the performance of the estimator. Comes from
        sklearn.metrics. See the model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    cv : int, default=3
        Cross-validation fold to be used to fit the estimator.

    Attributes
    ----------
    correlated_feature_sets_:
        Groups of correlated features.  Each list is a group of correlated features.

    features_to_drop_:
        The correlated features to remove from the dataset.

    Methods
    -------
    fit:
        Find best feature from each correlated groups.
    transform:
        Return selected features.
    fit_transform:
        Fit to the data. Then transform it.

    See Also
    --------
    pandas.corr
    feature_engine.selection.DropCorrelatedFeatures
    """

    def __init__(
        self,
        variables: Variables = None,
        method: str = "pearson",
        threshold: float = 0.8,
        missing_values: str = "ignore",
        selection_method: str = "missing_values",
        estimator=None,
        scoring: str = "roc_auc",
        cv: int = 3,
    ):

        if method not in ["pearson", "spearman", "kendall"]:
            raise ValueError(
                "correlation method takes only values 'pearson', 'spearman', 'kendall'"
            )

        if not isinstance(threshold, float) or threshold < 0 or threshold > 1:
            raise ValueError("threshold must be a float between 0 and 1")

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'.")

        if selection_method not in [
            "missing_values",
            "cardinality",
            "variance",
            "model_performance",
        ]:
            raise ValueError(
                "selection_method takes only values 'missing_values', 'cardinality', "
                "'variance' or 'model_performance'."
            )

        if not isinstance(cv, int) or cv < 1:
            raise ValueError("cv can only take positive integers bigger than 1")

        if selection_method == "model_performance" and estimator is None:
            raise ValueError(
                "Please provide an estimator, e.g., "
                "RandomForestClassifier or select another "
                "selection_method"
            )

        if selection_method == "missing_values" and missing_values == "raise":
            raise ValueError(
                "To select the variables with least missing values, we "
                "need to allow this transformer to contemplate variables "
                "with NaN by setting missing_values to 'ignore."
            )

        self.variables = _check_input_parameter_variables(variables)
        self.method = method
        self.threshold = threshold
        self.missing_values = missing_values
        self.selection_method = selection_method
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Find the correlated feature groups. Determine which feature should be selected
        from each group.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y : pandas series. Default = None
            y is needed if selection_method == 'model_performance'.

        Returns
        -------
        self
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find all numerical variables or check those entered are in the dataframe
        self.variables = _find_or_check_numerical_variables(X, self.variables)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables)

        if self.selection_method == "model_performance" and y is None:
            raise ValueError("y is needed to fit the transformer")

        # FIND CORRELATED FEATURES
        # ========================
        # create tuples of correlated feature groups
        self.correlated_feature_sets_ = []

        # the correlation matrix
        _correlated_matrix = X[self.variables].corr(method=self.method)

        # create set of examined features, helps to determine feature combinations
        # to evaluate below
        _examined_features = set()

        # for each feature in the dataset (columns of the correlation matrix)
        for feature in _correlated_matrix.columns:

            if feature not in _examined_features:

                # append so we can exclude when we create the combinations
                _examined_features.add(feature)

                # here we collect potentially correlated features
                # we need this for the correlated groups sets
                _temp_set = set([feature])

                # features that have not been examined, are not currently examined and
                # were not found correlated
                _features_to_compare = [
                    f for f in _correlated_matrix.columns if f not in _examined_features
                ]

                # create combinations:
                for f2 in _features_to_compare:

                    # if the correlation is higher than the threshold
                    # we are interested in absolute correlation coefficient value
                    if abs(_correlated_matrix.loc[f2, feature]) > self.threshold:
                        # add feature (f2) to our correlated set
                        _temp_set.add(f2)
                        _examined_features.add(f2)

                # if there are correlated features
                if len(_temp_set) > 1:
                    self.correlated_feature_sets_.append(_temp_set)

        # SELECT 1 FEATURE FROM EACH GROUP
        # ================================

        # list to collect selected features
        # we start it with all features that were either not examined, i.e., categorical
        # variables, or not found correlated
        _selected_features = [
            f for f in X.columns if f not in set().union(*self.correlated_feature_sets_)
        ]

        # select the feature with least missing values
        if self.selection_method == "missing_values":
            for feature_group in self.correlated_feature_sets_:
                f = X[feature_group].isnull().sum().sort_values(ascending=True).index[0]
                _selected_features.append(f)

        # select the feature with most unique values
        elif self.selection_method == "cardinality":
            for feature_group in self.correlated_feature_sets_:
                f = X[feature_group].nunique().sort_values(ascending=False).index[0]
                _selected_features.append(f)

        # select the feature with biggest variance
        elif self.selection_method == "variance":
            for feature_group in self.correlated_feature_sets_:
                f = X[feature_group].std().sort_values(ascending=False).index[0]
                _selected_features.append(f)

        # select best performing feature according to estimator
        else:
            for feature_group in self.correlated_feature_sets_:

                # feature_group = list(feature_group)
                temp_perf = []

                # train a model for every feature
                for feature in feature_group:
                    model = cross_validate(
                        self.estimator,
                        X[feature].to_frame(),
                        y,
                        cv=self.cv,
                        return_estimator=False,
                        scoring=self.scoring,
                    )

                    temp_perf.append(model["test_score"].mean())

                # select best performing feature from group
                f = list(feature_group)[temp_perf.index(max(temp_perf))]
                _selected_features.append(f)

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
