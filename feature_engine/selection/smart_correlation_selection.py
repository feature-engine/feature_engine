from typing import List, Union

import pandas as pd
from sklearn.model_selection import cross_validate

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    check_X,
)
from feature_engine._docstrings.selection._docstring import (
    _cv_docstring,
    _estimator_docstring,
    _get_support_docstring,
    _missing_values_docstring,
    _scoring_docstring,
    _variables_attribute_docstring,
    _variables_numerical_docstring,
)
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.variable_handling._init_parameter_checks import (
    _check_init_parameter_variables,
)
from feature_engine.variable_handling.variable_type_selection import (
    find_or_check_numerical_variables,
)

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    estimator=_estimator_docstring,
    scoring=_scoring_docstring,
    cv=_cv_docstring,
    confirm_variables=_confirm_variables_docstring,
    variables=_variables_numerical_docstring,
    missing_values=_missing_values_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
)
class SmartCorrelatedSelection(BaseSelector):
    """
    SmartCorrelatedSelection() finds groups of correlated features and then selects,
    from each group, a feature following certain criteria:

    - Feature with least missing values
    - Feature with most unique values
    - Feature with highest variance
    - Feature with highest importance according to an estimator

    SmartCorrelatedSelection() returns a dataframe containing from each group of
    correlated features, the selected variable, plus all original features that were
    not correlated to any other.

    Correlation is calculated with `pandas.corr()`.

    SmartCorrelatedSelection() works only with numerical variables. Categorical
    variables will need to be encoded to numerical or will be excluded from the
    analysis.

    More details in the :ref:`User Guide <smart_correlation>`.

    Parameters
    ----------
    {variables}

    method: string or callable, default='pearson'
        Can take 'pearson', 'spearman', 'kendall' or callable. It refers to the
        correlation method to be used to identify the correlated features.

        - 'pearson': standard correlation coefficient
        - 'kendall': Kendall Tau correlation coefficient
        - 'spearman': Spearman rank correlation
        - callable: callable with input two 1d ndarrays and returning a float.

        For more details on this parameter visit the `pandas.corr()` documentation.

    threshold: float, default=0.8
        The correlation threshold above which a feature will be deemed correlated with
        another one and removed from the dataset.

    {missing_values}

    selection_method: str, default= "missing_values"
        Takes the values "missing_values", "cardinality", "variance" and
        "model_performance".

        **"missing_values"**: keeps the feature from the correlated group with least
        missing observations

        **"cardinality"**: keeps the feature from the correlated group with the highest
        cardinality.

        **"variance"**: keeps the feature from the correlated group with the highest
        variance.

        **"model_performance"**: trains a machine learning model using the correlated
        feature group and retains the feature with the highest importance.

    {estimator}

    {scoring}

    {cv}

    {confirm_variables}

    Attributes
    ----------
    correlated_feature_sets_:
        Groups of correlated features. Each list is a group of correlated features.

    features_to_drop_:
        The correlated features to remove from the dataset.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find best feature from each correlated group.

    {fit_transform}

    {get_support}

    transform:
        Return selected features.

    Notes
    -----
    For brute-force correlation selection, check Feature-engine's
    DropCorrelatedFeatures().

    See Also
    --------
    pandas.corr
    feature_engine.selection.DropCorrelatedFeatures

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.selection import SmartCorrelatedSelection
    >>> X = pd.DataFrame(dict(x1 = [1,2,1,1],
    >>>                 x2 = [2,4,3,1],
    >>>                 x3 = [1, 0, 0, 0]))
    >>> scs = SmartCorrelatedSelection(threshold=0.7)
    >>> scs.fit_transform(X)
       x2  x3
    0   2   1
    1   4   0
    2   3   0
    3   1   0

    It is also possible alternative selection methods, in this case seleting
    features with higher variance:

    >>> X = pd.DataFrame(dict(x1 = [2,4,3,1],
    >>>                 x2 = [1000,2000,1500,500],
    >>>                 x3 = [1, 0, 0, 0]))
    >>> scs = SmartCorrelatedSelection(threshold=0.7, selection_method="variance")
    >>> scs.fit_transform(X)
         x2  x3
    0  1000   1
    1  2000   0
    2  1500   0
    3   500   0
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
        cv=3,
        confirm_variables: bool = False,
    ):
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

        super().__init__(confirm_variables)

        self.variables = _check_init_parameter_variables(variables)
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
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas series. Default = None
            y is needed if selection_method == 'model_performance'.
        """

        # check input dataframe
        X = check_X(X)

        # If required exclude variables that are not in the input dataframe
        self._confirm_variables(X)

        # find all numerical variables or check those entered are in the dataframe
        self.variables_ = find_or_check_numerical_variables(X, self.variables_)

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        if self.selection_method == "model_performance" and y is None:
            raise ValueError("y is needed to fit the transformer")

        # FIND CORRELATED FEATURES
        # ========================
        # create tuples of correlated feature groups
        self.correlated_feature_sets_ = []

        # the correlation matrix
        _correlated_matrix = X[self.variables_].corr(method=self.method)

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
                feature_group = list(feature_group)  # type: ignore
                f = X[feature_group].isnull().sum().sort_values(ascending=True).index[0]
                _selected_features.append(f)

        # select the feature with most unique values
        elif self.selection_method == "cardinality":
            for feature_group in self.correlated_feature_sets_:
                feature_group = list(feature_group)  # type: ignore
                f = X[feature_group].nunique().sort_values(ascending=False).index[0]
                _selected_features.append(f)

        # select the feature with biggest variance
        elif self.selection_method == "variance":
            for feature_group in self.correlated_feature_sets_:
                feature_group = list(feature_group)  # type: ignore
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
            f for f in self.variables_ if f not in _selected_features
        ]

        # save input features
        self._get_feature_names_in(X)

        return self
