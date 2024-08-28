from types import GeneratorType
from typing import List, Union

import pandas as pd

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.selection import (
    _confirm_variables_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.selection._docstring import (
    _cv_docstring,
    _groups_docstring,
    _estimator_docstring,
    _get_support_docstring,
    _missing_values_docstring,
    _scoring_docstring,
    _variables_attribute_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    check_X,
)
from feature_engine.selection.base_selector import BaseSelector

from .base_selection_functions import (
    _select_numerical_variables,
    find_correlated_features,
    single_feature_performance,
)

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    estimator=_estimator_docstring,
    scoring=_scoring_docstring,
    cv=_cv_docstring,
    groups=_groups_docstring,
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

    - Feature with the least missing values.
    - Feature with the highest cardinality (greatest number of unique values).
    - Feature with the highest variance.
    - Feature with the highest importance according to an estimator.

    SmartCorrelatedSelection() returns a dataframe containing from each group of
    correlated features, the selected variable, plus all the features that were
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

        **"missing_values"**: keeps the feature from the correlated group with the least
        missing observations.

        **"cardinality"**: keeps the feature from the correlated group with the highest
        cardinality.

        **"variance"**: keeps the feature from the correlated group with the highest
        variance.

        **"model_performance"**: trains a machine learning model using each of the
        features in a correlated group and retains the feature with the highest
        importance.

    {estimator}

    {scoring}

    {cv}

    {groups}

    {confirm_variables}

    Attributes
    ----------
    correlated_feature_sets_:
        Groups of correlated features. Each list is a group of correlated features.

    correlated_feature_dict_: dict
        Dictionary containing the correlated feature groups. The key is the feature
        against which all other features were evaluated. The values are the features
        correlated with the key. Key + values should be the same as the set found in
        `correlated_feature_groups`. We introduced this attribute in version 1.17.0
        because from the set, it is not easy to see which feature will be retained and
        which ones will be removed. The key is retained, the values will be dropped.

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

    It is also possible to use alternative selection methods. Here, we select those
    features with the higher variance:

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
        groups=None,
        confirm_variables: bool = False,
    ):
        if not isinstance(threshold, float) or threshold < 0 or threshold > 1:
            raise ValueError(
                f"`threshold` must be a float between 0 and 1. Got {threshold} instead."
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "missing_values takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        if selection_method not in [
            "missing_values",
            "cardinality",
            "variance",
            "model_performance",
        ]:
            raise ValueError(
                "selection_method takes only values 'missing_values', 'cardinality', "
                f"'variance' or 'model_performance'. Got {selection_method} instead."
            )

        if selection_method == "model_performance" and estimator is None:
            raise ValueError(
                "Please provide an estimator, e.g., "
                "RandomForestClassifier or select another "
                "selection_method."
            )

        if selection_method == "missing_values" and missing_values == "raise":
            raise ValueError(
                "When `selection_method = 'missing_values'`, you need to set "
                f"`missing_values` to `'ignore'`. Got {missing_values} instead."
            )

        super().__init__(confirm_variables)

        self.variables = _check_variables_input_value(variables)
        self.method = method
        self.threshold = threshold
        self.missing_values = missing_values
        self.selection_method = selection_method
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.groups = groups

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

        self.variables_ = _select_numerical_variables(
            X, self.variables, self.confirm_variables
        )

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        if self.selection_method == "model_performance" and y is None:
            raise ValueError(
                "When `selection_method = 'model_performance'` y is needed to "
                "fit the transformer."
            )

        if self.selection_method == "missing_values":
            features = (
                X[self.variables_]
                .isnull()
                .sum()
                .sort_values(ascending=True)
                .index.to_list()
            )
        elif self.selection_method == "variance":
            features = (
                X[self.variables_].std().sort_values(ascending=False).index.to_list()
            )
        elif self.selection_method == "cardinality":
            features = (
                X[self.variables_]
                .nunique()
                .sort_values(ascending=False)
                .index.to_list()
            )
        else:
            features = sorted(self.variables_)

        correlated_groups, features_to_drop, correlated_dict = find_correlated_features(
            X, features, self.method, self.threshold
        )

        # select best performing feature according to estimator
        if self.selection_method == "model_performance":
            correlated_dict = dict()
            cv = list(self.cv) if isinstance(self.cv, GeneratorType) else self.cv
            for feature_group in correlated_groups:
                feature_performance, _ = single_feature_performance(
                    X=X,
                    y=y,
                    variables=feature_group,
                    estimator=self.estimator,
                    cv=cv,
                    groups=self.groups,
                    scoring=self.scoring,
                )
                # get most important feature
                f_i = (
                    pd.Series(feature_performance).sort_values(ascending=False).index[0]
                )
                correlated_dict[f_i] = feature_group.difference({f_i})

            # convoluted way to pick up the variables from the sets in the
            # order shown in the dictionary. Helps make transformer deterministic
            features_to_drop = [
                variable
                for set_ in correlated_dict.values()
                for variable in sorted(set_)
            ]

        self.features_to_drop_ = features_to_drop
        self.correlated_feature_sets_ = correlated_groups
        self.correlated_feature_dict_ = correlated_dict

        # save input features
        self._get_feature_names_in(X)

        return self
