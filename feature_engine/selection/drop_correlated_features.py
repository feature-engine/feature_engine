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
    _get_support_docstring,
    _missing_values_docstring,
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
)

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    confirm_variables=_confirm_variables_docstring,
    variables=_variables_numerical_docstring,
    missing_values=_missing_values_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
)
class DropCorrelatedFeatures(BaseSelector):
    """
    DropCorrelatedFeatures() finds and removes correlated features. Correlation is
    calculated with `pandas.corr()`. Features are removed on first found, first removed
    basis, without any further insight.

    DropCorrelatedFeatures() works only with numerical variables. Categorical variables
    will need to be encoded to numerical or will be excluded from the analysis.

    To make the selector deterministic, features are sorted alphabetically before
    examining correlation.

    More details in the :ref:`User Guide <drop_correlated>`.

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

        For more details on this parameter visit the  `pandas.corr()` documentation.

    threshold: float, default=0.8
        The correlation threshold above which a feature will be deemed correlated with
        another one and removed from the dataset.

    {missing_values}

    {confirm_variables}

    Attributes
    ----------
    features_to_drop_:
        Set with the correlated features that will be dropped.

    correlated_feature_sets_:
        Groups of correlated features. Each list is a group of correlated features.

    correlated_feature_dict_: dict
        Dictionary containing the correlated feature groups. The key is the feature
        against which all other features were evaluated. The values are the features
        correlated with the key. Key + values should be the same as the set found in
        `correlated_feature_groups`. We introduced this attribute in version 1.17.0
        because from the set, it is not easy to see which feature will be retained and
        which ones will be removed. The key is retained, the values will be dropped.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find correlated features.

    {fit_transform}

    {get_support}

    transform:
        Remove correlated features.

    Notes
    -----
    If you want to select from each group of correlated features those that are perhaps
    more predictive or more complete, check Feature-engine's SmartCorrelationSelection.

    See Also
    --------
    pandas.corr
    feature_engine.selection.SmartCorrelationSelection

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.selection import DropCorrelatedFeatures
    >>> X = pd.DataFrame(dict(x1 = [1,2,1,1], x2 = [2,4,3,1], x3 = [1, 0, 0, 1]))
    >>> dcf = DropCorrelatedFeatures(threshold=0.7)
    >>> dcf.fit_transform(X)
        x1  x3
    0   1   1
    1   2   0
    2   1   0
    3   1   1
    """

    def __init__(
        self,
        variables: Variables = None,
        method: str = "pearson",
        threshold: float = 0.8,
        missing_values: str = "ignore",
        confirm_variables: bool = False,
    ):

        if not isinstance(threshold, float) or threshold < 0 or threshold > 1:
            raise ValueError(
                "`threshold` must be a float between 0 and 1. "
                f"Got {threshold} instead."
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError(
                "`missing_values` takes only values 'raise' or 'ignore'. "
                f"Got {missing_values} instead."
            )

        super().__init__(confirm_variables)

        self.variables = _check_variables_input_value(variables)
        self.method = method
        self.threshold = threshold
        self.missing_values = missing_values

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Find the correlated features.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y : pandas series. Default = None
            y is not needed in this transformer. You can pass y or None.
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

        # sort features alphabetically to make selector deterministic
        features = sorted(self.variables_)

        correlated_groups, features_to_drop, correlated_dict = find_correlated_features(
            X, features, self.method, self.threshold
        )

        self.features_to_drop_ = features_to_drop
        self.correlated_feature_sets_ = correlated_groups
        self.correlated_feature_dict_ = correlated_dict

        # save input features
        self._get_feature_names_in(X)

        return self
