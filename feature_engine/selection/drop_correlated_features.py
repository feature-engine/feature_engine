from typing import List, Union

import pandas as pd

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
    _get_support_docstring,
    _missing_values_docstring,
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
    calculated with `pandas.corr()`. Features are removed on first found first removed
    basis, without any further insight.

    DropCorrelatedFeatures() works only with numerical variables. Categorical variables
    will need to be encoded to numerical or will be excluded from the analysis.

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
            raise ValueError("threshold must be a float between 0 and 1")

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'.")

        super().__init__(confirm_variables)

        self.variables = _check_init_parameter_variables(variables)
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

        # set to collect features that are correlated
        self.features_to_drop_ = set()

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
                        self.features_to_drop_.add(f2)
                        _temp_set.add(f2)
                        _examined_features.add(f2)

                # if there are correlated features
                if len(_temp_set) > 1:
                    self.correlated_feature_sets_.append(_temp_set)

        # save input features
        self._get_feature_names_in(X)

        return self
