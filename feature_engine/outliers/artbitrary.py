# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause


from typing import Optional

import pandas as pd

from feature_engine._check_init_parameters.check_input_dictionary import (
    _check_numerical_dict,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _left_tail_caps_docstring,
    _n_features_in_docstring,
    _right_tail_caps_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _missing_values_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    check_X,
)
from feature_engine.outliers.base_outlier import BaseOutlier
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import check_numerical_variables


@Substitution(
    missing_values=_missing_values_docstring,
    right_tail_caps_=_right_tail_caps_docstring,
    left_tail_caps_=_left_tail_caps_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class ArbitraryOutlierCapper(BaseOutlier):
    """
    The ArbitraryOutlierCapper() caps the maximum or minimum values of a variable
    at an arbitrary value indicated by the user.

    You must provide the maximum or minimum values that will be used to cap each
    variable in a dictionary containing the features as keys and the capping values as
    values.

    More details in the :ref:`User Guide <arbitrary_capper>`.

    Parameters
    ----------
    max_capping_dict: dictionary, default=None
        Dictionary containing the user specified capping values for the right tail of
        the distribution of each variable to cap (maximum values).

    min_capping_dict: dictionary, default=None
        Dictionary containing user specified capping values for the eft tail of the
        distribution of each variable to cap (minimum values).

    {missing_values}

    Attributes
    ----------
    {right_tail_caps_}

    {left_tail_caps_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------

    {fit}

    {fit_transform}

    transform:
        Cap the variables.

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.outliers import ArbitraryOutlierCapper
    >>> X = pd.DataFrame(dict(x1 = [1,2,3,4,5,6,7,8,9,10]))
    >>> aoc = ArbitraryOutlierCapper(max_capping_dict=dict(x1 =  8),
    >>>                              min_capping_dict=dict(x1 = 2))
    >>> aoc.fit(X)
    >>> aoc.transform(X)
       x1
    0   2
    1   2
    2   3
    3   4
    4   5
    5   6
    6   7
    7   8
    8   8
    9   8
    """

    def __init__(
        self,
        max_capping_dict: Optional[dict] = None,
        min_capping_dict: Optional[dict] = None,
        missing_values: str = "raise",
    ) -> None:

        if not max_capping_dict and not min_capping_dict:
            raise ValueError(
                "Please provide at least 1 dictionary with the capping values."
            )

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'")

        _check_numerical_dict(max_capping_dict)
        _check_numerical_dict(min_capping_dict)

        self.max_capping_dict = max_capping_dict
        self.min_capping_dict = min_capping_dict
        self.missing_values = missing_values

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y: pandas Series, default=None
            y is not needed in this transformer. You can pass y or None.
        """
        X = check_X(X)

        # find variables to be capped
        if self.min_capping_dict is None and self.max_capping_dict:
            self.variables_ = [x for x in self.max_capping_dict.keys()]
        elif self.max_capping_dict is None and self.min_capping_dict:
            self.variables_ = [x for x in self.min_capping_dict.keys()]
        elif self.min_capping_dict and self.max_capping_dict:
            tmp = self.min_capping_dict.copy()
            tmp.update(self.max_capping_dict)
            self.variables_ = [x for x in tmp.keys()]

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)
            _check_contains_inf(X, self.variables_)

        # find or check for numerical variables
        self.variables_ = check_numerical_variables(X, self.variables_)

        if self.max_capping_dict is not None:
            self.right_tail_caps_ = self.max_capping_dict
        else:
            self.right_tail_caps_ = {}

        if self.min_capping_dict is not None:
            self.left_tail_caps_ = self.min_capping_dict
        else:
            self.left_tail_caps_ = {}

        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cap the variable values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the capped variables.
        """
        return super()._transform(X)

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
