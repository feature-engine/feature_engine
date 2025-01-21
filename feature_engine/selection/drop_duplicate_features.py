from collections import defaultdict
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
    _variables_all_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import _check_contains_na, check_X
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags

from .base_selection_functions import _select_all_variables

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    confirm_variables=_confirm_variables_docstring,
    variables=_variables_all_docstring,
    missing_values=_missing_values_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    get_support=_get_support_docstring,
)
class DropDuplicateFeatures(BaseSelector):
    """
    DropDuplicateFeatures() finds and removes duplicated features in a dataframe.

    Duplicated features are identical features, regardless of the variable or column
    name. If they show the same values for every observation, then they are considered
    duplicated.

    This transformer works with numerical and categorical variables. The user can
    indicate a list of variables to examine. Alternatively, the transformer will
    evaluate all the variables in the dataset.

    The transformer will first identify and store the duplicated variables. Next, the
    transformer will drop these variables from a dataframe.

    More details in the :ref:`User Guide <drop_duplicate>`.

    Parameters
    ----------
    {variables}

    {missing_values}

    {confirm_variables}

    Attributes
    ----------
    features_to_drop_:
        Set with the duplicated features that will be dropped.

    duplicated_feature_sets_:
        Groups of duplicated features. Each list is a group of duplicated features.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find duplicated features.

    {fit_transform}

    {get_support}

    transform:
        Remove duplicated features.

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.selection import DropDuplicateFeatures
    >>> X = pd.DataFrame(dict(x1 = [1,1,1,1],
    >>>                     x2 = [1,1,1,1],
    >>>                     x3 = [True, False, False, False]))
    >>> ddf = DropDuplicateFeatures()
    >>> ddf.fit_transform(X)
        x1     x3
    0   1   True
    1   1  False
    2   1  False
    3   1  False
    """

    def __init__(
        self,
        variables: Variables = None,
        missing_values: str = "ignore",
        confirm_variables: bool = False,
    ):
        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'.")

        super().__init__(confirm_variables)

        self.variables = _check_variables_input_value(variables)
        self.missing_values = missing_values

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Find duplicated features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The input dataframe.
        y: None
            y is not needed for this transformer. You can pass y or None.
        """

        # check input dataframe
        X = check_X(X)

        self.variables_ = _select_all_variables(
            X, self.variables, self.confirm_variables
        )

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)

        # collect duplicate features
        _features_hashmap = defaultdict(list)

        # hash the features
        _X_hash = pd.util.hash_pandas_object(X[self.variables_].T, index=False)

        # group the features by hash
        for feature, feature_hash in _X_hash.items():
            _features_hashmap[feature_hash].append(feature)

        # create tuples of duplicated feature groups
        self.duplicated_feature_sets_ = [
            set(duplicate)
            for duplicate in _features_hashmap.values()
            if len(duplicate) > 1
        ]

        # set to collect features that are duplicated
        self.features_to_drop_ = {
            item
            for duplicates in _features_hashmap.values()
            for item in duplicates[1:]
            if duplicates and len(duplicates) > 1
        }

        # save input features
        self._get_feature_names_in(X)

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "all"

        msg = "transformers need more than 1 feature to work"
        tags_dict["_xfail_checks"]["check_fit2d_1feature"] = msg

        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
