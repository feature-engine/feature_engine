from typing import List, Union

import pandas as pd

from feature_engine.dataframe_checks import _check_contains_na, check_X
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.selection._docstring import (
    _missing_values_docstring,
    _variables_all_docstring,
    _variables_attribute_docstring,
)
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.tags import _return_tags
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
)

Variables = Union[None, int, str, List[Union[str, int]]]


@Substitution(
    confirm_variables=BaseSelector._confirm_variables_docstring,
    variables=_variables_all_docstring,
    missing_values=_missing_values_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
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

    transform:
        Remove duplicated features.

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

        self.variables = _check_input_parameter_variables(variables)
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

        # If required exclude variables that are not in the input dataframe
        self._confirm_variables(X)

        # find all variables or check those entered are in the dataframe
        self.variables_ = _find_all_variables(X, self.variables_)

        if self.missing_values == "raise":
            # check if dataset contains na
            _check_contains_na(X, self.variables_)

        # create tuples of duplicated feature groups
        self.duplicated_feature_sets_ = []

        # set to collect features that are duplicated
        self.features_to_drop_ = set()  # type: ignore

        # create set of examined features
        _examined_features = set()

        for feature in self.variables_:

            # append so we can remove when we create the combinations
            _examined_features.add(feature)

            if feature not in self.features_to_drop_:

                _temp_set = set([feature])

                # features that have not been examined, are not currently examined and
                # were not found duplicates
                _features_to_compare = [
                    f
                    for f in self.variables_
                    if f not in _examined_features.union(self.features_to_drop_)
                ]

                # create combinations:
                for f2 in _features_to_compare:

                    if X[feature].equals(X[f2]):
                        self.features_to_drop_.add(f2)
                        _temp_set.add(f2)

                # if there are duplicated features
                if len(_temp_set) > 1:
                    self.duplicated_feature_sets_.append(_temp_set)

        # save input features
        self._get_feature_names_in(X)

        return self

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "all"
        return tags_dict
