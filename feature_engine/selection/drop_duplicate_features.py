from typing import List, Union

import pandas as pd

from feature_engine.dataframe_checks import _check_contains_na, _is_dataframe
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.validation import _return_tags
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
)

Variables = Union[None, int, str, List[Union[str, int]]]


class DropDuplicateFeatures(BaseSelector):
    """
    DropDuplicateFeatures() finds and removes duplicated features in a dataframe.

    Duplicated features are identical features, regardless of the variable or column
    name. If they show the same values for every observation, then they are considered
    duplicated.

    The transformer will first identify and store the duplicated variables. Next, the
    transformer will drop these variables from a dataframe.

    Parameters
    ----------
    variables: list, default=None
        The list of variables to evaluate. If None, the transformer will evaluate all
        variables in the dataset.

    missing_values : str, default=ignore
        Takes values 'raise' and 'ignore'. Whether the missing values should be raised
        as error or ignored when finding duplicated features.

    Attributes
    ----------
    features_to_drop_:
        Set with the duplicated features that will be dropped.

    duplicated_feature_sets_:
        Groups of duplicated features. Each list is a group of duplicated features.

    variables_:
        The variables to consider for the feature selection.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Find duplicated features.
    transform:
        Remove duplicated features
    fit_transform:
        Fit to data. Then transform it.
    """

    def __init__(self, variables: Variables = None, missing_values: str = "ignore"):

        if missing_values not in ["raise", "ignore"]:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'.")

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

        Returns
        -------
        self
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find all variables or check those entered are in the dataframe
        self.variables_ = _find_all_variables(X, self.variables)

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

        self.n_features_in_ = X.shape[1]

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseSelector.transform.__doc__

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict
