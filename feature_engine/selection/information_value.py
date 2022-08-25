# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from typing import List, Union

import numpy as np
import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.encoding._docstrings import (
    _ignore_format_docstring,
    _variables_docstring,
)
from feature_engine.encoding.base_encoder import CategoricalMethodsMixin
from feature_engine.encoding.woe import WoE
from feature_engine.selection._docstring import (
    _threshold_docstring,
    _features_to_drop_docstring,
)
from feature_engine.selection.base_selector import BaseSelector
from feature_engine.variable_manipulation import _check_input_parameter_variables


@Substitution(
    variables=_variables_docstring,
    threshold=_threshold_docstring,
    ignore_format=_ignore_format_docstring,
    variables_=_variables_attribute_docstring,
    features_to_drop=_features_to_drop_docstring,
    feature_names_in=_feature_names_in_docstring,
    n_features_in=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    confirm_variables=BaseSelector._confirm_variables_docstring,
)
class SelectByInformationValue(BaseSelector, CategoricalMethodsMixin, WoE):
    """
    InformationValue() calculates the information value (IV) for each variable.
    The transformer is only compatible with categorical variables (type 'object'
    or 'categorical') and binary classification.

    You can pass a list of variables to score. Alternatively, the
    transformer will find and score all categorical variables (type 'object'
    or 'categorical').

    IV will allow you to assess each variable's independent contribution to
    the target variable and rank the variables in terms of their univariate
    predictive strength.

    Parameters
    ----------
    {variables}

    {threshold}

    {ignore_format}

    {confirm_variables}

    Attributes
    ----------
    {variables_}

    {feature_names_in}

    {n_features_in}

    information_values_:
        A dictionary with the information values for each feature.

    {features_to_drop}

    Methods
    -------
    fit:
        Find features with high information value.

    {fit_transform}

    transform:
        Remove features with low information value.

    See Also
    --------
    feature_engine.encoding.WoEEncoder
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        threshold: Union[float, int] = 0.2,
        ignore_format: bool = False,
        confirm_variables: bool = False,
    ) -> None:
        if not isinstance(threshold, (int, float)):
            raise Warning(
                f"threshold must be a an integer or a float. Got {threshold} "
                "instead."
            )

        super().__init__(confirm_variables)
        self.variables = _check_input_parameter_variables(variables)
        self.threshold = threshold
        # Used in WoE class
        self.ignore_format = ignore_format

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Learn the information value.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y: pandas series of shape = [n_samples, ]
            Target, must be binary.
        """
        # check input dataframe
        X, y = self._check_fit_input(X, y)

        # find categorical variables, and check for NA
        # comes from base encoder
        self._fit(X)
        self._get_feature_names_in(X)

        # check that there are more than 1 variable to select from
        self._check_variable_number()

        self.information_values_ = {}

        for var in self.variables_:
            total_pos, total_neg, woe = self._calculate_woe(X, y, var)
            iv = self._calculate_iv(total_pos, total_neg, woe)
            self.information_values_[var] = iv

        self.features_to_drop_ = [
            f
            for f in self.information_values_.keys()
            if self.information_values_[f] < self.threshold
        ]

        return self

    def _calculate_iv(self, pos, neg, woe):
        return np.sum((pos - neg) * woe)
