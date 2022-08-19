# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine._variable_handling.init_parameter_checks import (
    _check_init_parameter_variables,
)
from feature_engine._variable_handling.variable_type_selection import (
    _find_or_check_numerical_variables,
)
from feature_engine.dataframe_checks import check_X
from feature_engine.imputation.base_imputer import BaseImputer


@Substitution(
    variables=BaseImputer._variables_numerical_docstring,
    imputer_dict_=BaseImputer._imputer_dict_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    transform=BaseImputer._transform_docstring,
    fit_transform=_fit_transform_docstring,
)
class MeanMedianImputer(BaseImputer):
    """
    The MeanMedianImputer() replaces missing data by the mean or median value of the
    variable. It works only with numerical variables.

    You can pass a list of variables to impute. Alternatively, the
    MeanMedianImputer() will automatically select all variables of type numeric in the
    training set.

    More details in the :ref:`User Guide <mean_median_imputer>`.

    Parameters
    ----------
    imputation_method: str, default='median'
        Desired method of imputation. Can take 'mean' or 'median'.

    {variables}

    Attributes
    ----------
    {imputer_dict_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the mean or median values.

    {fit_transform}

    {transform}

    """

    def __init__(
        self,
        imputation_method: str = "median",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if imputation_method not in ["median", "mean"]:
            raise ValueError("imputation_method takes only values 'median' or 'mean'")

        self.imputation_method = imputation_method
        self.variables = _check_init_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the mean or median values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: pandas series or None, default=None
            y is not needed in this imputation. You can pass None or y.
        """

        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        # find imputation parameters: mean or median
        if self.imputation_method == "mean":
            self.imputer_dict_ = X[self.variables_].mean().to_dict()

        elif self.imputation_method == "median":
            self.imputer_dict_ = X[self.variables_].median().to_dict()

        self._get_feature_names_in(X)

        return self
