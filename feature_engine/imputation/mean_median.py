# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)
from feature_engine.docstrings import (
    Substitution,
    _variables_attribute,
    _n_features_in,
    _fit_transform,
)


@Substitution(
    variables=BaseImputer._variables_numerical_docstring,
    imputer_dict_=BaseImputer._imputer_dict_docstring,
    variables_=_variables_attribute,
    n_features_in_=_n_features_in,
    transform=BaseImputer._transform_docstring,
    fit_transform=_fit_transform,
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

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the mean or median values.

    {transform}

    {fit_transform}

    """

    def __init__(
        self,
        imputation_method: str = "median",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if imputation_method not in ["median", "mean"]:
            raise ValueError("imputation_method takes only values 'median' or 'mean'")

        self.imputation_method = imputation_method
        self.variables = _check_input_parameter_variables(variables)

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
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        # find imputation parameters: mean or median
        if self.imputation_method == "mean":
            self.imputer_dict_ = X[self.variables_].mean().to_dict()

        elif self.imputation_method == "median":
            self.imputer_dict_ = X[self.variables_].median().to_dict()

        self.n_features_in_ = X.shape[1]

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseImputer.transform.__doc__
