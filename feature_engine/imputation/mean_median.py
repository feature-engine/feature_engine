# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _imputer_dict_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _transform_imputers_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)


@Substitution(
    variables=_variables_numerical_docstring,
    imputer_dict_=_imputer_dict_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    transform=_transform_imputers_docstring,
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

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from feature_engine.imputation import MeanMedianImputer
    >>> X = pd.DataFrame(dict(
    >>>        x1 = [np.nan,1,1,0,np.nan],
    >>>        x2 = ["a", np.nan, "b", np.nan, "a"],
    >>>        ))
    >>> mmi = MeanMedianImputer(imputation_method='median')
    >>> mmi.fit(X)
    >>> mmi.transform(X)
        x1   x2
    0  1.0    a
    1  1.0  NaN
    2  1.0    b
    3  0.0  NaN
    4  1.0    a
    """

    def __init__(
        self,
        imputation_method: str = "median",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if imputation_method not in ["median", "mean"]:
            raise ValueError("imputation_method takes only values 'median' or 'mean'")

        self.imputation_method = imputation_method
        self.variables = _check_variables_input_value(variables)

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
        if self.variables is None:
            self.variables_ = find_numerical_variables(X)
        else:
            self.variables_ = check_numerical_variables(X, self.variables)

        # find imputation parameters: mean or median
        if self.imputation_method == "mean":
            self.imputer_dict_ = X[self.variables_].mean().to_dict()

        elif self.imputation_method == "median":
            self.imputer_dict_ = X[self.variables_].median().to_dict()

        self._get_feature_names_in(X)

        return self
