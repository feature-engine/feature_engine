# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine._check_init_parameters.check_input_dictionary import (
    _check_numerical_dict,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _imputer_dict_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
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
    imputer_dict_=_imputer_dict_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    transform=_transform_imputers_docstring,
    fit_transform=_fit_transform_docstring,
)
class ArbitraryNumberImputer(BaseImputer):
    """
    The ArbitraryNumberImputer() replaces missing data by an arbitrary
    value determined by the user. It works only with numerical variables.

    You can impute all variables with the same number by defining
    the variables to impute in `variables` and the imputation number in
    `arbitrary_number`. Alternatively, you can pass a dictionary with the variable
    names and the numbers to use for their imputation in the `imputer_dict` parameter.

    More details in the :ref:`User Guide <arbitrary_number_imputer>`.

    Parameters
    ----------
    arbitrary_number: int or float, default=999
        The number to replace the missing data. This parameter is used only if
        `imputer_dict` is None.

    variables: list, default=None
        The list of variables to impute. If None, the imputer will
        select all numerical variables. This parameter is used only if `imputer_dict`
        is None.

    imputer_dict: dict, default=None
        The dictionary of variables and the arbitrary numbers for their imputation. If
        specified, it overrides the above parameters.


    Attributes
    ----------

    {imputer_dict_}

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {transform}

    See Also
    --------
    feature_engine.imputation.EndTailImputer

    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> from feature_engine.imputation import ArbitraryNumberImputer
    >>> X = pd.DataFrame(dict(
    >>>        x1 = [np.nan,1,1,0,np.nan],
    >>>        x2 = ["a", np.nan, "b", np.nan, "a"],
    >>>       ))
    >>> ani = ArbitraryNumberImputer(arbitrary_number=-999)
    >>> ani.fit(X)
    >>> ani.transform(X)
          x1   x2
    0 -999.0    a
    1    1.0  NaN
    2    1.0    b
    3    0.0  NaN
    4 -999.0    a
    """

    def __init__(
        self,
        arbitrary_number: Union[int, float] = 999,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        imputer_dict: Optional[dict] = None,
    ) -> None:

        if isinstance(arbitrary_number, int) or isinstance(arbitrary_number, float):
            self.arbitrary_number = arbitrary_number
        else:
            raise ValueError("arbitrary_number must be numeric of type int or float")

        _check_numerical_dict(imputer_dict)

        self.variables = _check_variables_input_value(variables)

        self.imputer_dict = imputer_dict

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This method does not learn any parameter.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: None
            y is not needed in this imputation. You can pass None or y.
        """

        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        # create the imputer dictionary
        if self.imputer_dict:
            self.variables_ = check_numerical_variables(
                X, list(self.imputer_dict.keys())
            )
            self.imputer_dict_ = self.imputer_dict
        else:
            if self.variables is None:
                self.variables_ = find_numerical_variables(X)
            else:
                self.variables_ = check_numerical_variables(X, self.variables)
            self.imputer_dict_ = {var: self.arbitrary_number for var in self.variables_}

        self._get_feature_names_in(X)

        return self
