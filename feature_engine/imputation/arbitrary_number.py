# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Optional, List, Union

import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.parameter_checks import _define_numerical_dict
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)


class ArbitraryNumberImputer(BaseImputer):
    """
    The ArbitraryNumberImputer() replaces missing data in each variable by an arbitrary
    value determined by the user. It works only with numerical variables.

    We can impute all variables with the same number, in which case we need to define
    the variables to impute in `variables` and the imputation number in
    `arbitrary_number`. Alternatively, we can pass a dictionary of variable and numbers
    to use for their imputation.

    For example, we can impute varA and varB with 99 like this:

    .. code-block:: python

        transformer = ArbitraryNumberImputer(
                variables = ['varA', 'varB'],
                arbitrary_number = 99
                )

        Xt = transformer.fit_transform(X)

    Alternatively, we can impute varA with 1 and varB with 99 like this:

    .. code-block:: python

        transformer = ArbitraryNumberImputer(
                imputer_dict = {'varA' : 1, 'varB': 99]
                )

        Xt = transformer.fit_transform(X)

    Parameters
    ----------
    arbitrary_number : int or float, default=999
        The number to be used to replace missing data.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all numerical type variables. This parameter is used only if
        `imputer_dict` is None.

    imputer_dict : dict, default=None
        The dictionary of variables and the arbitrary numbers for their imputation.

    Attributes
    ----------
    imputer_dict_ :
        Dictionary with the values to replace NAs in each variable.

    Methods
    -------
    fit:
        This transformer does not learn parameters.
    transform:
        Impute missing data.
    fit_transform:
        Fit to the data, then transform it.

    See Also
    --------
    feature_engine.imputation.EndTailImputer
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

        self.variables = _check_input_parameter_variables(variables)

        self.imputer_dict = _define_numerical_dict(imputer_dict)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This method does not learn any parameter. Checks dataframe and finds numerical
        variables, or checks that the variables entered by user are numerical.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y : None
            y is not needed in this imputation. You can pass None or y.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame
            - If any of the user provided variables are not numerical
        ValueError
            If there are no numerical variables in the df or the df is empty

        Returns
        -------
        self
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        if self.imputer_dict:
            self.variables = _find_or_check_numerical_variables(
                X, self.imputer_dict.keys()  # type: ignore
            )
        else:
            self.variables = _find_or_check_numerical_variables(X, self.variables)

        # create the imputer dictionary
        if self.imputer_dict:
            self.imputer_dict_ = self.imputer_dict
        else:
            self.imputer_dict_ = {var: self.arbitrary_number for var in self.variables}

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseImputer.transform.__doc__
