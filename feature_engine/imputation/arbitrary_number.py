# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

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
    imputer_dict_:
        Dictionary with the values to replace NAs in each variable.

    variables_:
        The group of variables that will be transformed.

    n_features_in_:
        The number of features in the train set used in fit.

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
        This method does not learn any parameter.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: None
            y is not needed in this imputation. You can pass None or y.
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        # create the imputer dictionary
        if self.imputer_dict:
            self.variables_ = _find_or_check_numerical_variables(
                X, self.imputer_dict.keys()  # type: ignore
            )
            self.imputer_dict_ = self.imputer_dict
        else:
            self.variables_ = _find_or_check_numerical_variables(X, self.variables)
            self.imputer_dict_ = {var: self.arbitrary_number for var in self.variables_}

        self.n_features_in_ = X.shape[1]

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseImputer.transform.__doc__
