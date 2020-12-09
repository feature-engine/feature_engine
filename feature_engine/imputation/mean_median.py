# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Optional, List, Union

import pandas as pd

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)


class MeanMedianImputer(BaseImputer):
    """
    The MeanMedianImputer() replaces missing data by the mean or median value of the
    variable. It works only with numerical variables.

    We can pass a list of variables to be imputed. Alternatively, the
    MeanMedianImputer() will automatically select all variables of type numeric in the
    training set.

    The imputer:

    - first calculates the mean / median values of the variables (fit).
    - Then replaces the missing data with the estimated mean / median (transform).


    Parameters
    ----------
    imputation_method : str, default=median
        Desired method of imputation. Can take 'mean' or 'median'.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will select
        all variables of type numeric.

    Attributes
    ----------
    imputer_dict_ :
        Dictionary with the mean or median values per variable.

    Methods
    -------
    fit:
        Learn the mean or median values.
    transform:
        Impute missing data.
    fit_transform:
        Fit to the data, then transform it.
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
        X : pandas dataframe of shape = [n_samples, n_features]
            The training dataset.

        y : pandas series or None, default=None
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
        self.variables = _find_or_check_numerical_variables(X, self.variables)

        # find imputation parameters: mean or median
        if self.imputation_method == "mean":
            self.imputer_dict_ = X[self.variables].mean().to_dict()

        elif self.imputation_method == "median":
            self.imputer_dict_ = X[self.variables].median().to_dict()

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseImputer.transform.__doc__
