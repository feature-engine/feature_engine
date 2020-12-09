# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class LogTransformer(BaseNumericalTransformer):
    """
    The LogTransformer() applies the natural logarithm or the base 10 logarithm to
    numerical variables. The natural logarithm is logarithm in base e.

    The LogTransformer() only works with numerical non-negative values. If the variable
    contains a zero or a negative value, the transformer will return an error.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type numeric.

    Parameters
    ----------
    base: string, default='e'
        Indicates if the natural or base 10 logarithm should be applied. Can take
        values 'e' or '10'.

    variables : list, default=None
        The list of numerical variables to be transformed. If None, the transformer
        will find and select all numerical variables.

    Methods
    -------
    fit:
        This transformer does not learn parameters.
    transform:
        Transforms the variables using log transformation.
    fit_transform:
        Fit to data, then transform it.
    """

    def __init__(
        self,
        base: str = "e",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if base not in ["e", "10"]:
            raise ValueError("base can take only '10' or 'e' as values")

        self.variables = _check_input_parameter_variables(variables)
        self.base = base

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Select the numerical variables and determines whether the logarithm
        can be applied on the selected variables (it checks if the variables
        are all positive).

        Parameters
        ----------
        X : Pandas DataFrame of shape = [n_samples, n_features].
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y : pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame
            - If any of the user provided variables are not numerical
        ValueError
            - If there are no numerical variables in the df or the df is empty
            - If the variable(s) contain null values
            - If some variables contain zero or negative values

        Returns
        -------
        self
        """

        # check input dataframe
        X = super().fit(X)

        # check contains zero or negative values
        if (X[self.variables] <= 0).any().any():
            raise ValueError(
                "Some variables contain zero or negative values, can't apply log"
            )

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the variables using log transformation.

        Parameters
        ----------
        X : Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values.
            - If the dataframe not of the same size as that used in fit().
            - If some variables contains zero or negative values.

        Returns
        -------
        X : pandas dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # check contains zero or negative values
        if (X[self.variables] <= 0).any().any():
            raise ValueError(
                "Some variables contain zero or negative values, can't apply log"
            )

        # transform
        if self.base == "e":
            X.loc[:, self.variables] = np.log(X.loc[:, self.variables])
        elif self.base == "10":
            X.loc[:, self.variables] = np.log10(X.loc[:, self.variables])

        return X
