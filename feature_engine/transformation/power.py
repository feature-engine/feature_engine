# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class PowerTransformer(BaseNumericalTransformer):
    """
    The PowerTransformer() applies power or exponential transformations to
    numerical variables.

    The PowerTransformer() works only with numerical variables.

    A list of variables can be passed as an argument. Alternatively, the
    transformer will automatically select and transform all numerical
    variables.

    Parameters
    ----------
    variables: list, default=None
        The list of numerical variables to transform. If None, the transformer will
        automatically find and select all numerical variables.

    exp: float or int, default=0.5
        The power (or exponent).

    Attributes
    ----------
    variables_:
        The group of variables that will be transformed.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        This transformer does not learn parameters.
    transform:
        Apply the power transformation to the variables.
    fit_transform:
        Fit to data, then transform it.
    inverse_transform:
        Convert the data back to the original representation.
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        exp: Union[float, int] = 0.5,
    ):

        if not isinstance(exp, (float, int)):
            raise ValueError("exp must be a float or an int")

        self.exp = exp
        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame
            - If any of the user provided variables are not numerical
        ValueError
            - If there are no numerical variables in the df or the df is empty
            - If the variable(s) contain null values

        Returns
        -------
        self
        """

        # check input dataframe
        X = super().fit(X)

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the power transformation to the variables.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values
            - If the df has different number of features than the df used in fit()

        Returns
        -------
        X: pandas Dataframe
            The dataframe with the power transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform
        X.loc[:, self.variables_] = np.power(X.loc[:, self.variables_], self.exp)

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values
            - If the df has different number of features than the df used in fit()

        Returns
        -------
        X: pandas Dataframe
            The dataframe with the power transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # inverse_transform
        X.loc[:, self.variables_] = np.power(X.loc[:, self.variables_], 1 / self.exp)

        return X
