# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.variable_manipulation import _define_variables


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

    variables : list, default=None
        The list of numerical variables that will be transformed. If None, the
        transformer will automatically find and select all numerical variables.

    exp : float or int, default=0.5
        The power (or exponent).
    """

    def __init__(
        self, exp: Union[float, int] = 0.5, variables: Union[List[str], str] = None
    ):

        if not isinstance(exp, (float, int)):
            raise ValueError("exp must be a float or an int")

        self.exp = exp
        self.variables = _define_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits the power transformation.

        Args:
            X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.

            y: It is not needed in this transformer. Defaults to None.
            Alternatively takes Pandas Series.

        Returns:
            self
        """

        # check input dataframe
        X = super().fit(X)

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the power transformation to the variables.

        Args:
            X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Returns:
            The dataframe with the power transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform
        X.loc[:, self.variables] = np.power(X.loc[:, self.variables], self.exp)

        return X
