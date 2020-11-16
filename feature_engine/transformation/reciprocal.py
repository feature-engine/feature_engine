# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class ReciprocalTransformer(BaseNumericalTransformer):
    """
    The ReciprocalTransformer() applies the reciprocal transformation 1 / x
    to numerical variables.

    The ReciprocalTransformer() only works with numerical variables with non-zero
    values. If a variable contains the value 0, the transformer will raise an error.

    A list of variables can be passed as an argument. Alternatively, the
    transformer will automatically select and transform all numerical
    variables.

    Parameters
    ----------

    variables : list, default=None
        The list of numerical variables that will be transformed. If None, the
        transformer will automatically find and select all numerical variables.
    """

    def __init__(
        self, variables: Union[None, int, str, List[Union[str, int]]] = None
    ) -> None:

        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits the reciprocal transformation

        Args:
            X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.

            y: It is not needed in this transformer. Defaults to None.
            Alternatively takes Pandas Series.ss

        Raises:
            ValueError: If some variables contains zero

        Returns:
            self
        """

        # check input dataframe
        X = super().fit(X)

        # check if the variables contain the value 0
        if (X[self.variables] == 0).any().any():
            raise ValueError(
                "Some variables contain the value zero, can't apply reciprocal "
                "transformation."
            )

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the reciprocal 1 / x transformation.

        Args:
            X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to transform.

        Raises:
            ValueError: If some variables contain zero values.

        Returns:
            The dataframe with reciprocally transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # check if the variables contain the value 0
        if (X[self.variables] == 0).any().any():
            raise ValueError(
                "Some variables contain the value zero, can't apply reciprocal "
                "transformation."
            )

        # transform
        # for some reason reciprocal does not work with integers
        X.loc[:, self.variables] = X.loc[:, self.variables].astype("float")
        X.loc[:, self.variables] = np.reciprocal(X.loc[:, self.variables])

        return X
