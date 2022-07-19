from typing import List, Union

import numpy as np
import pandas as pd

from feature_engine.dataframe_checks import check_X_y

from feature_engine.selection.base_selector import BaseSelector
from feature_engine.encoding import WoEEncoder
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_categorical_variables,
)


class InformationValue(BaseSelector):
    """

    """

    def __init__(
            self,
            variables: Union[None, int, str, List[Union[str, int]]] = None,
            confirm_variables: bool = False,
    )

        super().__init__(confirm_variables)
        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Learn the information value.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y: pandas series.
            Target, must be binary.

        """

        X, y = check_X_y(X, y)

        if y.nunique() != 2:
            raise ValueError(
                "This selector is designed for binary classification. The target "
                "used has more than 2 unique values."
            )