from typing import Dict

import pandas as pd

from feature_engine._variable_handling.variable_type_selection import (
    _find_or_check_numerical_variables,
)
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    check_X,
)


class FitFromDictMixin:
    def _fit_from_dict(self, X: pd.DataFrame, user_dict_: Dict) -> pd.DataFrame:
        """
        Checks that input is a dataframe, checks that variables in the dictionary
        entered by the user are of type numerical.

        Parameters
        ----------
        X : Pandas DataFrame

        user_dict_ : Dictionary. Default = None
            Any dictionary allowed by the transformer and entered by user.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame or a numpy array
            If any of the variables in the dictionary are not numerical
        ValueError
            If there are no numerical variables in the df or the df is empty
            If the variable(s) contain null values

        Returns
        -------
        X : Pandas DataFrame
            The same dataframe entered as parameter
        """
        # check input dataframe
        X = check_X(X)

        # find or check for numerical variables
        variables = [x for x in user_dict_.keys()]
        self.variables_ = _find_or_check_numerical_variables(X, variables)

        # check if dataset contains na or inf
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        # save input features
        self.feature_names_in_ = X.columns.tolist()

        # save train set shape
        self.n_features_in_ = X.shape[1]

        return X
