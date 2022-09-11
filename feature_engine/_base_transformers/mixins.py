from typing import Dict, List, Union, Optional

import pandas as pd

from feature_engine._variable_handling.variable_type_selection import (
    _find_or_check_numerical_variables,
)
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    check_X,
)

from numpy import ndarray
from numpy.typing import ArrayLike


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


class GetFeatureNamesOutMixin:
    def get_feature_names_out(
        self,
        input_features: Union[List[Union[str, int]], ArrayLike] = None,
    ) -> List[Union[str, int]]:
        """
        Get output feature names for transformation. In other words, returns the
        variable names of transformed dataframe.

        Parameters
        ----------
        input_features : array-like of str or None, default=None

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Raises
        ------
        ValueError
            If input_features is not a list or any of the features in input_features are
            not transformed by this transformer.

        Returns
        -------
        feature_names: list
            The name of the features.
        """
        if input_features is not None:
            msg = "input_features is not equal to feature_names_in_"
            if isinstance(input_features, list):
                if input_features != self.feature_names_in_:
                    raise ValueError(msg)
            elif isinstance(input_features, ndarray):
                if list(input_features) != self.feature_names_in_:
                    raise ValueError(msg)
            else:
                raise ValueError(
                    "input_features must be a list or an array. "
                    "Got {input_features} instead."
                )

        return self.feature_names_in_
