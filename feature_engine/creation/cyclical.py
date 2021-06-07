from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class CyclicalTransformer(BaseNumericalTransformer):
    """
    The CyclicalTransformer() applies cyclical transformations to numerical
    variables. The transformations returns 2 new features per variable, according to:

    - var_sin = sin(variable * (2. * pi / max_value))
    - var_cos = cos(variable * (2. * pi / max_value))

    where max_value is the maximum value in the variable, and pi is 3.14...

    **Motivation**: There are some features that are cyclic by nature. For example the
    hours of a day or the months in a year. In these cases, the higher values of
    the variable are closer to the lower values. For example, December (12) is closer
    to January (1) than to June (6). By applying a cyclical transformation we capture
    this cycle or proximity between values.

    The CyclicalTransformer() works only with numerical variables. Missing data should
    be imputed before applying this transformer.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all numerical variables.

    Parameters
    ----------
    variables: list, default=None
        The list of numerical variables to transform. If None, the transformer will
        automatically find and select all numerical variables.
    max_values: dict, default=None
        A dictionary with the maximum value of each variable to transform. Useful when
        the maximum value is not present in the dataset. If None, the transformer will
        automatically find the maximum value of each variable.
    drop_original: bool, default=False
        If True, the original variables to transform will be dropped from the dataframe.

    Attributes
    ----------
    max_values_:
        The maximum value of the cyclical feature.

    variables_:
        The group of variables that will be transformed.

    n_features_in_:
        The number of features in the train set used in fit.


    Methods
    -------
    fit:
        Learns the maximum values of the cyclical features.
    transform:
        Applies the cyclical transformation, creates 2 new features.
    fit_transform:
        Fit to data, then transform it.


    References
    ----------
    http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        max_values: Optional[Dict[str, Union[int, float]]] = None,
        drop_original: Optional[bool] = False,
    ) -> None:

        if max_values:
            if not isinstance(max_values, dict) or not all(
                isinstance(var, (int, float)) for var in list(max_values.values())
            ):
                raise TypeError(
                    "max_values takes a dictionary of strings as keys, "
                    "and numbers as items to be used as the reference for"
                    "the max value of each column."
                )

        if not isinstance(drop_original, bool):
            raise TypeError("drop_original takes only boolean values True and False.")

        self.variables = _check_input_parameter_variables(variables)
        self.max_values = max_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learns the maximum value of each of the cyclical variables.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame.
        ValueError:
            - If some of the columns contains NaNs.
            - If some of the mapping keys are not present in variables.

        Returns
        -------
        self
        """

        # check input dataframe
        X = super().fit(X)

        if self.max_values is None:
            self.max_values_ = X[self.variables_].max().to_dict()
        else:
            for key in list(self.max_values.keys()):
                if key not in self.variables_:
                    raise ValueError(
                        f"The mapping key {key} is not present" f" in variables."
                    )
            self.max_values_ = self.max_values

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame):
        """
        Creates new features using the cyclical transformation.

        Parameters
        ----------
        X: Pandas DataFrame of shame = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not Pandas DataFrame.

        Returns
        -------
        X: Pandas dataframe.
            The dataframe with the additional new features. The original variables will
            be dropped if drop_originals is False, or retained otherwise.
        """
        X = super().transform(X)

        for variable in self.variables_:
            max_value = self.max_values_[variable]
            X[f"{variable}_sin"] = np.sin(X[variable] * (2.0 * np.pi / max_value))
            X[f"{variable}_cos"] = np.cos(X[variable] * (2.0 * np.pi / max_value))

        if self.drop_original:
            X.drop(columns=self.variables_, inplace=True)

        return X
