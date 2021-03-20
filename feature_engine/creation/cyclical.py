from typing import List, Optional, Union
import numpy as np
import pandas as pd


from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables
from feature_engine.dataframe_checks import _check_contains_na


class CyclicalTransformer(BaseNumericalTransformer):
    """
    The CyclicalTransformer() applies a Ciclycal transformation to numerical
    variables.

    There are some feature that are cyclic by nature. One example of this are
    the hours of a day or the months of a year. In both cases the higher values of
    a set of data are closer to the lower values of that set.

    For example the month October (10) is closer to month January (1) than
    to month February (2).

    To check an explanation about this:
    http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

    The CyclicalTransformer() works only with numerical variables. But does not
    allow null values on any row of original column

    A list of variables can be passed as an argument. Alternatively, the
    transformer will automatically select and transform all numerical
    variables.

    Parameters
    ----------
    variables : list, default=None
        The list of numerical variables that will be transformed. If None, the
        transformer will automatically find and select all numerical variables.

    Methods
    -------
    fit:
        This transformer does not learn parameters.
    transform:
        Apply the CyclicalTransformer transformation.
    fit_transform:
        Fit to data, then transform it.

    References
    ----------
    ..
    """

    def __init__(
            self, variables: Union[None, int, str, List[Union[str, int]]] = None,
            drop_original: bool = False
    ) -> None:
        self.variables = _check_input_parameter_variables(variables)
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learns the max_value of each of the numerical variables.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y : pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame.
        ValueError:
            If some of the columns contains NaNs

        Returns
        -------
        self
        """

        # check input dataframe
        X = super().fit(X)
        _check_contains_na(X, self.variables)

        self.input_shape_ = X.shape

        # check for nans
        self.max_values_ = X[self.variables].max().to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        """
        Apply a Ciclycal transformation.

        Parameters
        ----------
        X : Pandas DataFrame of shame = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not Pandas DataFrame.
        ValueError:
            If some of the columns contains NaNs
        """
        X = super().transform(X)

        for variable in self.variables:
            max_value = self.max_values_[variable]
            X[f'{variable}_sin'] = np.sin(X[variable] * (2. * np.pi / max_value))
            X[f'{variable}_cos'] = np.cos(X[variable] * (2. * np.pi / max_value))
            if self.drop_original:
                X.drop(columns=variable, inplace=True)

        return X
