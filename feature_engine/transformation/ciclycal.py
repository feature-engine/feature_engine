from typing import List, Optional, Union
import numpy as np
import pandas as pd
import scipy.stats as stats

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


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
            self, variables: Union[None, int, str, List[Union[str, int]]] = None
    ) -> None:
        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter. But keeps a map of the max values
        for every column

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
            If the input is not a Pandas DataFrame

        Returns
        -------
        self
        """

        # check input dataframe
        X = super().fit(X)

        # check for nans
        self.max_values = {}
        for variable in self.variables:
            if X[variable].isna().sum() > 0:
                raise ValueError(f'The transformer CiclycalTransformer does not allow to have NaN values,'
                                 f'please check the column {variable}')
            self.max_values[variable] = X[variable].max()
        return self

    def transform(self, X: pd.DataFrame):
        """
        Apply a Ciclycal transformation.

        Parameters
        ----------
        X : Pandas DataFrame of shame = [n_samples, n_features]
            The data to be transformed.
        """
        # TODO aqui quedé el check_fitted no está pasando
        X = super().transform(X)
        #X = X.copy() # This works


        for variable in self.variables:
            max_value = self.max_values[variable]
            X[f'{variable}_sin'] = np.sin(X[variable] * (2. * np.pi / max_value))
            X[f'{variable}_cos'] = np.cos(X[variable] * (2. * np.pi / max_value))
            X.drop(columns=variable, inplace=True)

        return X
