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
        Apply the BoxCox transformation.
    fit_transform:
        Fit to data, then transform it.

    References
    ----------
    .. [1] Box and Cox. "An Analysis of Transformations". Read at a RESEARCH MEETING,
        1964.
        https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1964.tb00553.x
    """

    def __init__(
        self, variables: Union[None, int, str, List[Union[str, int]]] = None
    ) -> None:
        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.copy()
        for variable in self.variables:
            # check if the varible has nan:
            if X[variable].isna().sum() > 0:
                raise NotImplementedError(f'The transformer CiclycalTransformer does not allow to have NaN values,'
                                          f'please check the column {variable}')
            max_value = X[variable].max()
            X[f'{variable}_sin'] = np.sin(X[variable] * (2. * np.pi / max_value))
            X[f'{variable}_cos'] = np.cos(X[variable] * (2. * np.pi / max_value))
        return X.drop(self.variables, axis=1)