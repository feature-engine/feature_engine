from typing import List, Optional, Union, Dict
import numpy as np
import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class CyclicalTransformer(BaseNumericalTransformer):
    """
    The CyclicalTransformer() applies a cyclical transformation to numerical
    variables.

    There are some features that are cyclic by nature. Examples of this are
    the hours of a day or the months of a year. In both cases, the higher values of
    a set of data are closer to the lower values of that set. For example, December
    (12) is closer to January (1) than to June (6).

    The CyclicalTransformer() works only with numerical variables. Missing data should
    be imputed before applying this transformer.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all numerical variables.

    Parameters
    ----------
    variables : list, default=None
        The list of numerical variables that will be transformed. If None, the
        transformer will automatically find and select all numerical variables.
    max_values: dict, default=None
        A dictionary that maps the natural maximum or a variable. Useful when
        the maximum value is not present in the dataset.
    drop_original: bool, default=False
        Use this if you want to drop the original variables from the output.


    Attributes
    ----------
    max_values_ :
        The maximum value of the cylcical feature that will be used for the
        transformation.


    Methods
    -------
    fit:
        Learns the maximum values of the cyclical features.
    transform:
        Apply the cyclical transformation transformation, crates 2 new features.
    fit_transform:
        Fit to data, then transform it.


    References
    ----------
    http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    """

    def __init__(
            self, variables: Union[None, int, str, List[Union[str, int]]] = None,
            max_values: Optional[Dict[str, Union[int, float]]] = None,
            drop_original: Optional[bool] = False
    ) -> None:

        if max_values:
            if not isinstance(max_values, dict) or not all(
                    isinstance(var, (int, float)) for var in list(max_values.values())):
                raise TypeError(
                    'max_values takes a dictionary of strings as keys, '
                    'and numbers as items to be used as the reference for'
                    'the max value of each column.'
                )

        if not isinstance(drop_original, bool):
            raise TypeError(
                'drop_original takes only boolean values True and False.'
            )

        self.variables = _check_input_parameter_variables(variables)
        self.max_values = max_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learns the maximmum value of each of the cyclical variables.

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
            - If some of the columns contains NaNs.
            - If some of the mapping keys are not present in variables.

        Returns
        -------
        self
        """

        # check input dataframe
        X = super().fit(X)

        if self.max_values is None:
            self.max_values_ = X[self.variables].max().to_dict()
        else:
            for key in list(self.max_values.keys()):
                if key not in self.variables:
                    raise ValueError(f'The mapping key {key} is not present'
                                     f' in variables.')
            self.max_values_ = self.max_values

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame):
        """
        Crates new features using the cyiclical transformation.

        Parameters
        ----------
        X : Pandas DataFrame of shame = [n_samples, n_features]
            The data to be transformed.

        Raises
        ------
        TypeError
            If the input is not Pandas DataFrame.

        Returns
        -------
        X : Pandas dataframe.
            The dataframe with the original variables plus the new variables if
            drop_originals was False, alternatively, the original variables are
            removed from the dataset.
        """
        X = super().transform(X)

        for variable in self.variables:
            max_value = self.max_values_[variable]
            X[f'{variable}_sin'] = np.sin(X[variable] * (2. * np.pi / max_value))
            X[f'{variable}_cos'] = np.cos(X[variable] * (2. * np.pi / max_value))

        if self.drop_original:
            X.drop(columns=self.variables, inplace=True)

        return X
