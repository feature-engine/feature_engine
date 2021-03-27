from typing import List, Optional, Union, Dict
import numpy as np
import pandas as pd


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
    max_values: dict(str, Union)
        A dictionary that maps the natural maximum or a variable. Useful when
        the maximum value is not present in the dataset.
    drop_original: bool, default=False
        Use this if you want to drop the original columns from the output.


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
    http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    """

    def __init__(
            self, variables: Union[None, int, str, List[Union[str, int]]] = None,
            max_values: Optional[Dict[str, Union[int, float]]] = None,
            drop_original: Optional[bool] = False
    ) -> None:
        self.variables = _check_input_parameter_variables(variables)
        self.max_values = self._check_max_values(max_values)
        self.drop_original = self._check_drop_original(drop_original)

    def _check_max_values(self, max_values):
        if max_values:
            if not isinstance(max_values, dict) or not all(
                    isinstance(var, (int, float)) for var in list(max_values.values())):
                raise TypeError(
                    'max_values takes a dictionary of strings as keys, '
                    'and numbers as items to be used as the reference for'
                    'the max value of each column.'
                )
        return max_values

    def _check_drop_original(self, drop_original):
        if drop_original:
            if not isinstance(drop_original, bool):
                raise TypeError(
                    'drop_original takes a boolean value in order to know'
                    'if the variable(s) are going to be deleted.'
                )
        return drop_original

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
        Apply a Ciclycal transformation.

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
            Depending if drop_original was set to True the dataframe will have
            new columns which are twice the amount of columns in variables. If
            set to False it will have twice the amount of columns in variables
            and the original columns.
        """
        X = super().transform(X)

        for variable in self.variables:
            max_value = self.max_values_[variable]
            X[f'{variable}_sin'] = np.sin(X[variable] * (2. * np.pi / max_value))
            X[f'{variable}_cos'] = np.cos(X[variable] * (2. * np.pi / max_value))
            if self.drop_original:
                X.drop(columns=variable, inplace=True)

        return X
