# Authors: Vasco Schiavo <vasco.schiavo@protonmail.com>
# License: BSD 3 clause

from typing import Dict, List, Optional, Union

import pandas as pd

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _inverse_transform_docstring,
)
from feature_engine._docstrings.substitute import Substitution


@Substitution(
    variables=_variables_numerical_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class MeanNormalizationScaling(BaseNumericalTransformer):
    """
    The MeanNormalizationScaling() applies the mean normalization scaling techinques
    to one or multuple columns of a dataframe.

    Mean normalization is a way to implement feature scaling. Mean normalization
    calculates and subtracts the mean of for every feature and divide this value
    by the range (max - min).

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type numeric.

    If a column is constant, then it will be transformed to the zero constant column.

    Parameters
    ----------
    {variables}


    Attributes
    ----------
    params:
        a dictionary containing the mean, max and min of every given variable

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {inverse_transform}

    transform:
        Scale the variables using mean normalization.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.scaling import MeanNormalizationScaling
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.lognormal(size = 100)))
    >>> mns = LogTransformer()
    >>> mns.fit(X)
    >>> X = mns.transform(X)
    >>> X.head()
            x
    0  0.496714
    1 -0.138264
    2  0.647689
    3  1.523030
    4 -0.234153
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        self.variables = _check_variables_input_value(variables)
        # the following variable will be populated in the during the fit
        self.params: Dict = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        The method populate the variable `params`.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features].
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = super().fit(X)

        for variable in self.variables_:
            self.params[variable] = {
                "mean": X[variable].mean(),
                "max": X[variable].max(),
                "min": X[variable].min(),
            }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the variables using mean normalization.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        # transformation
        for variable in self.variables_:
            numerator = X[variable] - self.params[variable]["mean"]
            denominator = self.params[variable]["max"] - self.params[variable]["min"]

            # If max and min are equal, then the column's variable is constant.
            # We set denominator to 1, and the column will be normalized to 0
            if denominator == 0:
                denominator = 1

            X[variable] = numerator / denominator

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_tr: pandas dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        # inverse_transform
        for variable in self.variables_:

            if self.params[variable]["min"] == self.params[variable]["max"]:
                denominator = 1
            else:
                denominator = (
                    self.params[variable]["max"] - self.params[variable]["min"]
                )
            X[variable] = X[variable] * denominator + self.params[variable]["mean"]

        return X
