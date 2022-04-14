# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _inverse_transform_docstring,
)
from feature_engine._docstrings.fit_attributes import (
    _variables_attribute_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.class_inputs import _variables_numerical_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.variable_manipulation import _check_input_parameter_variables


@Substitution(
    variables=_variables_numerical_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class PowerTransformer(BaseNumericalTransformer):
    """
    The PowerTransformer() applies power or exponential transformations to
    numerical variables.

    The PowerTransformer() works only with numerical variables.

    A list of variables can be passed as an argument. Alternatively, the
    transformer will automatically select and transform all numerical
    variables.

    More details in the :ref:`User Guide <power>`.

    Parameters
    ----------
    {variables}

    exp: float or int, default=0.5
        The power (or exponent).

    Attributes
    ----------
    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {inverse_transform}

    transform:
        Apply the power transformation to the variables.

    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        exp: Union[float, int] = 0.5,
    ):

        if not isinstance(exp, (float, int)):
            raise ValueError("exp must be a float or an int")

        self.exp = exp
        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        super()._fit_from_varlist(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the power transformation to the variables.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas Dataframe
            The dataframe with the power transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform
        X.loc[:, self.variables_] = np.power(X.loc[:, self.variables_], self.exp)

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
        X_tr: pandas Dataframe
            The dataframe with the power transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        # inverse_transform
        X.loc[:, self.variables_] = np.power(X.loc[:, self.variables_], 1 / self.exp)

        return X
