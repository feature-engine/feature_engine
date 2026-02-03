# Authors: Vasco Schiavo <vasco.schiavo@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

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
    _fit_transform_docstring,
    _inverse_transform_docstring,
)
from feature_engine._docstrings.substitute import Substitution


@Substitution(
    variables=_variables_numerical_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class MeanNormalizationScaler(BaseNumericalTransformer):
    """
    MeanNormalizationScaler() applies mean normalization, which consists of subtracting
    the mean of each feature and then dividing the result by the value range, that is,
    the difference between its maximum and minimum value. The method aims to center the
    variables at 0, and rescale the distribution between -1 and 1.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatmypy featureically select and transform all variables of type numeric.

    Constant variables will raise an error due to division by zero.

    More details in the :ref:`User Guide <mean_normalization_scaler>`.


    Parameters
    ----------
    {variables}


    Attributes
    ----------
    mean_:
        Dictionary containing the mean of the variables.

    range_:
        Dictionary containing the value range of of the variables.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find variables' mean and value range.

    {fit_transform}

    {inverse_transform}

    transform:
        Scale the variables using mean normalization.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.scaling import MeanNormalizationScaler
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.lognormal(size = 100)))
    >>> mns = MeanNormalizationScaler()
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

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Finds the mean and value range of each variable.

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
        self.mean_ = X[self.variables_].mean().to_dict()
        self.range_ = (X[self.variables_].max() - X[self.variables_].min()).to_dict()

        # check for constant columns
        constant_columns = [col for col, value in self.range_.items() if value == 0]
        if constant_columns:
            raise ValueError(
                f"The following variable(s) are constant: {constant_columns}. "
                "Division by zero is not allowed. Please remove constant columns."
            )

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
        X[self.variables_] = (X[self.variables_] - self.mean_) / self.range_

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

        # inverse transform
        X[self.variables_] = X[self.variables_] * self.range_ + self.mean_

        return X
