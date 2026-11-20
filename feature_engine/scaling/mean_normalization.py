# Authors: Vasco Schiavo <vasco.schiavo@protonmail.com>
# License: BSD 3 clause

import warnings
from typing import List, Optional, Union

import narwhals as nw
from narwhals.typing import IntoDataFrame, IntoSeries

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _return_empty_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _inverse_transform_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_helpers import replace_columns_in_df, to_numpy


@Substitution(
    variables=_variables_numerical_docstring,
    return_empty=_return_empty_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class MeanNormalisationScaler(BaseNumericalTransformer):
    """
    MeanNormalisationScaler() applies mean normalisation, which consists of subtracting
    the mean of each feature and then dividing the result by the value range, that is,
    the difference between its maximum and minimum value. The method aims to center the
    variables at 0, and rescale the distribution between -1 and 1.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type numeric.

    Constant variables will raise an error due to division by zero.

    More details in the :ref:`User Guide <mean_normalisation_scaler>`.

    Parameters
    ----------
    {variables}

    {return_empty}

    Attributes
    ----------
    mean_:
        Dictionary containing the mean of the variables.

    range_:
        Dictionary containing the value range of the variables.

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
        Scale the variables using mean normalisation.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.scaling import MeanNormalisationScaler
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.lognormal(size = 100)))
    >>> mns = MeanNormalisationScaler()
    >>> mns.fit(X)
    >>> X = mns.transform(X)
    >>> X.head()
              x
    0  0.051125
    1 -0.071456
    2  0.093623
    3  0.518122
    4 -0.084093

    With polars:

    >>> import numpy as np
    >>> import polars as pl
    >>> from feature_engine.scaling import MeanNormalisationScaler
    >>> np.random.seed(42)
    >>> X = pl.DataFrame(dict(x = np.random.lognormal(size = 100)))
    >>> mns = MeanNormalisationScaler()
    >>> mns.fit(X)
    >>> X = mns.transform(X)
    >>> X.head()
    shape: (5, 1)
    ┌───────────┐
    │ x         │
    │ ---       │
    │ f64       │
    ╞═══════════╡
    │ 0.051125  │
    │ -0.071456 │
    │ 0.093623  │
    │ 0.518122  │
    │ -0.084093 │
    └───────────┘
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
    ) -> None:

        _check_return_empty_is_bool(return_empty)

        self.variables = _check_variables_input_value(variables)
        self.return_empty = return_empty

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Finds the mean and value range of each variable.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features].
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """

        X, variables_ = self._fit_setup(X)

        arrays = to_numpy(X, variables_)
        # .item() returns plain Python numbers, not numpy scalars.
        mean_: dict = {
            var: values.mean().item() for var, values in zip(variables_, arrays)
        }
        range_: dict = {
            var: (values.max() - values.min()).item()
            for var, values in zip(variables_, arrays)
        }

        # check for constant columns
        constant_columns = [col for col, value in range_.items() if value == 0]
        if constant_columns:
            raise ValueError(
                f"The following variable(s) are constant: {constant_columns}. "
                "Division by zero is not allowed. Please remove constant columns."
            )

        self.variables_ = variables_
        self.mean_ = mean_
        self.range_ = range_
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Transform the variables using mean normalisation.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: dataframe
            The dataframe with the transformed variables.
        """

        X = self._check_transform_input_and_state(X)
        arrays = to_numpy(X, self.variables_)
        X = replace_columns_in_df(
            X,
            {
                var: (values - self.mean_[var]) / self.range_[var]
                for var, values in zip(self.variables_, arrays)
            },
        )
        return nw.to_native(X, pass_through=True)

    def inverse_transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_tr: dataframe
            The dataframe with the transformed variables.
        """

        X = self._check_transform_input_and_state(X)
        arrays = to_numpy(X, self.variables_)
        X = replace_columns_in_df(
            X,
            {
                var: values * self.range_[var] + self.mean_[var]
                for var, values in zip(self.variables_, arrays)
            },
        )
        return nw.to_native(X, pass_through=True)


# TODO: remove in version 2.1.0
class MeanNormalizationScaler(MeanNormalisationScaler):
    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
    ) -> None:
        warnings.warn(
            "MeanNormalizationScaler was deprecated in favour of "
            "MeanNormalisationScaler in version 2.0.0 and will be removed in "
            "version 2.1.0. To silence this warning, use MeanNormalisationScaler "
            "instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(variables=variables, return_empty=return_empty)
