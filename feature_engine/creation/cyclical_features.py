from typing import Dict, List, Optional, Union

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._base_transformers.mixins import (
    FitFromDictMixin,
    GetFeatureNamesOutMixin,
)
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_param_drop_original,
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_input_dictionary import (
    _check_numerical_dict,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine.dataframe_helpers import (
    add_columns_to_df,
    drop_variables_from_df,
    to_numpy,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _drop_original_docstring,
    _return_empty_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _transform_creation_docstring,
)
from feature_engine._docstrings.substitute import Substitution


@Substitution(
    variables=_variables_numerical_docstring,
    drop_original=_drop_original_docstring,
    return_empty=_return_empty_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    transform=_transform_creation_docstring,
)
class CyclicalFeatures(
    BaseNumericalTransformer, FitFromDictMixin, GetFeatureNamesOutMixin
):
    """
    CyclicalFeatures() applies cyclical transformations to numerical
    variables, returning 2 new features per variable, according to:

    - var_sin = sin(variable * (2. * pi / max_value))
    - var_cos = cos(variable * (2. * pi / max_value))

    where max_value is the maximum value in the variable, and pi is 3.14...

    CyclicalFeatures() works only with numerical variables. A list of variables
    to transform can be passed as an argument. Alternatively, the transformer will
    automatically select and transform all numerical variables.

    Missing data should be imputed before using this transformer.

    More details in the :ref:`User Guide <cyclical_features>`.

    Parameters
    ----------
    {variables}

    {return_empty}

    max_values: dict, default=None
        A dictionary with the maximum value of each variable to transform. Useful when
        the maximum value is not present in the dataset. If None, the transformer will
        automatically find the maximum value of each variable.

    {drop_original}

    Attributes
    ----------
    max_values_:
        The feature's maximum values.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learns the variable's maximum values.

    {fit_transform}

    {transform}

    References
    ----------
    Debaditya Chakraborty & Hazem Elzarka (2019), Advanced machine learning techniques
    for building performance simulation: a comparative analysis, Journal of Building
    Performance Simulation, 12:2, 193-207

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.creation import CyclicalFeatures
    >>> X = pd.DataFrame(dict(x= [1,4,3,3,4,2,1,2]))
    >>> cf = CyclicalFeatures()
    >>> cf.fit(X)
    >>> cf.transform(X)
       x         x_sin         x_cos
    0  1  1.000000e+00  6.123234e-17
    1  4 -2.449294e-16  1.000000e+00
    2  3 -1.000000e+00 -1.836970e-16
    3  3 -1.000000e+00 -1.836970e-16
    4  4 -2.449294e-16  1.000000e+00
    5  2  1.224647e-16 -1.000000e+00
    6  1  1.000000e+00  6.123234e-17
    7  2  1.224647e-16 -1.000000e+00

    With polars:

    >>> import polars as pl
    >>> from feature_engine.creation import CyclicalFeatures
    >>> X = pl.DataFrame({"x": [1, 4, 3, 3, 4, 2, 1, 2]})
    >>> cf = CyclicalFeatures()
    >>> cf.fit(X)
    >>> cf.transform(X)
    shape: (8, 3)
    ┌─────┬─────────────┬─────────────┐
    │ x   ┆ x_sin       ┆ x_cos       │
    │ --- ┆ ---         ┆ ---         │
    │ i64 ┆ f64         ┆ f64         │
    ╞═════╪═════════════╪═════════════╡
    │ 1   ┆ 1.0         ┆ 6.1232e-17  │
    │ 4   ┆ -2.4493e-16 ┆ 1.0         │
    │ 3   ┆ -1.0        ┆ -1.8370e-16 │
    │ 3   ┆ -1.0        ┆ -1.8370e-16 │
    │ 4   ┆ -2.4493e-16 ┆ 1.0         │
    │ 2   ┆ 1.2246e-16  ┆ -1.0        │
    │ 1   ┆ 1.0         ┆ 6.1232e-17  │
    │ 2   ┆ 1.2246e-16  ┆ -1.0        │
    └─────┴─────────────┴─────────────┘
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        max_values: Optional[Dict[str, Union[int, float]]] = None,
        drop_original: Optional[bool] = False,
    ) -> None:

        _check_numerical_dict(max_values)
        _check_param_drop_original(drop_original)
        _check_return_empty_is_bool(return_empty)

        self.variables = _check_variables_input_value(variables)
        self.return_empty = return_empty
        self.max_values = max_values
        self.drop_original = drop_original

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        Learns the maximum value of each variable.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """
        if self.max_values is None:
            X, variables_ = self._fit_setup(X)
            # .item() returns a plain Python int or float, not a numpy scalar
            max_values_: dict = {
                var: values.max().item()
                for var, values in zip(variables_, to_numpy(X, variables_))
            }
        else:
            X, variables_ = super()._fit_from_dict(X, self.max_values)
            max_values_ = self.max_values

        self.variables_ = variables_
        self.max_values_ = max_values_
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Creates new features using the cyclical transformations.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: dataframe.
            The original dataframe plus the additional features.
        """
        X = self._check_transform_input_and_state(X)

        new_features = {}
        arrays = to_numpy(X, self.variables_)
        for variable, values in zip(self.variables_, arrays):
            scaled = values * (2.0 * np.pi / self.max_values_[variable])
            new_features[f"{variable}_sin"] = np.sin(scaled)
            new_features[f"{variable}_cos"] = np.cos(scaled)
        X = add_columns_to_df(X, new_features)
        if self.drop_original is True:
            X = drop_variables_from_df(X, self.variables_)

        return nw.to_native(X, pass_through=True)

    def _get_new_features_name(self) -> List:
        """Return names of the created features."""
        feature_names = [
            f"{var}_{suffix}" for var in self.variables_ for suffix in ["sin", "cos"]
        ]
        return feature_names
