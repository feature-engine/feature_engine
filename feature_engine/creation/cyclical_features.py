from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._base_transformers.mixins import (
    FitFromDictMixin,
    GetFeatureNamesOutMixin,
)
from feature_engine._check_input_parameters.check_init_input_params import (
    _check_param_drop_original,
)
from feature_engine._check_input_parameters.check_input_dictionary import (
    _check_numerical_dict,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters import (
    _drop_original_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _transform_creation_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine._variable_handling.init_parameter_checks import (
    _check_init_parameter_variables,
)


@Substitution(
    variables=_variables_numerical_docstring,
    drop_original=_drop_original_docstring,
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
    https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/

    https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca

    http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        max_values: Optional[Dict[str, Union[int, float]]] = None,
        drop_original: Optional[bool] = False,
    ) -> None:

        _check_numerical_dict(max_values)
        _check_param_drop_original(drop_original)

        self.variables = _check_init_parameter_variables(variables)
        self.max_values = max_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learns the maximum value of each variable.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """
        if self.max_values is None:
            X = super().fit(X)
            self.max_values_ = X[self.variables_].max().to_dict()
        else:
            super()._fit_from_dict(X, self.max_values)
            self.max_values_ = self.max_values

        return self

    def transform(self, X: pd.DataFrame):
        """
        Creates new features using the cyclical transformations.

        Parameters
        ----------
        X: Pandas DataFrame of shame = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: Pandas dataframe.
            The original dataframe plus the additional features.
        """
        X = super().transform(X)

        for variable in self.variables_:
            max_value = self.max_values_[variable]
            X[f"{variable}_sin"] = np.sin(X[variable] * (2.0 * np.pi / max_value))
            X[f"{variable}_cos"] = np.cos(X[variable] * (2.0 * np.pi / max_value))

        if self.drop_original:
            X.drop(columns=self.variables_, inplace=True)

        return X

    def _get_new_features_name(self) -> List:
        """Return names of the created features."""
        feature_names = [
            f"{var}_{suffix}" for var in self.variables_ for suffix in ["sin", "cos"]
        ]
        return feature_names
