from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils import deprecated

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.fit_attributes import (
    _variables_attribute_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.class_inputs import (
    _variables_numerical_docstring,
    _drop_original_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.variable_manipulation import _check_input_parameter_variables


@deprecated(
    "CyclicalTransformer() is deprecated in version 1.3 and will be removed in "
    "version 1.4. Use CyclicalFeatures() instead."
)
@Substitution(
    variables=_variables_numerical_docstring,
    drop_original=_drop_original_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class CyclicalTransformer(BaseNumericalTransformer):
    """
    **DEPRECATED: CyclicalTransformer() is deprecated in version 1.3 and will be removed
    in version 1.4. Use CyclicalFeatures() instead.**

    The CyclicalTransformer() applies cyclical transformations to numerical
    variables, returning 2 new features per variable, according to:

    - var_sin = sin(variable * (2. * pi / max_value))
    - var_cos = cos(variable * (2. * pi / max_value))

    where max_value is the maximum value in the variable, and pi is 3.14...

    The CyclicalTransformer() works only with numerical variables. A list of variables
    to transform can be passed as an argument. Alternatively, the transformer will
    automatically select and transform all numerical variables.

    Missing data should be imputed before applying this transformer.

    More details in the :ref:`User Guide <cyclical_transformer>`.

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
        The maximum value of the cyclical feature.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learns the maximum values of the cyclical features.

    transform:
        Create new features.

    {fit_transform}

    References
    ----------
    http://blog.davidkaleko.com/feature-engineering-cyclical-features.html
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        max_values: Optional[Dict[str, Union[int, float]]] = None,
        drop_original: Optional[bool] = False,
    ) -> None:

        if max_values:
            if not isinstance(max_values, dict) or not all(
                isinstance(var, (int, float)) for var in list(max_values.values())
            ):
                raise TypeError(
                    "max_values takes a dictionary of strings as keys, "
                    "and numbers as items to be used as the reference for"
                    "the max value of each column."
                )

        if not isinstance(drop_original, bool):
            raise TypeError("drop_original takes only boolean values True and False.")

        self.variables = _check_input_parameter_variables(variables)
        self.max_values = max_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learns the maximum value of each cyclical variable.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = super()._fit_from_varlist(X)

        if self.max_values is None:
            self.max_values_ = X[self.variables_].max().to_dict()
        else:
            for key in list(self.max_values.keys()):
                if key not in self.variables_:
                    raise ValueError(
                        f"The mapping key {key} is not present in variables."
                    )
            self.max_values_ = self.max_values

        return self

    def transform(self, X: pd.DataFrame):
        """
        Creates new features using the cyclical transformation.

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
