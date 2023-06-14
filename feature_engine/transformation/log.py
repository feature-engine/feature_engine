# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._base_transformers.mixins import FitFromDictMixin
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
from feature_engine.tags import _return_tags
from feature_engine.variable_handling._init_parameter_checks import (
    _check_init_parameter_variables,
)


@Substitution(
    variables=_variables_numerical_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class LogTransformer(BaseNumericalTransformer):
    """
    The LogTransformer() applies the natural logarithm or the base 10 logarithm to
    numerical variables. The natural logarithm is the logarithm in base e.

    The LogTransformer() only works with positive values. If the variable
    contains a zero or a negative value the transformer will return an error.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type numeric.

    More details in the :ref:`User Guide <log_transformer>`.

    Parameters
    ----------
    {variables}

    base: string, default='e'
        Indicates if the natural or base 10 logarithm should be applied. Can take
        values 'e' or '10'.

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
        Transform the variables using the logarithm.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.transformation import LogTransformer
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.lognormal(size = 100)))
    >>> lt = LogTransformer()
    >>> lt.fit(X)
    >>> X = lt.transform(X)
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
        base: str = "e",
    ) -> None:

        if base not in ["e", "10"]:
            raise ValueError("base can take only '10' or 'e' as values")

        self.variables = _check_init_parameter_variables(variables)
        self.base = base

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn parameters.

        Selects the numerical variables and determines whether the logarithm
        can be applied on the selected variables, i.e., it checks that the variables
        are positive.

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

        # check contains zero or negative values
        if (X[self.variables_] <= 0).any().any():
            raise ValueError(
                "Some variables contain zero or negative values, can't apply log"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the variables with the logarithm.

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

        # check contains zero or negative values
        if (X[self.variables_] <= 0).any().any():
            raise ValueError(
                "Some variables contain zero or negative values, can't apply log"
            )

        # transform
        if self.base == "e":
            X.loc[:, self.variables_] = np.log(X.loc[:, self.variables_])
        elif self.base == "10":
            X.loc[:, self.variables_] = np.log10(X.loc[:, self.variables_])

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
        if self.base == "e":
            X.loc[:, self.variables_] = np.exp(X.loc[:, self.variables_])
        elif self.base == "10":
            X.loc[:, self.variables_] = np.array(10 ** X.loc[:, self.variables_])

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        # =======  this tests fail because the transformers throw an error
        # when the values are 0. Nothing to do with the test itself but
        # mostly with the data created and used in the test
        msg = (
            "transformers raise errors when data contains zeroes, thus this check fails"
        )
        tags_dict["_xfail_checks"]["check_estimators_dtypes"] = msg
        tags_dict["_xfail_checks"]["check_estimators_fit_returns_self"] = msg
        tags_dict["_xfail_checks"]["check_pipeline_consistency"] = msg
        tags_dict["_xfail_checks"]["check_estimators_overwrite_params"] = msg
        tags_dict["_xfail_checks"]["check_estimators_pickle"] = msg
        tags_dict["_xfail_checks"]["check_transformer_general"] = msg

        return tags_dict


@Substitution(
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class LogCpTransformer(BaseNumericalTransformer, FitFromDictMixin):
    """
    The LogCpTransformer() applies the transformation log(x + C), where C is a positive
    constant, to the input variable. It applies the natural logarithm or the base 10
    logarithm, where the natural logarithm is logarithm in base e.

    The logarithm can only be applied to numerical non-negative values. If the
    variable contains a zero or a negative value after adding a constant C, the
    transformer will return an error.

    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type numeric.

    More details in the :ref:`User Guide <log_cp>`.

    Parameters
    ----------
    variables: list, default=None
        The list of numerical variables to transform. If None, the transformer
        will find and select all numerical variables. If C is a dictionary, then this
        parameter is ignored and the variables to transform are selected from the
        dictionary keys.

    base: string, default='e'
        Indicates if the natural or base 10 logarithm should be applied. Can take
        values 'e' or '10'.

    C: "auto", int or dict, default="auto"
        The constant C to add to the variable before the logarithm, i.e., log(x + C).

        - If int, then log(x + C)
        - If "auto", then C = abs(min(x)) + 1
        - If dict, dictionary mapping the constant C to apply to each variable.

        Note, when C is a dictionary, the parameter `variables` is ignored.

    Attributes
    ----------
    {variables_}

    C_:
        The constant C to add to each variable. If C = "auto" a dictionary with
        C = abs(min(variable)) + 1.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the constant C.

    {fit_transform}

    {inverse_transform}

    transform:
        Transform the variables with the logarithm of x plus C.

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.transformation import LogCpTransformer
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x = np.random.lognormal(size = 100)))
    >>> lct = LogCpTransformer()
    >>> lct.fit(X)
    >>> X = lct.transform(X)
    >>> X.head()
              x
    0  0.944097
    1  0.586701
    2  1.043204
    3  1.707159
    4  0.541405
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        base: str = "e",
        C: Union[int, float, str, Dict[Union[str, int], Union[float, int]]] = "auto",
    ) -> None:

        if base not in ["e", "10"]:
            raise ValueError("base can take only '10' or 'e' as values")

        if not isinstance(C, (int, float, dict)) and not C == "auto":
            raise ValueError("C can take only 'auto', integers or floats")

        self.variables = _check_init_parameter_variables(variables)
        self.base = base
        self.C = C

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the constant C to add to the variable before the logarithm transformation
        if C="auto".

        Select the numerical variables or check that the variables entered by the user
        are numerical. Then check that the selected variables are positive after
        addition of C.

        Parameters
        ----------
        X: Pandas DataFrame of shape = [n_samples, n_features].
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, default=None
            It is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        if isinstance(self.C, dict):
            X = super()._fit_from_dict(X, self.C)
        else:
            X = super().fit(X)

        self.C_ = self.C

        # calculate C to add to each variable
        if self.C == "auto":
            self.C_ = dict(X[self.variables_].min(axis=0).abs() + 1)

        # check variables are positive after adding C
        if (X[self.variables_] + self.C_ <= 0).any().any():
            raise ValueError(
                "Some variables contain zero or negative values after adding"
                + "constant C, can't apply log"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the variables with the logarithm of x plus a constant C.

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

        # check variable is positive after adding c
        if (X[self.variables_] + self.C_ <= 0).any().any():
            raise ValueError(
                "Some variables contain zero or negative values after adding"
                + "constant C, can't apply log"
            )

        # transform
        if self.base == "e":
            X.loc[:, self.variables_] = np.log(X.loc[:, self.variables_] + self.C_)
        elif self.base == "10":
            X.loc[:, self.variables_] = np.log10(X.loc[:, self.variables_] + self.C_)

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
        X_tr: Pandas dataframe
            The dataframe with the transformed variables.
        """

        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        # inverse transform
        if self.base == "e":
            X.loc[:, self.variables_] = np.exp(X.loc[:, self.variables_]) - self.C_
        elif self.base == "10":
            X.loc[:, self.variables_] = 10 ** X.loc[:, self.variables_] - self.C_

        return X
