from typing import List, Optional, Union

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from feature_engine.creation.base_creation import BaseCreation
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.class_inputs import (
    _variables_numerical_docstring,
    _drop_original_docstring,
    _missing_values_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.variable_manipulation import _find_or_check_numerical_variables

_PERMITTED_FUNCTIONS = [
    "add",
    "sub",
    "mul",
    "div",
    "truediv",
    "floordiv",
    "mod",
    "pow",
]


@Substitution(
    variables=_variables_numerical_docstring,
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    transform=BaseCreation._transform_docstring,
    fit_transform=_fit_transform_docstring,
)
class RelativeFeatures(BaseCreation):
    """
    RelativeFeatures() applies basic mathematical operations between a group
    of variables and one or more reference features. It adds the resulting features
    to the dataframe.

    In other words, RelativeFeatures() adds, subtracts, multiplies, performs the
    division, true division, floor division, module or exponentiation of a group of
    features to / by a group of reference variables. The features resulting from these
    functions are added to the dataframe.

    This transformer works only with numerical variables. It uses the pandas methods
    `pd.DataFrme.add`, `pd.DataFrme.sub`, `pd.DataFrme.mul`, `pd.DataFrme.div`,
    `pd.DataFrme.truediv`, `pd.DataFrme.floordiv`, `pd.DataFrme.mod` and
    `pd.DataFrme.pow`.
    Find out more in `pandas documentation
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.add.html>`_.

    More details in the :ref:`User Guide <relative_features>`.

    Parameters
    ----------
    variables: list
        The list of numerical variables to combine with the reference variables.

    reference: list
        The list of reference variables that will be added, subtracted, multiplied,
        used as denominator for division and module, or exponent for the exponentiation.

    func: list
        The list of functions to be used in the transformation. The list can contain
        one or more of the following strings: 'add', 'mul','sub', 'div', truediv,
        'floordiv', 'mod', 'pow'.

    {missing_values}

    {drop_original}

    Attributes
    ----------
    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    {transform}

    Notes
    -----
    Although the transformer allows us to combine any feature with any function, we
    recommend its use to create domain knowledge variables. Typical examples within the
    financial sector are:

    - Ratio between income and debt to create the debt_to_income_ratio.
    - Subtraction of rent from income to obtain the disposable_income.
    """

    def __init__(
        self,
        variables: List[Union[str, int]],
        reference: List[Union[str, int]],
        func: List[str],
        missing_values: str = "ignore",
        drop_original: bool = False,
    ) -> None:

        if (
            not isinstance(variables, list)
            or not all(isinstance(var, (int, str)) for var in variables)
            or len(set(variables)) != len(variables)
        ):
            raise ValueError(
                "variables must be a list of strings or integers. "
                f"Got {variables} instead."
            )

        if (
            not isinstance(reference, list)
            or not all(isinstance(var, (int, str)) for var in reference)
            or len(set(reference)) != len(reference)
        ):
            raise ValueError(
                "reference must be a list of strings or integers. "
                f"Got {reference} instead."
            )

        if (
            not isinstance(func, list)
            or any(fun not in _PERMITTED_FUNCTIONS for fun in func)
            or len(set(func)) != len(func)
        ):
            raise ValueError(
                "At least one of the entered functions is not supported or you entered "
                "duplicated functions. "
                "Supported functions are {}. ".format(", ".join(_PERMITTED_FUNCTIONS))
            )

        super().__init__(missing_values, drop_original)
        self.variables = variables
        self.reference = reference
        self.func = func

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        This transformer does not learn any parameter.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to transform.

        y: pandas Series, or np.array. Default=None.
            It is not needed in this transformer. You can pass y or None.
        """
        # Common checks and attributes
        X = super().fit(X, y)

        # check variables are numerical
        self.reference = _find_or_check_numerical_variables(X, self.reference)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add new features.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: Pandas dataframe
            The input dataframe plus the new variables.
        """

        X = super().transform(X)

        methods_dict = {
            "add": self._add,
            "mul": self._mul,
            "sub": self._sub,
            "div": self._div,
            "truediv": self._truediv,
            "floordiv": self._floordiv,
            "mod": self._mod,
            "pow": self._pow,
        }

        for func in self.func:
            methods_dict[func](X)

        if self.drop_original:
            X.drop(
                columns=set(self.variables + self.reference),
                inplace=True,
            )

        return X

    def _sub(self, X):
        for reference in self.reference:
            varname = [f"{var}_sub_{reference}" for var in self.variables]
            X[varname] = X[self.variables].sub(X[reference], axis=0)
        return X

    def _div(self, X):
        for reference in self.reference:
            if (X[reference] == 0).any():
                raise ValueError(
                    "Some of the reference variables contain 0 as values. Check and "
                    "remove those before using this transformer."
                )
            varname = [f"{var}_div_{reference}" for var in self.variables]
            X[varname] = X[self.variables].div(X[reference], axis=0)
        return X

    def _add(self, X):
        for reference in self.reference:
            varname = [f"{var}_add_{reference}" for var in self.variables]
            X[varname] = X[self.variables].add(X[reference], axis=0)
        return X

    def _mul(self, X):
        for reference in self.reference:
            varname = [f"{var}_mul_{reference}" for var in self.variables]
            X[varname] = X[self.variables].mul(X[reference], axis=0)
        return X

    def _truediv(self, X):

        for reference in self.reference:
            if (X[reference] == 0).any():
                raise ValueError(
                    "Some of the reference variables contain 0 as values. Check and "
                    "remove those before using this transformer."
                )
            varname = [f"{var}_truediv_{reference}" for var in self.variables]
            X[varname] = X[self.variables].truediv(X[reference], axis=0)
        return X

    def _floordiv(self, X):
        for reference in self.reference:
            if (X[reference] == 0).any():
                raise ValueError(
                    "Some of the reference variables contain 0 as values. Check and "
                    "remove those before using this transformer."
                )
            varname = [f"{var}_floordiv_{reference}" for var in self.variables]
            X[varname] = X[self.variables].floordiv(X[reference], axis=0)
        return X

    def _mod(self, X):
        for reference in self.reference:
            if (X[reference] == 0).any():
                raise ValueError(
                    "Some of the reference variables contain 0 as values. Check and "
                    "remove those before using this transformer."
                )
            varname = [f"{var}_mod_{reference}" for var in self.variables]
            X[varname] = X[self.variables].mod(X[reference], axis=0)
        return X

    def _pow(self, X):
        for reference in self.reference:
            varname = [f"{var}_pow_{reference}" for var in self.variables]
            X[varname] = X[self.variables].pow(X[reference], axis=0)
        return X

    def get_feature_names_out(self, input_features: Optional[bool] = None) -> List:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features: bool, default=None
            If `input_features` is `None`, then the names of all the variables in the
            transformed dataset (original + new variables) is returned. Alternatively,
            if `input_features` is True, only the names for the new features will be
            returned.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        check_is_fitted(self)

        if input_features is not None and not isinstance(input_features, bool):
            raise ValueError(
                "input_features takes None or a boolean, True or False. "
                f"Got {input_features} instead."
            )

        # Names of new features
        feature_names = [
            f"{var}_{fun}_{reference}"
            for fun in self.func
            for reference in self.reference
            for var in self.variables
        ]

        if input_features is None or input_features is False:
            if self.drop_original is True:
                # Remove names of variables to drop.
                original = [
                    f
                    for f in self.feature_names_in_
                    if f not in self.variables + self.reference
                ]
                feature_names = original + feature_names
            else:
                feature_names = self.feature_names_in_ + feature_names

        return feature_names
