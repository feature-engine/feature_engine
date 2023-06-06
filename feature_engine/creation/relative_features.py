from typing import List, Union

import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _drop_original_docstring,
    _missing_values_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _transform_creation_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.creation.base_creation import BaseCreation

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
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    transform=_transform_creation_docstring,
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

    fill_value: int, float, default=None
        When dividing by zero, this value is used in place of infinity. If None,
        then an error will be raised when dividing by zero.

    {missing_values}

    {drop_original}

    Attributes
    ----------
    {variables_}

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

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.creation import RelativeFeatures
    >>> X = pd.DataFrame(dict(x1 = [1,2,3], x2 = [4,5,6], x3 = [3,4,5]))
    >>> rf = RelativeFeatures(variables = ["x1","x2"],
    >>>                     reference = ["x3"],
    >>>                     func = ["div"])
    >>> rf.fit(X)
    >>> rf.transform(X)
       x1  x2  x3  x1_div_x3  x2_div_x3
    0   1   4   3   0.333333   1.333333
    1   2   5   4   0.500000   1.250000
    2   3   6   5   0.600000   1.200000
    """

    def __init__(
        self,
        variables: List[Union[str, int]],
        reference: List[Union[str, int]],
        func: List[str],
        fill_value: Union[int, float, None] = None,
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

        if fill_value is not None and not isinstance(fill_value, (float, int)):
            raise ValueError(
                "fill_value must be a float, integer or None. "
                f"Got {fill_value} instead."
            )
        super().__init__(missing_values, drop_original)
        self.variables = variables
        self.reference = reference
        self.func = func
        self.fill_value = fill_value

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

        X = self._check_transform_input_and_state(X)

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

    def _div(self, X):
        for reference in self.reference:
            zeros_ix, contains_zero = self._find_zeroes_in_reference(X, reference)

            if self.fill_value is None and contains_zero:
                self._raise_error_when_zero_in_denominator()

            varname = [f"{var}_div_{reference}" for var in self.variables]
            X[varname] = X[self.variables].div(X[reference], axis=0)

            if contains_zero:
                X.loc[zeros_ix, varname] = self.fill_value
        return X

    def _truediv(self, X):
        for reference in self.reference:
            zeros_ix, contains_zero = self._find_zeroes_in_reference(X, reference)

            if self.fill_value is None and contains_zero:
                self._raise_error_when_zero_in_denominator()

            varname = [f"{var}_truediv_{reference}" for var in self.variables]
            X[varname] = X[self.variables].truediv(X[reference], axis=0)

            if contains_zero:
                X.loc[zeros_ix, varname] = self.fill_value
        return X

    def _floordiv(self, X):
        for reference in self.reference:
            zeros_ix, contains_zero = self._find_zeroes_in_reference(X, reference)

            if self.fill_value is None and contains_zero:
                self._raise_error_when_zero_in_denominator()

            varname = [f"{var}_floordiv_{reference}" for var in self.variables]
            X[varname] = X[self.variables].floordiv(X[reference], axis=0)

            if contains_zero:
                X.loc[zeros_ix, varname] = self.fill_value
        return X

    def _mod(self, X):
        for reference in self.reference:
            zeros_ix, contains_zero = self._find_zeroes_in_reference(X, reference)

            if self.fill_value is None and contains_zero:
                self._raise_error_when_zero_in_denominator()

            varname = [f"{var}_mod_{reference}" for var in self.variables]
            X[varname] = X[self.variables].mod(X[reference], axis=0)

            if contains_zero:
                X.loc[zeros_ix, varname] = self.fill_value
        return X

    def _pow(self, X):
        for reference in self.reference:
            varname = [f"{var}_pow_{reference}" for var in self.variables]
            X[varname] = X[self.variables].pow(X[reference], axis=0)
        return X

    def _raise_error_when_zero_in_denominator(self):
        raise ValueError(
            "Some of the reference variables contain zeroes. Division by zero "
            "does not exist. Replace zeros before using this transformer for division "
            "or set `fill_value` to a number."
        )

    def _find_zeroes_in_reference(self, X, var):
        zero_ix = X[var] == 0
        zero_bool = (zero_ix).any()
        return zero_ix, zero_bool

    def _get_new_features_name(self) -> List:
        """Return names of the created features."""

        # Names of new features
        feature_names = [
            f"{var}_{fun}_{reference}"
            for fun in self.func
            for reference in self.reference
            for var in self.variables
        ]
        return feature_names
