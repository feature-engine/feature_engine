from typing import List, Union

import narwhals as nw
import numpy as np
from narwhals.typing import IntoDataFrame

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
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

_NUMPY_OPS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "truediv": np.true_divide,
    "floordiv": np.floor_divide,
    "mod": np.mod,
    "pow": np.power,
}

# these can divide by zero; fill_value handling applies only to them.
_DIVISION_LIKE = {"div", "truediv", "floordiv", "mod"}


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

    This transformer works only with numerical variables. It uses NumPy's `add`,
    `subtract`, `multiply`, `divide`, `true_divide`, `floor_divide`, `mod` and
    `power` under the hood, matching the semantics of the equivalent pandas
    `DataFrame.add`, `DataFrame.sub`, etc. methods.

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

    With polars:

    >>> import polars as pl
    >>> from feature_engine.creation import RelativeFeatures
    >>> X = pl.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "x3": [3, 4, 5]})
    >>> rf = RelativeFeatures(variables=["x1", "x2"],
    >>>                     reference=["x3"],
    >>>                     func=["div"])
    >>> rf.fit(X)
    >>> rf.transform(X)
    shape: (3, 5)
    ┌─────┬─────┬─────┬───────────┬───────────┐
    │ x1  ┆ x2  ┆ x3  ┆ x1_div_x3 ┆ x2_div_x3 │
    │ --- ┆ --- ┆ --- ┆ ---       ┆ ---       │
    │ i64 ┆ i64 ┆ i64 ┆ f64       ┆ f64       │
    ╞═════╪═════╪═════╪═══════════╪═══════════╡
    │ 1   ┆ 4   ┆ 3   ┆ 0.333333  ┆ 1.333333  │
    │ 2   ┆ 5   ┆ 4   ┆ 0.5       ┆ 1.25      │
    │ 3   ┆ 6   ┆ 5   ┆ 0.6       ┆ 1.2       │
    └─────┴─────┴─────┴───────────┴───────────┘
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

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Add new features.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe
            The input dataframe plus the new variables.
        """

        X = self._check_transform_input_and_state(X)

        nw_X = nw.from_native(X, eager_only=True)
        # Extract each column as its own 1D array (not one batched 2D array
        # via select().to_numpy()) so mixed int/float variables each keep
        # their own dtype promotion, matching pandas' per-column .sub()/
        # .div()/etc. instead of upcasting everything to a common dtype.
        var_arrays = {var: nw_X.get_column(var).to_numpy() for var in self.variables}
        ref_arrays = {ref: nw_X.get_column(ref).to_numpy() for ref in self.reference}

        new_series = []
        for func in self.func:
            op = _NUMPY_OPS[func]
            for reference in self.reference:
                ref_arr = ref_arrays[reference]

                if func in _DIVISION_LIKE:
                    zero_mask = ref_arr == 0
                    contains_zero = zero_mask.any()
                    if self.fill_value is None and contains_zero:
                        self._raise_error_when_zero_in_denominator()

                for var in self.variables:
                    name = f"{var}_{func}_{reference}"
                    if func in _DIVISION_LIKE:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            result = op(var_arrays[var], ref_arr)
                        if contains_zero:
                            # floordiv/mod on integer input stay integer-typed;
                            # widen to match fill_value if it wouldn't fit,
                            # mirroring pandas' automatic dtype promotion.
                            fill_arr = np.asarray(self.fill_value)
                            if not np.can_cast(fill_arr, result.dtype, casting="safe"):
                                result = result.astype(
                                    np.result_type(result.dtype, fill_arr.dtype)
                                )
                            result[zero_mask] = self.fill_value
                    else:
                        result = op(var_arrays[var], ref_arr)

                    new_series.append(
                        nw.new_series(name, result, backend=nw_X.implementation)
                    )

        nw_X = nw_X.with_columns(*new_series)
        if self.drop_original is True:
            nw_X = nw_X.drop(list(set(self.variables + self.reference)))

        return nw_X.to_native()

    def _raise_error_when_zero_in_denominator(self):
        raise ValueError(
            "Some of the reference variables contain zeroes. Division by zero "
            "does not exist. Replace zeros before using this transformer for division "
            "or set `fill_value` to a number."
        )

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
