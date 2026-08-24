import warnings
from typing import Any, List, Optional, Union

import narwhals as nw
import narwhals.dependencies as nwd
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
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
    _transform_creation_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.creation.base_creation import BaseCreation


def _pandas_version() -> int:
    return int(nwd.get_pandas().__version__.split(".")[0])


# In pandas < 3, agg() maps these callables to the pandas methods and warns that
# this will change; the string alias keeps that behaviour (e.g., np.std ->
# Series.std with ddof=1) without the warning. In pandas >= 3 the callables are
# used directly (np.std applies ddof=0), so they must not be aliased.
_FUNC_TO_STRING_ALIAS = {
    sum: "sum",
    min: "min",
    max: "max",
    np.sum: "sum",
    np.mean: "mean",
    np.std: "std",
    np.var: "var",
    np.median: "median",
    np.min: "min",
    np.max: "max",
    np.prod: "prod",
}

# The kwargs preserve pandas' string-reduction defaults. In particular, pandas
# uses one degree of freedom for ``std`` and ``var`` while NumPy uses zero.
# NumPy callables have their direct pandas >= 3 semantics instead (ddof=0).
_NUMPY_REDUCERS = {
    "sum": (np.nansum, {}),
    "mean": (np.nanmean, {}),
    "std": (np.nanstd, {"ddof": 1}),
    "var": (np.nanvar, {"ddof": 1}),
    "min": (np.nanmin, {}),
    "max": (np.nanmax, {}),
    "prod": (np.nanprod, {}),
    "median": (np.nanmedian, {}),
    np.sum: (np.nansum, {}),
    np.mean: (np.nanmean, {}),
    np.std: (np.nanstd, {"ddof": 0}),
    np.var: (np.nanvar, {"ddof": 0}),
    np.min: (np.nanmin, {}),
    np.max: (np.nanmax, {}),
    np.prod: (np.nanprod, {}),
    np.median: (np.median, {}),
}


def _get_numpy_reducer(func):
    """Return the NumPy reducer for a supported aggregation."""
    return _NUMPY_REDUCERS.get(func)


@Substitution(
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    transform=_transform_creation_docstring,
    fit_transform=_fit_transform_docstring,
)
class MathFeatures(BaseCreation):
    """
    MathFeatures() applies functions across multiple features returning one or more
    additional features as a result. Common reductions use vectorized NumPy
    operations. Other functions fall back to `pandas.agg()` with `axis=1` for
    pandas input, or to polars' native `map_rows()` for polars input — in that
    case, the callable receives each row as a plain tuple, not a `Series`, so
    it must not rely on `Series` methods (e.g. use `max(row)` instead of
    `row.max()`) to work on both backends.

    For supported aggregation functions, see `pandas documentation
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html>`_.

    Note that if some of the variables have missing data and `missing_values='ignore'`,
    the value will be ignored in the computation. To be clear, if variables A, B and C,
    have values 10, 20 and NA, and we perform the sum, the result will be A + B = 30.

    More details in the :ref:`User Guide <math_features>`.

    Parameters
    ----------
    variables: list
        The list of input variables. Variables must be numerical and there must be at
        least 2 different variables in the list.

    func: function, string, list
        Functions to use for aggregating the data. Same functionality as parameter
        `func` in `pandas.agg()`. If a function, it must either work when passed a
        DataFrame or when passed to DataFrame.apply. Accepted combinations are:

        - function
        - string function name
        - list of functions and/or function names, e.g. [np.sum, 'mean']

        Each function will result in a new variable that will be added to the
        transformed dataset.

    new_variables_names: list, default=None
        Names of the new variables. If passing a list with names (recommended), enter
        one name per function. If None, the transformer will assign arbitrary names,
        starting with the function and followed by the variables separated by _.

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
    Although the transformer allows us to combine any features with any functions, we
    recommend using it to create features based on domain knowledge. Typical examples
    in finance are:

    - Sum debt across financial products, i.e., credit cards, to obtain the total debt.
    - Take the average payments to various financial products.
    - Find the minimum payment done at any one month.

    In insurance, we can sum the damage to various parts of a car to obtain the
    total damage.

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.creation import MathFeatures
    >>> X = pd.DataFrame(dict(x1 = [1,2,3], x2 = [4,5,6]))
    >>> mf = MathFeatures(variables = ["x1","x2"], func = "sum")
    >>> mf.fit(X)
    >>> mf.transform(X)
       x1  x2  sum_x1_x2
    0   1   4          5
    1   2   5          7
    2   3   6          9

    >>> mf = MathFeatures(variables = ["x1","x2"], func = "prod")
    >>> mf.fit(X)
    >>> mf.transform(X)
       x1  x2  prod_x1_x2
    0   1   4           4
    1   2   5          10
    2   3   6          18

    >>> mf = MathFeatures(variables = ["x1","x2"], func = "mean")
    >>> mf.fit(X)
    >>> mf.transform(X)
       x1  x2  mean_x1_x2
    0   1   4         2.5
    1   2   5         3.5
    2   3   6         4.5

    With polars:

    >>> import polars as pl
    >>> from feature_engine.creation import MathFeatures
    >>> X = pl.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
    >>> mf = MathFeatures(variables=["x1", "x2"], func="sum")
    >>> mf.fit(X)
    >>> mf.transform(X)
    shape: (3, 3)
    ┌─────┬─────┬───────────┐
    │ x1  ┆ x2  ┆ sum_x1_x2 │
    │ --- ┆ --- ┆ ---       │
    │ i64 ┆ i64 ┆ i64       │
    ╞═════╪═════╪═══════════╡
    │ 1   ┆ 4   ┆ 5         │
    │ 2   ┆ 5   ┆ 7         │
    │ 3   ┆ 6   ┆ 9         │
    └─────┴─────┴───────────┘
    """

    def __init__(
        self,
        variables: List[Union[str, int]],
        func: Any,
        new_variables_names: Optional[List[str]] = None,
        missing_values: str = "raise",
        drop_original: bool = False,
    ) -> None:

        if (
            not isinstance(variables, list)
            or not all(isinstance(var, (int, str)) for var in variables)
            or len(variables) < 2
            or len(set(variables)) != len(variables)
        ):
            raise ValueError(
                "variables must be a list of strings or integers with at least 2 "
                f"different variables. Got {variables} instead."
            )

        if isinstance(func, dict):
            raise NotImplementedError(
                "func does not work with dictionaries in this transformer."
            )

        if new_variables_names is not None:
            if (
                not isinstance(new_variables_names, list)
                or not all(isinstance(var, str) for var in new_variables_names)
                or len(set(new_variables_names)) != len(new_variables_names)
            ):
                raise ValueError(
                    "new_variable_names should be None or a list of unique strings. "
                    f"Got {new_variables_names} instead."
                )

        if new_variables_names is not None:
            if isinstance(func, list):
                if len(new_variables_names) != len(func):
                    raise ValueError(
                        "The number of new feature names must coincide with the number "
                        "of functions."
                    )
            else:
                if len(new_variables_names) != 1:
                    raise ValueError(
                        "The number of new feature names must coincide with the number "
                        "of functions."
                    )

        super().__init__(missing_values, drop_original)

        self.variables = variables
        self.func = func
        self.new_variables_names = new_variables_names

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Create and add new variables.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe, shape = [n_samples, n_features + n_operations]
            The input dataframe plus the new variables.
        """
        X = self._check_transform_input_and_state(X)

        new_variable_names = self._get_new_features_name()

        func = self.func
        is_pandas = nwd.is_pandas_dataframe(X)
        if is_pandas is True and _pandas_version() < 3:
            if isinstance(func, list):
                func = [_FUNC_TO_STRING_ALIAS.get(fun, fun) for fun in func]
            else:
                func = _FUNC_TO_STRING_ALIAS.get(func, func)

        functions = func if isinstance(func, list) else [func]
        reducers = [_get_numpy_reducer(fun) for fun in functions]

        nw_X = nw.from_native(X, eager_only=True)
        if is_pandas is True:
            values = X[self.variables].to_numpy()
        else:
            values = nw_X.select(self.variables).to_numpy()

        # Nullable extension dtypes produce object arrays. Keep those, custom
        # callables, and less common aggregations on the fallback path below.
        if reducers and values.dtype.kind in "biuf" and all(reducers):
            new_series = []
            for (reducer, kwargs), name in zip(reducers, new_variable_names):
                # pandas' named reductions do not warn for empty/all-missing rows.
                # NumPy returns the same values but emits RuntimeWarning for some
                # reducers, so silence only those warnings on this equivalent path.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = reducer(values, axis=1, **kwargs)
                new_series.append(
                    nw.new_series(name, result, backend=nw_X.implementation)
                )
            nw_X = nw_X.with_columns(*new_series)
            if self.drop_original is True:
                nw_X = nw_X.drop(self.variables)
            X = nw_X.to_native()
        elif is_pandas is True:
            result = X[self.variables].agg(func, axis=1)
            if len(new_variable_names) == 1:
                X[new_variable_names[0]] = result
            else:
                X[new_variable_names] = result
            if self.drop_original is True:
                X = X.drop(columns=self.variables)
        else:
            # polars has no equivalent to pandas' agg(func, axis=1): apply each
            # function natively via map_rows, one call per function. map_rows
            # passes each row as a plain tuple, not a Series, so callables that
            # rely on Series methods (e.g. `row.max()`) need `max(row)` instead.
            sub_native = nw_X.select(self.variables).to_native()
            new_series = []
            for fun, name in zip(functions, new_variable_names):
                if not callable(fun):
                    raise NotImplementedError(
                        f"'{fun}' has no NumPy-vectorized implementation, and "
                        "non-callable aggregation names are not supported for "
                        "polars input. Pass a Python callable instead."
                    )
                result_df = sub_native.map_rows(fun)
                new_series.append(
                    nw.new_series(
                        name, result_df.to_series(0), backend=nw_X.implementation
                    )
                )
            nw_X = nw_X.with_columns(*new_series)
            if self.drop_original is True:
                nw_X = nw_X.drop(self.variables)
            X = nw_X.to_native()

        return X

    def _get_new_features_name(self) -> List:
        """Return names of the created features."""

        # create name of the new variables
        if self.new_variables_names is not None:
            feature_names = self.new_variables_names

        else:
            varlist = [f"{var}" for var in self.variables_]

            if isinstance(self.func, list):
                functions = [
                    fun if type(fun) is str else fun.__name__ for fun in self.func
                ]
                feature_names = [
                    f"{function}_{'_'.join(varlist)}" for function in functions
                ]
            else:
                feature_names = [f"{self.func}_{'_'.join(varlist)}"]

        return feature_names
