"""Series of checks to be performed on dataframes used as inputs of methods fit() and
transform().
"""

from typing import List, Tuple, Union

import narwhals as nw
import narwhals.dependencies as nwd
import narwhals.selectors as nws
import numpy as np
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.utils.validation import _check_y, check_consistent_length, column_or_1d


def check_X(X: IntoDataFrame) -> IntoDataFrame:
    """
    Checks that X is a dataframe from any library supported by narwhals (for example
    pandas, polars, modin, cuDF, or PyArrow).

    Parameters
    ----------
    X : dataframe (pandas, polars, PyArrow, modin, or cuDF). Feature-engine does
        not support libraries that build a deferred query plan (for example Dask,
        DuckDB, PySpark, Ibis, or a polars LazyFrame). Convert those to an eager
        dataframe (e.g. `LazyFrame.collect()`) before passing them in.
        The input to check and transform.

    Raises
    ------
    TypeError
        If the input is not a recognised dataframe.
    ValueError
        If the input has duplicated column names, or 0 columns or rows.

    Returns
    -------
    X : narwhals dataframe.
        The validated dataframe in narwhals format.
    """
    if nwd.is_into_dataframe(X):
        # from_native() raises narwhals.exceptions.DuplicateError, a ValueError
        # subclass, when the dataframe has duplicated column names.
        nw_X = nw.from_native(X, eager_only=True)
        if nw_X.is_empty() or nw_X.shape[1] == 0:
            raise ValueError(
                f"Found array with 0 feature(s) (shape={nw_X.shape}) while a "
                "minimum of 1 is required."
            )

    else:
        raise TypeError(
            "X must be a dataframe from a library supported by narwhals "
            f"(e.g. pandas, polars, PyArrow). Got {type(X)} instead."
        )

    return nw_X


def check_y(
    y: Union[IntoSeries, IntoDataFrame, np.generic, np.ndarray, List],
    y_numeric: bool = False,
):
    """
    Checks that y is a Series or DataFrame from a library supported by narwhals (for
    example pandas or polars), or alternatively, if it can be converted to a numpy
    array.

    Parameters
    ----------
    y : Series or DataFrame (pandas, polars, PyArrow, modin, or cuDF), np.array,
        list. Feature-engine does not support libraries that build a deferred
        query plan (for example Dask, DuckDB, PySpark, Ibis, or a polars
        LazyFrame). Convert those to an eager dataframe (e.g. `LazyFrame.collect()`)
        before passing them in.
        The input to check.

    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is not numeric,
        it is cast to float64. Should only be used for regression algorithms.

    Returns
    -------
    y: Series, DataFrame, or numpy array
    """
    if y is None:
        raise ValueError(
            "requires y to be passed, but the target y is None",
            "Expected array-like (array or non-string sequence), got None",
            "y should be a 1d array",
        )

    if nwd.is_into_series(y):
        nw_y = nw.from_native(y, series_only=True)
        if nw_y.is_null().any() or (
            nw_y.dtype.is_numeric() and nw_y.is_nan().any()
        ):
            raise ValueError("y contains NaN values.")
        if nw_y.dtype.is_numeric():
            if not np.isfinite(nw_y.to_numpy()).all():
                raise ValueError("y contains infinity values.")
        elif y_numeric:
            nw_y = nw_y.cast(nw.Float64())
        return nw_y.to_native()

    if nwd.is_into_dataframe(y):
        nw_y = nw.from_native(y, eager_only=True)
        if (
            nw_y.select(nw.all().is_null().any()).to_numpy().any()
            or nw_y.select(nws.numeric().is_nan().any()).to_numpy().any()
        ):
            raise ValueError("y contains NaN values.")
        if not np.isfinite(nw_y.to_numpy()).all():
            raise ValueError("y contains infinity values.")
        return nw_y.to_native()

    try:
        y = column_or_1d(y)
        return _check_y(y, multi_output=False, y_numeric=y_numeric)
    except ValueError:
        return _check_y(y, multi_output=True, y_numeric=y_numeric)


def check_X_y(
    X: IntoDataFrame,
    y: Union[IntoSeries, IntoDataFrame, np.generic, np.ndarray, List],
    y_numeric: bool = False,
) -> Tuple[IntoDataFrame, Union[IntoSeries, IntoDataFrame, np.ndarray]]:
    """
    Ensures X and y are compatible dataframe/array-like objects with a consistent
    number of rows. If both are pandas objects, checks that their indexes match.

    Parameters
    ----------
    X: dataframe (pandas, polars, PyArrow, modin, or cuDF). Feature-engine does
        not support libraries that build a deferred query plan (for example Dask,
        DuckDB, PySpark, Ibis, or a polars LazyFrame). Convert those to an eager
        dataframe (e.g. `LazyFrame.collect()`) before passing them in.
        The input to check.

    y: Series, DataFrame (pandas, polars, or any other library supported by
        narwhals), np.array, list
        The input to check.

    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is not numeric,
        it is cast to float64. Should only be used for regression algorithms.

    Raises
    ------
    TypeError
        If X is not a recognised dataframe.
    ValueError
        If X has duplicated column names, 0 columns, or 0 rows; if y is None, or
        contains NaN or infinity values; if X and y have a different number of
        rows; or if X and y are pandas objects with mismatched indexes.

    Returns
    -------
    X: narwhals dataframe
    y: Series, DataFrame, or numpy array
    """
    X = check_X(X)
    y = check_y(y, y_numeric=y_numeric)
    check_consistent_length(X, y)

    if X.implementation.is_pandas():
        if nwd.is_pandas_series(y) or nwd.is_pandas_dataframe(y):
            if not X.to_native().index.equals(y.index):
                raise ValueError("The indexes of X and y do not match.")

    return X, y


def _check_X_matches_training_df(X: IntoDataFrame, reference: int) -> None:
    """
    Checks that the dataframe to transform has the same number of columns as the
    dataframe used with the fit() method.

    Parameters
    ----------
    X : dataframe (pandas, polars, or any other library supported by narwhals)
        The df to be checked.
    reference : int
        The number of columns in the dataframe that was used with the fit() method.

    Raises
    ------
    ValueError
        If the number of columns does not match.
    """
    if X.shape[1] != reference:
        raise ValueError(
            "The number of columns in this dataset is different from the one used to "
            "fit this transformer (when using the fit() method)."
        )


def _check_contains_na(
    X: IntoDataFrame,
    variables: List[Union[str, int]],
    error_msg: str = "simple",
) -> None:
    """
    Checks if the dataframe contains null values in the selected columns.

    Parameters
    ----------
    X : dataframe

    variables : List
        The selected group of variables in which null values will be examined.

    error_msg : str, default="simple"
        The message in the error. Some transformers can ignore null values.

    Raises
    ------
    ValueError
        If the variable(s) contain null values.
    """
    error_msg_simple = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    error_msg_ignore = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    if len(variables) == 0:
        return
    nw_X = nw.from_native(X, eager_only=True)
    if nwd.is_pandas_dataframe(X):
        numeric_vars = list(X[variables].select_dtypes(include="number").columns)
    else:
        numeric_vars = nw_X.select(variables).select(nw.selectors.numeric()).columns
    if nw_X.select(nw.col(variables).is_null().any()).to_numpy().any() or (
        numeric_vars
        and nw_X.select(nw.col(numeric_vars).is_nan().any()).to_numpy().any()
    ):
        if error_msg == "simple":
            raise ValueError(error_msg_simple)
        else:
            raise ValueError(error_msg_ignore)


def _check_contains_inf(X: IntoDataFrame, variables: List[Union[str, int]]) -> None:
    """
    Checks if the dataframe contains inf values in the selected columns.

    Parameters
    ----------
    X : dataframe
    variables : List
        The selected group of variables in which infinite values will be examined.

    Raises
    ------
    ValueError
        If the variable(s) contain np.inf values
    """
    values = nw.from_native(X, eager_only=True).select(nw.col(variables)).to_numpy()
    if np.isinf(values.astype(float)).any():
        raise ValueError(
            "Some of the variables to transform contain inf values. Check and "
            "remove those before using this transformer."
        )
