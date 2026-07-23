"""Series of checks to be performed on dataframes used as inputs of methods fit() and
transform().
"""

from typing import List, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from scipy.sparse import issparse
from sklearn.utils.validation import (
    _check_y,
    check_array,
    check_consistent_length,
    column_or_1d,
)


def check_X(X):
    """
    Checks that X is a dataframe from any library supported by narwhals (for example
    pandas, polars, modin, cuDF, or PyArrow), or a numpy array, and returns a copy.

    Numpy arrays are validated but otherwise returned as numpy arrays, unchanged.
    Feature-engine accepts numpy arrays, in addition to dataframes, so that:

    - transformers can be evaluated with Scikit-learn's `check_estimator`, which
      passes numpy arrays to `fit()` and `transform()`.
    - transformers can be used within a Scikit-learn Pipeline, next to Scikit-learn
      transformers like the `SimpleImputer`, which return numpy arrays by default.

    Parameters
    ----------
    X : dataframe (pandas, polars, or any other library supported by narwhals), or
        numpy array.
        The input to check and copy or transform.

    Raises
    ------
    TypeError
        If the input is not a recognised dataframe, or a numpy array.
    ValueError
        If the input has duplicated column names, or 0 columns or rows.

    Returns
    -------
    X : dataframe or numpy array.
        A copy of the original dataframe, or the validated numpy array.
    """
    if nwd.is_into_dataframe(X):
        # from_native() raises narwhals.exceptions.DuplicateError, a ValueError
        # subclass, when the dataframe has duplicated column names.
        nw_X = nw.from_native(X, eager_only=True)
        if nw_X.is_empty():
            raise ValueError(
                f"Found array with 0 feature(s) (shape={nw_X.shape}) while a "
                "minimum of 1 is required."
            )
        return nw_X.to_native()

    if isinstance(X, (np.generic, np.ndarray)) or issparse(X):
        return check_array(
            X, accept_sparse=False, dtype=None, ensure_all_finite=False, ensure_2d=True
        )

    raise TypeError(
        "X must be a numpy array or a dataframe from a library supported by "
        f"narwhals (e.g. pandas, polars). Got {type(X)} instead."
    )


def _copy_series(y: nw.Series):
    """Returns a copy of a narwhals Series, in its native library format."""
    name = y.name
    frame = y.to_frame().clone()
    return frame.get_column(frame.columns[0]).alias(name).to_native()


def check_y(
    y: Union[np.generic, np.ndarray, List],
    y_numeric: bool = False,
):
    """
    Checks that y is a Series or DataFrame from a library supported by narwhals (for
    example pandas or polars), or alternatively, if it can be converted to a numpy
    array.

    Parameters
    ----------
    y : Series or DataFrame (pandas, polars, or any other library supported by
        narwhals), np.array, list
        The input to check and copy or transform.

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
        if nw_y.is_null().any():
            raise ValueError("y contains NaN values.")
        if nw_y.dtype.is_numeric():
            if not np.isfinite(nw_y.to_numpy()).all():
                raise ValueError("y contains infinity values.")
        elif y_numeric:
            nw_y = nw_y.cast(nw.Float64())
        return _copy_series(nw_y)

    if nwd.is_into_dataframe(y):
        nw_y = nw.from_native(y, eager_only=True)
        if nw_y.select(nw.all().is_null().any()).to_numpy().any():
            raise ValueError("y contains NaN values.")
        if not np.isfinite(nw_y.to_numpy()).all():
            raise ValueError("y contains infinity values.")
        return nw_y.clone().to_native()

    try:
        y = column_or_1d(y)
        return _check_y(y, multi_output=False, y_numeric=y_numeric)
    except ValueError:
        return _check_y(y, multi_output=True, y_numeric=y_numeric)


def check_X_y(
    X,
    y: Union[np.generic, np.ndarray, List],
    y_numeric: bool = False,
):
    """
    Ensures X and y are compatible dataframe/array-like objects with a consistent
    number of rows. If both are pandas objects, also checks that their indexes match.

    Parameters
    ----------
    X: dataframe (pandas, polars, or any other library supported by narwhals), or
        numpy ndarray
        The input to check and copy or transform.

    y: Series (pandas, polars, or any other library supported by narwhals), np.array,
        list
        The input to check and copy or transform.

    y_numeric : bool, default=False
        Whether to ensure that y has a numeric type. If dtype of y is not numeric,
        it is cast to float64. Should only be used for regression algorithms.

    Raises
    ------
    ValueError: if X and y have a different number of rows, or if they are pandas
        objects with inconsistent indexes.
    TypeError: if X is a sparse matrix, an empty dataframe, or not a recognised type.

    Returns
    -------
    X: dataframe or numpy array
    y: Series, DataFrame, or numpy array
    """
    X = check_X(X)
    y = check_y(y, y_numeric=y_numeric)
    check_consistent_length(X, y)

    # A numpy array, or a dataframe/Series from a library other than pandas, has no
    # index to reconcile. `is_pandas_dataframe`/`is_pandas_series`, unlike
    # `isinstance(X, pd.DataFrame)`, don't require pandas to be installed.
    if nwd.is_pandas_dataframe(X) and (
        nwd.is_pandas_series(y) or nwd.is_pandas_dataframe(y)
    ):
        if not X.index.equals(y.index):  # type: ignore[union-attr]
            raise ValueError("The indexes of X and y do not match.")

    return X, y


def _check_X_matches_training_df(X, reference: int) -> None:
    """
    Checks that the dataframe to transform has the same number of columns as the
    dataframe used with the fit() method.

    Parameters
    ----------
    X : dataframe
        The df to be checked
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
    X,
    variables: List[Union[str, int]],
) -> None:
    """
    Checks if the dataframe contains null values in the selected columns.

    Parameters
    ----------
    X : dataframe

    variables : List
        The selected group of variables in which null values will be examined.

    Raises
    ------
    ValueError
        If the variable(s) contain null values.
    """
    nw_X = nw.from_native(X, eager_only=True)
    if nw_X.select(nw.col(variables).is_null().any()).to_numpy().any():
        raise ValueError(
            "Some of the variables in the dataset contain NaN. Check and "
            "remove those before using this transformer."
        )


def _check_optional_contains_na(X, variables: List[Union[str, int]]) -> None:
    """
    Checks if the dataframe contains null values in the selected columns.

    Parameters
    ----------
    X : dataframe

    variables : List
        The selected group of variables in which null values will be examined.

    Raises
    ------
    ValueError
        If the variable(s) contain null values.
    """
    nw_X = nw.from_native(X, eager_only=True)
    if nw_X.select(nw.col(variables).is_null().any()).to_numpy().any():
        raise ValueError(
            "Some of the variables in the dataset contain NaN. Check and "
            "remove those before using this transformer or set the parameter "
            "`missing_values='ignore'` when initialising this transformer."
        )


def _check_contains_inf(X, variables: List[Union[str, int]]) -> None:
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
