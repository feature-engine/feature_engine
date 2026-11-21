"""Functions to pass data to a scikit-learn transformer and take its output back.

The wrappers call these functions instead of handling each dataframe library
themselves.

All functions take the dataframe returned by `check_X`: a pandas dataframe, or a
narwhals dataframe for the other libraries.
"""

from typing import Any, List, Sequence, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from narwhals.typing import IntoDataFrame

DataFrame = Union[IntoDataFrame, nw.DataFrame]


def set_sklearn_output(transformer: Any, X: DataFrame) -> None:
    """
    Sets the output container of the scikit-learn transformer for the dataframe
    library of X, so that a global scikit-learn configuration can't change it.

    Parameters
    ----------
    transformer : scikit-learn transformer
        The unfitted transformer.

    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.
    """
    # pandas is faster than narwhals. FunctionTransformer warns with "pandas"
    # when its function returns an array.
    if (
        not isinstance(X, nw.DataFrame)
        and transformer.__class__.__name__ != "FunctionTransformer"
    ):
        transformer.set_output(transform="pandas")
    else:
        transformer.set_output(transform="default")


def to_sklearn_input(
    X: DataFrame, variables: Sequence[Union[str, int]], as_native: bool = False
) -> Any:
    """
    Returns the variables in the format passed to the scikit-learn transformer.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables to pass to the transformer.

    as_native : bool, default=False
        Whether to return the variables in the dataframe library of the user, as
        needed by the function of a FunctionTransformer. pandas dataframes are
        always returned as they are.

    Returns
    -------
    X : pandas dataframe, dataframe of the user's library or numpy array
        The variables.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        return X[list(variables)]

    X_vars = X.select(list(variables))
    if as_native is True:
        return X_vars.to_native()

    X_np = X_vars.to_numpy()
    # scikit-learn treats NaN, not None, as missing in text columns, which is
    # what it gets from pandas.
    if X_np.dtype == object and sum(X_vars.null_count().row(0)) > 0:
        X_np[np.equal(X_np, None)] = np.nan
    return X_np


def replace_with_sklearn_output(
    X: DataFrame, X_new: Any, variables: List[Union[str, int]]
) -> DataFrame:
    """
    Replaces the variables with the output of the scikit-learn transformer.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    X_new : numpy array or dataframe
        The output of the transformer, with one column per variable.

    variables : list
        The names of the variables to replace.

    Returns
    -------
    X : pandas or narwhals dataframe
        A new dataframe with the replaced variables, in the original column order.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        # assigning whole columns replaces them in the copy and leaves the user's
        # data untouched, so a shallow copy is enough.
        X = X.copy(deep=False)
        X[variables] = X_new
        return X
    else:
        X_new = _sklearn_output_to_df(X, X_new, variables)
        return X.with_columns(*X_new.iter_columns())


def add_sklearn_output(
    X: DataFrame, X_new: Any, transformer: Any, variables: List[Union[str, int]]
) -> DataFrame:
    """
    Adds the output of the scikit-learn transformer to the end of the dataframe.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`, without the transformed variables.

    X_new : numpy array or pandas dataframe
        The output of the transformer. With pandas, it is a dataframe with the
        index of X and the column names, as set by `set_sklearn_output`.

    transformer : scikit-learn transformer
        The fitted transformer, which names the new columns.

    variables : list
        The names of the variables passed to the transformer.

    Returns
    -------
    X : pandas or narwhals dataframe
        The dataframe with the new columns.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        return nwd.get_pandas().concat([X, X_new], axis=1)
    else:
        columns = list(transformer.get_feature_names_out(variables))
        X_new = _sklearn_output_to_df(X, X_new, columns)
        return nw.concat([X, X_new], how="horizontal")


def _sklearn_output_to_df(
    X: nw.DataFrame, X_new: Any, columns: List[Union[str, int]]
) -> nw.DataFrame:
    """Returns the output of the transformer as a narwhals dataframe with the
    given column names, in the dataframe library of X."""
    # the function in a FunctionTransformer may return a dataframe.
    if nwd.is_into_dataframe(X_new) is True:
        X_new = nw.from_native(X_new, eager_only=True)
        return X_new.rename(dict(zip(X_new.columns, columns)))

    # scikit-learn marks missing values with NaN, polars and others with null.
    if X_new.dtype == object:
        X_new[np.not_equal(X_new, X_new)] = None
    elif X_new.dtype.kind == "f" and bool(np.isnan(X_new).any()) is True:
        return nw.from_numpy(
            X_new, schema=columns, backend=X.implementation
        ).with_columns(nw.all().fill_nan(None))

    return nw.from_numpy(X_new, schema=columns, backend=X.implementation)
