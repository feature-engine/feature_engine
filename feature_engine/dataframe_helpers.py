"""Functions to move data between dataframes and numpy arrays.

Transformers call these functions instead of handling each dataframe library
themselves. A library-specific fast path, if one is needed, lives here.

All functions take the dataframe returned by `check_X`: a pandas dataframe, or a
narwhals dataframe for the other libraries.
"""

from typing import Any, Dict, List, Sequence, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from narwhals.typing import IntoDataFrame

DataFrame = Union[IntoDataFrame, nw.DataFrame]


def to_numpy(X: DataFrame, variables: Sequence[Union[str, int]]) -> List[np.ndarray]:
    """
    Returns the variables as numpy arrays.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the numerical variables to convert.

    Returns
    -------
    arrays : list
        One 1D numpy array per variable, in the order of `variables`. Needed
        to preserve dtype.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        return [X[var].to_numpy() for var in variables]
    else:
        return [X.get_column(var).to_numpy() for var in variables]


def add_columns_to_df(X: DataFrame, arrays: Dict[Any, np.ndarray]) -> DataFrame:
    """
    Adds numpy arrays to the end of the dataframe as new columns.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    arrays : dict
        The names of the new columns and the 1D numpy array with the values of
        each column.

    Returns
    -------
    X : pandas or narwhals dataframe
        The dataframe with the new columns.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        # add all columns at once to avoid fragmentation warning
        new_X = type(X)(arrays, index=X.index)
        return nwd.get_pandas().concat([X, new_X], axis=1)
    else:
        return X.with_columns(
            nw.new_series(name, values, backend=X.implementation)
            for name, values in arrays.items()
        )


def subset_df(X: DataFrame, variables: Sequence[Union[str, int]]) -> DataFrame:
    """
    Returns the dataframe with the variables in the order of `variables`.
    Used in transform methods to ensure the output has the same order
    of columns as the train set.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables.

    Returns
    -------
    X : pandas or narwhals dataframe
        The dataframe with only the variables.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        return X[list(variables)]
    else:
        return X.select(list(variables))


def drop_variables_from_df(
    X: DataFrame, variables: Sequence[Union[str, int]]
) -> DataFrame:
    """
    Returns the dataframe without the variables.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables to drop.

    Returns
    -------
    X : pandas or narwhals dataframe
        The dataframe without the variables.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        return X.drop(columns=list(variables))
    else:
        return X.drop(list(variables))


def replace_columns_in_df(X: DataFrame, arrays: Dict[Any, np.ndarray]) -> DataFrame:
    """
    Replaces the values of existing columns with numpy arrays.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    arrays : dict
        The names of the columns to replace and the 1D numpy array with the new
        values of each column.

    Returns
    -------
    X : pandas or narwhals dataframe
        A new dataframe with the replaced columns, in the original column order.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        # replace all columns at once to avoid fragmentation warning
        new_X = type(X)(arrays, index=X.index)
        X_new = nwd.get_pandas().concat(
            [X.drop(columns=list(arrays)), new_X], axis=1
        )
        return X_new[X.columns]
    else:
        return X.with_columns(
            nw.new_series(name, values, backend=X.implementation)
            for name, values in arrays.items()
        )
