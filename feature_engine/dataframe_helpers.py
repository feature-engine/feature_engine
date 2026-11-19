"""Functions to move data between dataframes and numpy arrays.

Transformers call these functions instead of handling each dataframe library
themselves. A library-specific fast path, if one is needed, lives here.

All functions take the dataframe returned by `check_X`: a pandas dataframe, or a
narwhals dataframe for the other libraries. They return the same kind.
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
        One 1D numpy array per variable, in the order of `variables`. The arrays
        can share memory with the dataframe, so they must not be modified in place.
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
        The dataframe with the new columns. The input is not modified.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        # adding the columns one by one fragments the dataframe
        new_X = type(X)(arrays, index=X.index)
        return nwd.get_pandas().concat([X, new_X], axis=1)
    else:
        return X.with_columns(
            nw.new_series(name, values, backend=X.implementation)
            for name, values in arrays.items()
        )


def subset_df(X: DataFrame, variables: Sequence[Union[str, int]]) -> DataFrame:
    """
    Returns the dataframe with only the variables, in the order of `variables`.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables to keep.

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
