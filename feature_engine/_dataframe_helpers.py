"""Functions to move data between dataframes and numpy arrays.

Transformers call these functions instead of handling each dataframe library
themselves. A library-specific fast path, if one is needed, lives here.
"""

from typing import Any, Dict, List, Sequence, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np


def to_numpy(
    nw_X: nw.DataFrame, variables: Sequence[Union[str, int]]
) -> List[np.ndarray]:
    """
    Returns the variables as numpy arrays.

    Parameters
    ----------
    nw_X : narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables to convert.

    Returns
    -------
    arrays : list
        One 1D numpy array per variable, in the order of `variables`. The arrays
        can share memory with the dataframe, so they must not be modified in place.
    """
    X = nw_X.to_native()
    # pandas is faster than narwhals.
    if nwd.is_pandas_dataframe(X) is True:
        arrays = []
        for var in variables:
            col = X[var]
            # extension dtypes, like Int64, need narwhals to turn nulls into NaN
            if isinstance(col.dtype, np.dtype):
                arrays.append(col.to_numpy())
            else:
                arrays.append(nw_X.get_column(var).to_numpy())
        return arrays
    else:
        return [nw_X.get_column(var).to_numpy() for var in variables]


def add_arrays(nw_X: nw.DataFrame, arrays: Dict[Any, np.ndarray]) -> nw.DataFrame:
    """
    Adds numpy arrays to the dataframe as columns. An array named after a column
    of the dataframe replaces that column. New columns go at the end.

    Parameters
    ----------
    nw_X : narwhals dataframe
        The dataframe returned by `check_X`.

    arrays : dict
        The column names and the 1D numpy array with the values of each column.

    Returns
    -------
    nw_X : narwhals dataframe
        The dataframe with the arrays as columns. The input is not modified.
    """
    X = nw_X.to_native()
    # pandas is faster than narwhals.
    if nwd.is_pandas_dataframe(X) is True:
        X = X.copy(deep=False)
        new_columns = {}
        for name, values in arrays.items():
            if name in X.columns:
                X[name] = values
            else:
                new_columns[name] = values
        # adding the new columns one by one fragments the dataframe
        new_X = type(X)(new_columns, index=X.index)
        return nw.concat(
            [
                nw.from_native(X, eager_only=True),
                nw.from_native(new_X, eager_only=True),
            ],
            how="horizontal",
        )
    else:
        return nw_X.with_columns(
            nw.new_series(name, values, backend=nw_X.implementation)
            for name, values in arrays.items()
        )


def select_columns(
    nw_X: nw.DataFrame, variables: List[Union[str, int]]
) -> nw.DataFrame:
    """
    Returns the dataframe with the selected variables, in the order of `variables`.

    Parameters
    ----------
    nw_X : narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables to select.

    Returns
    -------
    nw_X : narwhals dataframe
        The dataframe with the selected variables.
    """
    X = nw_X.to_native()
    # pandas is faster than narwhals.
    if nwd.is_pandas_dataframe(X) is True:
        return nw.from_native(X[variables], eager_only=True)
    else:
        return nw_X.select(nw.col(*variables))
