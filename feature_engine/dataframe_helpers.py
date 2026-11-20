"""Functions to move data between dataframes and numpy arrays.

Transformers call these functions instead of handling each dataframe library
themselves. A library-specific fast path, if one is needed, lives here.

All functions take the dataframe returned by `check_X`: a pandas dataframe, or a
narwhals dataframe for the other libraries.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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


def map_categories_in_df(
    X: DataFrame, mappings: Dict[Any, Dict[Any, Any]], default: Any = None
) -> DataFrame:
    """
    Replaces the categories of existing columns with the values in a mapping.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    mappings : dict
        The names of the columns to replace and, for each column, a dictionary
        with the categories as keys and their new values.

    default : default=None
        The value for categories that are not in the mapping, and for missing
        values. If None, they become missing values.

    Returns
    -------
    X : pandas or narwhals dataframe
        A new dataframe with the replaced columns, in the original column order.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        new_columns = {}
        for var, mapping in mappings.items():
            values = X[var].map(mapping)
            # a categorical column stays categorical after map().
            if values.dtype.name == "category":
                if all(isinstance(x, int) for x in values):
                    values = values.astype("int")
                else:
                    values = values.astype("float")
            if default is not None:
                values = values.fillna(default)
            new_columns[var] = values.to_numpy()
        return replace_columns_in_df(X, new_columns)
    else:
        return X.with_columns(
            X.get_column(var).replace_strict(mapping, default=default)
            for var, mapping in mappings.items()
        )


def add_category_to_df(
    X: DataFrame, variables: Sequence[Union[str, int]], category: Any
) -> DataFrame:
    """
    Adds a category to the categorical variables, so that their values can be
    replaced by it. Only pandas categorical columns need this: the other columns,
    and the other libraries, take the new value as is.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables that will take the new category.

    category :
        The new category.

    Returns
    -------
    X : pandas or narwhals dataframe
        The dataframe with the new category in its pandas categorical variables.
    """
    if not isinstance(X, nw.DataFrame):
        new_columns = {
            var: X[var].cat.add_categories(category)
            for var in variables
            if X[var].dtype.name == "category"
        }
        if len(new_columns) > 0:
            return replace_columns_in_df(X, new_columns)
    return X


def missing_values_mask(
    X: DataFrame, variables: Sequence[Union[str, int]]
) -> np.ndarray:
    """
    Returns where the variables have missing values.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables.

    Returns
    -------
    mask : numpy array of shape = [n_samples, len(variables)]
        True where the value is missing.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        return X[list(variables)].isna().to_numpy()
    elif len(variables) == 0:
        return np.zeros((X.shape[0], 0), dtype=bool)
    else:
        return X.select(nw.col(var).is_null() for var in variables).to_numpy()


def missing_values_count(
    X: DataFrame, variables: Sequence[Union[str, int]]
) -> List[int]:
    """
    Returns the number of missing values of each variable.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables.

    Returns
    -------
    counts : list
        The number of missing values, in the order of `variables`.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        return X[list(variables)].isna().sum().tolist()
    elif len(variables) == 0:
        return []
    else:
        return list(X.select(variables).null_count().row(0))


def aggregate_columns(
    X: DataFrame,
    variables: Sequence[Union[str, int]],
    method: str,
    quantile: Optional[float] = None,
) -> Dict[Any, Any]:
    """
    Returns a statistic of each variable, ignoring missing values.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the numerical variables.

    method : str
        The statistic: "mean", "median", "std", "max" or "quantile".

    quantile : float, default=None
        The quantile to compute, with linear interpolation, when method is
        "quantile".

    Returns
    -------
    statistics : dict
        The variable names and their statistic.
    """
    args: Tuple[Any, ...] = () if quantile is None else (quantile,)
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        return getattr(X[list(variables)], method)(*args).to_dict()
    else:
        if quantile is not None:
            args = (quantile, "linear")
        statistics = X.select(
            getattr(nw.col(var), method)(*args) for var in variables
        )
        statistics = statistics.to_dict(as_series=False)
        return {var: values[0] for var, values in statistics.items()}


def fill_missing_in_df(X: DataFrame, values: Dict[Any, Any]) -> DataFrame:
    """
    Replaces the missing values of each variable with a value.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    values : dict
        The names of the variables and the value that replaces their missing
        values.

    Returns
    -------
    X : pandas or narwhals dataframe
        A new dataframe without missing values in the variables.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        return X.fillna(value=values).infer_objects()
    else:
        schema = X.schema
        for var, value in values.items():
            # a polars Enum can't take new categories: fill_null would leave
            # the value missing without raising.
            if isinstance(schema[var], nw.Enum) and (
                value not in schema[var].categories
            ):
                raise ValueError(
                    f"Cannot fill variable '{var}' with '{value}': it is a "
                    f"polars Enum with fixed categories {schema[var].categories} "
                    "that do not include the fill value. Cast the column to "
                    "Categorical or String before imputing."
                )
        return X.with_columns(
            nw.col(var).fill_null(value) for var, value in values.items()
        )


def filter_rows(X: DataFrame, mask: np.ndarray) -> DataFrame:
    """
    Returns the rows of the dataframe where the mask is True.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    mask : numpy array of shape = [n_samples]
        Boolean array, True for the rows to keep.

    Returns
    -------
    X : pandas or narwhals dataframe
        The dataframe with the selected rows.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        return X[mask]
    else:
        return X.filter(nw.new_series("__mask__", mask, backend=X.implementation))
