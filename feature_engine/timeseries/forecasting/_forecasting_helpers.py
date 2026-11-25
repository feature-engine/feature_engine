"""Functions to order the rows of a dataframe in time and add forecasting features.

With pandas, the rows are ordered in time by the dataframe's index. Other
dataframes, like polars, have no index: their rows are taken in the order given.
The forecasting transformers call these functions instead of handling each
dataframe library themselves.

All functions take the dataframe returned by `check_X`: a pandas dataframe, or a
narwhals dataframe for the other libraries.
"""

from typing import Any, List, Optional, Sequence, Union

import narwhals as nw
from narwhals.typing import IntoDataFrame

DataFrame = Union[IntoDataFrame, nw.DataFrame]


def check_index(X: DataFrame, freq: Any = None) -> None:
    """
    Checks that the rows of the dataframe can be ordered in time. With pandas, the
    index must be unique and not contain missing data. Other dataframes have no
    index, so they can't use `freq`.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    freq : str, list or None, default=None
        The `freq` parameter of the transformer. Only needed in fit.
    """
    # only pandas has an index.
    if not isinstance(X, nw.DataFrame):
        if X.index.isnull().any():
            raise NotImplementedError(
                "The dataframe's index contains NaN values or missing data. "
                "Only dataframes with complete indexes are compatible with "
                "this transformer."
            )

        if X.index.is_unique is False:
            raise NotImplementedError(
                "The dataframe's index does not contain unique values. "
                "Only dataframes with unique values in the index are "
                "compatible with this transformer."
            )

    elif freq is not None:
        raise NotImplementedError(
            "freq is only supported with pandas dataframes, because it uses the "
            "dataframe's DatetimeIndex. With other dataframes, leave freq=None "
            f"to shift the rows by periods. Got {freq} instead."
        )


def sort_by_index(X: DataFrame) -> DataFrame:
    """
    Returns the dataframe with its rows sorted by the index. Dataframes without an
    index are returned as is: their rows are used in the order given.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    Returns
    -------
    X : pandas or narwhals dataframe
        The dataframe with the rows sorted by the index.
    """
    # only pandas has an index.
    if not isinstance(X, nw.DataFrame):
        return X.sort_index()
    else:
        return X


def add_lag_features(
    X: DataFrame,
    variables: Sequence[Union[str, int]],
    periods: List[int],
    freq: Optional[List[Any]],
    fill_value: Any,
    names: List[str],
) -> DataFrame:
    """
    Adds the lags of the variables after the columns of the dataframe.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables to lag.

    periods : list
        The number of rows to shift. Used when `freq` is None.

    freq : list or None
        The time spans to shift the pandas index by: strings, like "1h", or pandas
        offsets.

    fill_value :
        The value for the rows that have no past values. If None, they are missing.

    names : list
        The names of the lags: for each period or freq, one per variable.

    Returns
    -------
    X : pandas or narwhals dataframe
        The dataframe with the lags.
    """
    # pandas is faster than narwhals, and freq needs the pandas index.
    if not isinstance(X, nw.DataFrame):
        if freq is None:
            lags = [X[list(variables)].shift(periods=periods, fill_value=fill_value)]
        else:
            lags = [X[list(variables)].shift(freq=fr) for fr in freq]
        # joining one lag at a time is faster than aligning all lags at once.
        new_names = iter(names)
        for lag in lags:
            lag.columns = [next(new_names) for _ in lag.columns]
            X = X.join(lag)
        # shift() does not take fill_value together with freq.
        if freq is not None and fill_value is not None:
            X = X.fillna({name: fill_value for name in names})
    else:
        lags = list(zip([(var, pr) for pr in periods for var in variables], names))
        if fill_value is None:
            X = X.with_columns(
                nw.col(var).shift(pr).alias(name) for (var, pr), name in lags
            )
        elif X.implementation is nw.Implementation.POLARS:
            # polars is faster than narwhals: it fills the rows that shift()
            # introduces without copying the column.
            col = nw.get_native_namespace(X).col
            X_pl = X.to_native().with_columns(
                col(var).shift(pr, fill_value=fill_value).alias(name)
                for (var, pr), name in lags
            )
            X = nw.from_native(X_pl, eager_only=True)
        else:
            # narwhals' shift() has no fill_value. The mask is True in the first
            # rows, which have no past values, so missing data in X is not filled.
            X = X.with_columns(
                nw.when(nw.col(var).is_null().shift(pr).is_null())
                .then(nw.lit(fill_value))
                .otherwise(nw.col(var).shift(pr))
                .alias(name)
                for (var, pr), name in lags
            )

    return X
