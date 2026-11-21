"""Functions to order the rows of a dataframe in time and add forecasting features.

With pandas, the rows are ordered in time by the dataframe's index. Other
dataframes, like polars, have no index: their rows are taken in the order given.
The forecasting transformers call these functions instead of handling each
dataframe library themselves.

All functions take the dataframe returned by `check_X`: a pandas dataframe, or a
narwhals dataframe for the other libraries.
"""

from itertools import product
from typing import Any, List, Optional, Sequence, Union

import narwhals as nw
from narwhals.typing import IntoDataFrame

DataFrame = Union[IntoDataFrame, nw.DataFrame]

# pandas' rolling functions that other dataframes support: with narwhals, with
# polars' rolling expressions when narwhals has none (they are faster than numpy),
# and with polars aggregations over the rows of each window when polars has none.
_NW_ROLLING = {
    "count": lambda col, win, min_samples: (~col.is_null())
    .cast(nw.Float64)
    .rolling_sum(win, min_samples=min_samples),
    "mean": lambda col, win, min_samples: col.rolling_mean(
        win, min_samples=min_samples
    ),
    "sem": lambda col, win, min_samples: col.rolling_std(win, min_samples=min_samples)
    / (~col.is_null()).cast(nw.Float64).rolling_sum(win, min_samples=1).sqrt(),
    "std": lambda col, win, min_samples: col.rolling_std(win, min_samples=min_samples),
    "sum": lambda col, win, min_samples: col.rolling_sum(win, min_samples=min_samples),
    "var": lambda col, win, min_samples: col.rolling_var(win, min_samples=min_samples),
}
_POLARS_ROLLING = {
    # bias=False gives the sample skewness and kurtosis, as pandas.
    "kurt": lambda col, win, min_samples: col.rolling_kurtosis(
        win, bias=False, min_samples=min_samples
    ),
    "max": lambda col, win, min_samples: col.rolling_max(win, min_samples=min_samples),
    "median": lambda col, win, min_samples: col.rolling_median(
        win, min_samples=min_samples
    ),
    "min": lambda col, win, min_samples: col.rolling_min(win, min_samples=min_samples),
    "rank": lambda col, win, min_samples: col.rolling_rank(
        win, min_samples=min_samples
    ),
    "skew": lambda col, win, min_samples: col.rolling_skew(
        win, bias=False, min_samples=min_samples
    ),
}
# like pandas, they skip missing values.
_POLARS_ROW_WINDOWS = {
    "first": lambda col: col.drop_nulls().first(),
    "last": lambda col: col.drop_nulls().last(),
    "nunique": lambda col: col.drop_nulls().n_unique(),
}


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


def check_functions_supported(
    X: DataFrame, functions: Union[str, List[str]], supported: List[str]
) -> None:
    """
    Checks that dataframes other than pandas use functions that they support.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    functions : str or list
        The `functions` parameter of the transformer.

    supported : list
        The functions supported with dataframes other than pandas.
    """
    # pandas supports every function.
    if not isinstance(X, nw.DataFrame):
        return

    function_list = functions if isinstance(functions, list) else [functions]
    if not all(function in supported for function in function_list):
        raise NotImplementedError(
            "With dataframes other than pandas, functions takes only "
            f"{sorted(supported)}. Got {functions} instead."
        )


def check_window_parameters(
    X: DataFrame,
    window: Any,
    functions: Union[str, List[str]],
    min_periods: Optional[int],
) -> None:
    """
    Checks that dataframes without an index use windows of rows, and functions and
    min_periods that they support.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    window : int, str or list
        The `window` parameter of WindowFeatures.

    functions : str or list
        The `functions` parameter of WindowFeatures.

    min_periods : int or None
        The minimum number of values in a window to return a value.
    """
    # only pandas has an index, and pandas supports every window.
    if isinstance(X, nw.DataFrame):
        windows = window if isinstance(window, list) else [window]
        if not all(isinstance(win, int) for win in windows):
            raise NotImplementedError(
                "Time spans in window, like '3D', are only supported with pandas "
                "dataframes, because they use the dataframe's DatetimeIndex. With "
                "other dataframes, window takes integers, the number of rows in "
                f"each window. Got {window} instead."
            )
        function_list = functions if isinstance(functions, list) else [functions]
        supported = sorted([*_NW_ROLLING, *_POLARS_ROLLING, *_POLARS_ROW_WINDOWS])
        if not all(function in supported for function in function_list):
            raise NotImplementedError(
                "With dataframes other than pandas, functions takes only "
                f"{supported}. Got {functions} instead."
            )
        if min_periods == 0:
            raise NotImplementedError(
                "min_periods=0 is only supported with pandas dataframes. With "
                "other dataframes, min_periods takes integers greater than 0 or "
                f"None. Got {min_periods} instead."
            )


def add_window_features(
    X: DataFrame,
    variables: Sequence[Union[str, int]],
    windows: List[Any],
    functions: List[str],
    min_periods: Optional[int],
    periods: int,
    freq: Optional[str],
    names: List[str],
) -> DataFrame:
    """
    Adds functions of the values in a window of past rows, after the columns of the
    dataframe.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables.

    windows : list
        The window sizes: numbers of rows, or time spans with pandas.

    functions : list
        The functions to apply to each window.

    min_periods : int or None
        The minimum number of values in a window to return a value. If None, the
        window size.

    periods : int
        The number of rows to shift the windows by, so they hold past values only.

    freq : str or None
        The time span to shift the pandas index by, instead of `periods`.

    names : list
        The names of the new features: for each window, variable and function.

    Returns
    -------
    X : pandas or narwhals dataframe
        The dataframe with the new features.
    """
    # pandas is faster than narwhals, and time spans and freq need its index.
    name_of = dict(zip(product(windows, variables, functions), names))
    if not isinstance(X, nw.DataFrame):
        tmp = []
        for win in windows:
            rolling = X[list(variables)].rolling(window=win, min_periods=min_periods)
            for function in functions:
                # the rolling methods are faster than agg() with many variables.
                rolled = getattr(rolling, function)().shift(periods=periods, freq=freq)
                rolled.columns = [name_of[(win, var, function)] for var in variables]
                tmp.append(rolled)
        # with freq, the index of the features moves: merge aligns them to X.
        X = X.merge(
            tmp[0].join(tmp[1:])[names], left_index=True, right_index=True, how="left"
        )
    else:
        plx = nw.get_native_namespace(X)
        nw_exprs, native_exprs = [], []
        for (win, var, function), name in name_of.items():
            # like pandas, windows of integers need all their rows by default.
            min_samples = win if min_periods is None else min_periods
            if function in _NW_ROLLING:
                rolled = _NW_ROLLING[function](nw.col(var), win, min_samples)
                nw_exprs.append(rolled.shift(periods).alias(name))
            elif function in _POLARS_ROLLING:
                rolled = _POLARS_ROLLING[function](plx.col(var), win, min_samples)
                native_exprs.append(rolled.shift(periods).alias(name))
            else:
                # like pandas, a window needs min_samples values that are not missing.
                not_missing = plx.col(var).is_not_null().cast(plx.UInt32)
                rolled = plx.when(
                    not_missing.rolling_sum(win, min_samples=1) >= min_samples
                ).then(
                    _POLARS_ROW_WINDOWS[function](plx.col(var)).rolling(
                        index_column="__row__", period=f"{win}i"
                    )
                )
                native_exprs.append(rolled.shift(periods).alias(name))

        columns = X.columns
        X = X.with_columns(nw_exprs)
        if len(native_exprs) > 0:
            # the polars features were added last: restore the order of the names,
            # which also drops the row numbers that the windows of rows need.
            X_pl = X.to_native().with_row_index("__row__").with_columns(native_exprs)
            X = nw.from_native(X_pl, eager_only=True).select(columns + names)

    return X


def add_expanding_window_features(
    X: DataFrame,
    variables: Sequence[Union[str, int]],
    functions: Union[str, List[str]],
    min_periods: Optional[int],
    periods: int,
    freq: Optional[str],
    ignore_missing: bool,
    names: List[str],
) -> DataFrame:
    """
    Adds functions of all the past values of the variables, after the columns of the
    dataframe.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the variables.

    functions : str or list
        The functions to apply to the past values.

    min_periods : int or None
        The minimum number of values to return a value. If None, 0.

    periods : int
        The number of rows to shift the features by, so they hold past values only.

    freq : str or None
        The time span to shift the pandas index by, instead of `periods`.

    ignore_missing : bool
        Whether the variables can have missing values, NaN and inf.

    names : list
        The names of the new features: for each variable, one per function.

    Returns
    -------
    X : pandas or narwhals dataframe
        The dataframe with the new features.
    """
    # pandas is faster than narwhals, and freq shifts the pandas index. agg() is
    # faster with one function as a string than as a list.
    if not isinstance(X, nw.DataFrame):
        tmp = (
            X[list(variables)]
            .expanding(min_periods=min_periods)
            .agg(functions)
            .shift(periods=periods, freq=freq)
        )
        tmp.columns = names
        return X.merge(tmp, left_index=True, right_index=True, how="left")

    # pandas treats min_periods=None as 0. narwhals' rolling functions need at
    # least 1, and 0 only changes the sum, done below.
    min_periods = 0 if min_periods is None else min_periods
    min_samples = max(min_periods, 1)
    # a rolling window as long as the dataframe is an expanding window.
    window = max(len(X), min_samples)
    function_list = functions if isinstance(functions, list) else [functions]
    new_features = []
    for var in variables:
        # pandas computes in float and treats NaN as missing data, and inf too,
        # except when counting.
        x = nw.col(var).cast(nw.Float64)
        if ignore_missing is True:
            n_values = x.fill_nan(None).cum_count()
            x = nw.when(x.is_finite()).then(x)
        else:
            n_values = x.cum_count()
        count = x.cum_count()
        for func in function_list:
            if func == "count":
                # pandas compares min_periods with the number of rows for count.
                feature = nw.when(x.is_null().cum_count() >= min_periods).then(
                    n_values.cast(nw.Float64)
                )
            elif func in ("min", "max"):
                # cumulative min and max are faster than rolling ones, but
                # they are null in the rows with missing data.
                cum = x.cum_min() if func == "min" else x.cum_max()
                feature = nw.when(count >= min_samples).then(
                    cum.fill_null(strategy="forward")
                )
            elif func == "median":
                # narwhals has no rolling median: use the polars series.
                values = X.select(x).get_column(var).to_native()
                feature = nw.from_native(
                    values.rolling_median(window, min_samples=min_samples),
                    series_only=True,
                )
            else:
                feature = getattr(x, f"rolling_{func}")(
                    window, min_samples=min_samples
                )
                if func == "sum" and min_periods == 0:
                    feature = feature.fill_null(0)
            new_features.append(feature.shift(periods))

    return X.with_columns(
        feature.alias(name) for feature, name in zip(new_features, names)
    )
