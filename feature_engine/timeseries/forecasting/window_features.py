from itertools import product
from typing import Callable, List, Union

import narwhals as nw
import narwhals.dependencies as nwd
from narwhals.typing import IntoDataFrame

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _drop_original_docstring,
    _missing_values_docstring,
    _return_empty_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_not_learn_docstring,
    _fit_transform_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.timeseries.forecasting.base_forecast_transformers import (
    BaseForecastTransformer,
)

# pandas' rolling functions that other dataframes support: with narwhals, or with
# polars' own expressions when narwhals has none (they are faster than numpy).
_NW_ROLLING = {
    "count": lambda col, win, min_samples: (~col.is_null())
    .cast(nw.Float64)
    .rolling_sum(win, min_samples=min_samples),
    "mean": lambda col, win, min_samples: col.rolling_mean(
        win, min_samples=min_samples
    ),
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
    "skew": lambda col, win, min_samples: col.rolling_skew(
        win, bias=False, min_samples=min_samples
    ),
}


@Substitution(
    variables=_variables_numerical_docstring,
    return_empty=_return_empty_docstring,
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit=_fit_not_learn_docstring,
    fit_transform=_fit_transform_docstring,
)
class WindowFeatures(BaseForecastTransformer):
    """
    WindowFeatures adds new features to a dataframe based on window operations. Window
    operations are operations that perform an aggregation over a sliding partition of
    past values. A window feature is, in other words, a feature created after computing
    statistics (e.g., mean, min, max, etc.) using a window over the past data. For
    example, the mean value of the previous 3 months of data is a window feature. The
    maximum value of the previous three rows of data is another window feature.

    WindowFeatures works with pandas and polars dataframes.

    With pandas, WindowFeatures uses pandas functions `rolling()` and `shift()`. With
    `rolling()`, it creates rolling windows and applies the functions within those
    windows. With `shift()` it allocates the values to the correct rows. For supported
    aggregation functions, see Rolling Window
    `Functions <https://pandas.pydata.org/docs/reference/window.html>`_.

    With pandas `rolling()` we can perform rolling operations over 1 window size at a
    time. WindowFeatures builds on top of pandas `rolling()` in that new features can
    be derived from multiple window sizes, and the created features will be
    automatically concatenated to the original dataframe.

    With pandas, the rows are ordered in time by the dataframe's index, which must have
    unique values and no missing data. Polars dataframes have no index: their rows are
    taken in the order given, so they must be sorted in time. With polars, the windows
    are a number of rows, and `freq` is not supported.

    WindowFeatures works only with numerical variables. You can pass a list of variables
    to use as input for the windows. Alternatively, WindowFeatures will automatically
    select all numerical variables in the training set.

    More details in the :ref:`User Guide <window_features>`.

    Parameters
    ----------
    {variables}

    {return_empty}

    window: int, offset, BaseIndexer subclass, or list, default=3
        Size of the moving window. If an integer, the fixed number of observations used
        for each window. If an offset (recommended), the time period of each window. It
        can also take a function. See parameter `windows` in pandas `rolling()`
        documentation for more details.

        In addition to pandas normal input values, `window` can also take a list with
        the above specified values, in which case, features will be created for each
        one of the windows specified in the list.

        With polars, `window` takes only integers or lists of integers.

    min_periods: int, default None.
        Minimum number of observations in the window required to have a value;
        otherwise, the result is np.nan. See parameter `min_periods` in pandas
        `rolling()` documentation for more details. With polars, `min_periods` must be
        greater than 0.

    functions: string or list of strings, default = 'mean'
        The functions to apply within the window. Valid functions can be found
        `here <https://pandas.pydata.org/docs/reference/window.html>`_. With polars,
        the valid functions are 'count', 'kurt', 'max', 'mean', 'median', 'min',
        'skew', 'std', 'sum' and 'var'.

    periods: int, default=1
        Number of periods to shift. Can be a positive integer. See param `periods` in
        pandas `shift()`.

    freq: str, default=None
        Offset to use from the tseries module or time rule. See parameter `freq` in
        pandas `shift()`. Only supported with pandas.

    sort_index: bool, default=True
        Whether to order the index of the dataframe before creating the features. It
        does not apply to polars dataframes, which have no index.

    {missing_values}

    {drop_original}

    drop_na: bool, default=False.
        Whether the NAN introduced in the lag features should be removed.

    Attributes
    ----------
    variables_:
        The group of variables that will be used to create the window features.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    transform:
        Add window features.

    transform_x_y:
        Remove rows with missing data from X and y.

    {fit_transform}

    See Also
    --------
    pandas.rolling
    pandas.aggregate
    pandas.shift

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.timeseries.forecasting import WindowFeatures
    >>> X = pd.DataFrame(dict(date = ["2022-09-18",
    >>>                               "2022-09-19",
    >>>                               "2022-09-20",
    >>>                               "2022-09-21",
    >>>                               "2022-09-22"],
    >>>                       x1 = [1,2,3,4,5],
    >>>                       x2 = [6,7,8,9,10]
    >>>                     ))
    >>> wf = WindowFeatures(window = 2)
    >>> wf.fit_transform(X)
             date  x1  x2  x1_window_2_mean  x2_window_2_mean
    0  2022-09-18   1   6               NaN               NaN
    1  2022-09-19   2   7               NaN               NaN
    2  2022-09-20   3   8               1.5               6.5
    3  2022-09-21   4   9               2.5               7.5
    4  2022-09-22   5  10               3.5               8.5

    With polars, the rows are taken in the order given:

    >>> import polars as pl
    >>> X = pl.DataFrame(dict(x1 = [1,2,3,4,5], x2 = [6,7,8,9,10]))
    >>> wf = WindowFeatures(window = 2)
    >>> wf.fit_transform(X)
    shape: (5, 4)
    ┌─────┬─────┬──────────────────┬──────────────────┐
    │ x1  ┆ x2  ┆ x1_window_2_mean ┆ x2_window_2_mean │
    │ --- ┆ --- ┆ ---              ┆ ---              │
    │ i64 ┆ i64 ┆ f64              ┆ f64              │
    ╞═════╪═════╪══════════════════╪══════════════════╡
    │ 1   ┆ 6   ┆ null             ┆ null             │
    │ 2   ┆ 7   ┆ null             ┆ null             │
    │ 3   ┆ 8   ┆ 1.5              ┆ 6.5              │
    │ 4   ┆ 9   ┆ 2.5              ┆ 7.5              │
    │ 5   ┆ 10  ┆ 3.5              ┆ 8.5              │
    └─────┴─────┴──────────────────┴──────────────────┘
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        window: Union[str, int, Callable, List[int], List[str]] = 3,
        min_periods: Union[int, None] = None,
        functions: Union[str, List[str]] = "mean",
        periods: int = 1,
        freq: Union[str, None] = None,
        sort_index: bool = True,
        missing_values: str = "raise",
        drop_original: bool = False,
        drop_na: bool = False,
    ) -> None:

        if isinstance(window, list) and len(window) != len(set(window)):
            raise ValueError(f"There are duplicated windows in the list: {window}")

        if not isinstance(functions, (str, list)) or not all(
            isinstance(val, str) for val in functions
        ):
            raise ValueError(
                f"functions must be a string or a list of strings. "
                f"Got {functions} instead."
            )
        if isinstance(functions, list) and len(functions) != len(set(functions)):
            raise ValueError(f"There are duplicated functions in the list: {functions}")

        if not isinstance(periods, int) or periods < 1:
            raise ValueError(
                f"periods must be a positive integer. Got {periods} instead."
            )

        super().__init__(
            variables, return_empty, missing_values, drop_original, drop_na
        )

        self.window = window
        self.min_periods = min_periods
        self.functions = functions
        self.periods = periods
        self.freq = freq
        self.sort_index = sort_index

    def _check_index(self, X: IntoDataFrame):
        """
        Checks that the rows of the dataframe can be ordered in time and, with
        dataframes other than pandas, that `window`, `functions` and `min_periods`
        are supported.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The dataset.
        """
        super()._check_index(X)

        if nwd.is_pandas_dataframe(X) is False:
            windows = self.window if isinstance(self.window, list) else [self.window]
            if not all(isinstance(win, int) for win in windows):
                raise NotImplementedError(
                    "Time spans in window, like '3D', are only supported with pandas "
                    "dataframes, because they use the dataframe's DatetimeIndex. With "
                    "other dataframes, window takes integers, the number of rows in "
                    f"each window. Got {self.window} instead."
                )

            functions = (
                self.functions if isinstance(self.functions, list) else [self.functions]
            )
            if not all(
                function in _NW_ROLLING or function in _POLARS_ROLLING
                for function in functions
            ):
                raise NotImplementedError(
                    "With dataframes other than pandas, functions takes only "
                    f"{sorted([*_NW_ROLLING, *_POLARS_ROLLING])}. "
                    f"Got {self.functions} instead."
                )

            if self.min_periods == 0:
                raise NotImplementedError(
                    "min_periods=0 is only supported with pandas dataframes. With "
                    "other dataframes, min_periods takes integers greater than 0 or "
                    f"None. Got {self.min_periods} instead."
                )

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Adds window features.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features + window_features]
            The dataframe with the original plus the new variables. If
            `drop_na=True`, rows with missing data in the new features are removed.
        """
        return super().transform(X)

    def _add_features(self, nw_X: nw.DataFrame) -> nw.DataFrame:
        """
        Adds the window features after the columns of `nw_X`.

        Parameters
        ----------
        nw_X: narwhals dataframe of shape = [n_samples, n_features]
            The data returned by `_check_transform_input_and_state()`.
        """
        if len(self.variables_) == 0:
            return nw_X

        windows = self.window if isinstance(self.window, list) else [self.window]
        functions = (
            self.functions if isinstance(self.functions, list) else [self.functions]
        )
        new_features = self._get_new_features_name()
        X = nw_X.to_native()

        if nwd.is_pandas_dataframe(X) is True:
            # pandas is faster than narwhals, and time spans and freq need its index.
            tmp = []
            for win in windows:
                rolling = X[self.variables_].rolling(
                    window=win, min_periods=self.min_periods
                )
                for function in functions:
                    # the rolling methods are faster than agg() with many variables.
                    rolled = getattr(rolling, function)().shift(
                        periods=self.periods, freq=self.freq
                    )
                    rolled.columns = [
                        f"{var}_window_{win}_{function}" for var in self.variables_
                    ]
                    tmp.append(rolled)
            # with freq, the index of the features moves: merge aligns them to X.
            X = X.merge(
                tmp[0].join(tmp[1:])[new_features],
                left_index=True,
                right_index=True,
                how="left",
            )
            return nw.from_native(X, eager_only=True)

        plx = nw.get_native_namespace(nw_X)
        nw_exprs, native_exprs = [], []
        for name, (win, var, function) in zip(
            new_features, product(windows, self.variables_, functions)
        ):
            # like pandas, windows of integers need all their rows by default.
            min_samples = win if self.min_periods is None else self.min_periods
            if function in _NW_ROLLING:
                rolled = _NW_ROLLING[function](nw.col(var), win, min_samples)
                nw_exprs.append(rolled.shift(self.periods).alias(name))
            else:
                rolled = _POLARS_ROLLING[function](plx.col(var), win, min_samples)
                native_exprs.append(rolled.shift(self.periods).alias(name))

        columns = nw_X.columns
        nw_X = nw_X.with_columns(nw_exprs)
        if len(native_exprs) > 0:
            # the polars features were added last: restore the order of the names.
            X = nw_X.to_native().with_columns(native_exprs)
            nw_X = nw.from_native(X, eager_only=True).select(columns + new_features)

        return nw_X

    def _get_new_features_name(self) -> List:
        """Get names of the lag features."""

        if not isinstance(self.functions, list):
            functions_ = [self.functions]
        else:
            functions_ = self.functions

        if isinstance(self.window, list):
            feature_names = [
                f"{feature}_window_{win}_{agg}"
                for win in self.window
                for feature in self.variables_
                for agg in functions_
            ]
        else:
            feature_names = [
                f"{feature}_window_{self.window}_{agg}"
                for feature in self.variables_
                for agg in functions_
            ]

        return feature_names
