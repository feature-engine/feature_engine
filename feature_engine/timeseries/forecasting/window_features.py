from typing import Callable, List, Optional, Union

import narwhals as nw
from narwhals.typing import IntoDataFrame, IntoSeries

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
from feature_engine.dataframe_checks import check_X
from feature_engine.dataframe_helpers import (
    drop_variables_from_df,
    filter_rows,
    rows_with_missing_values,
)
from feature_engine.timeseries.forecasting._forecasting_helpers import (
    add_window_features,
    check_window_parameters,
)
from feature_engine.timeseries.forecasting.base_forecast_transformers import (
    BaseForecastTransformer,
)


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
        The functions to apply to the values of each variable within the window.
        With pandas, the valid functions are the methods of pandas rolling windows
        that aggregate one variable, like 'mean', 'max' or 'std'. See the pandas
        `documentation <https://pandas.pydata.org/docs/reference/window.html>`_ for
        the functions of your pandas version. With polars, the valid functions are
        'count', 'first', 'kurt', 'last', 'max', 'mean', 'median', 'min', 'nunique',
        'rank', 'sem', 'skew', 'std', 'sum' and 'var', which compute the same
        statistics as the pandas functions with the same name. 'rank' needs polars
        1.35 or later.

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

        windows = window if isinstance(window, list) else [window]
        # bool is a subclass of int, but True is not a window size. Windows that
        # are not integers, like time spans, are checked by pandas.
        if len(windows) == 0 or any(
            isinstance(win, int) and (isinstance(win, bool) or win < 1)
            for win in windows
        ):
            raise ValueError(
                "window must be a positive integer, a time span or a non-empty list "
                f"of them. Got {window} instead."
            )
        if len(windows) != len(set(windows)):
            raise ValueError(
                f"There are duplicated windows in the list. Got {window} instead."
            )

        if min_periods is not None and (
            not isinstance(min_periods, int)
            or isinstance(min_periods, bool)
            or min_periods < 0
        ):
            raise ValueError(
                "min_periods must be None or a non-negative integer. "
                f"Got {min_periods} instead."
            )

        if (
            not isinstance(functions, (str, list))
            or len(functions) == 0
            or not all(isinstance(val, str) for val in functions)
        ):
            raise ValueError(
                "functions must be a string or a non-empty list of strings. "
                f"Got {functions} instead."
            )
        if isinstance(functions, list) and len(functions) != len(set(functions)):
            raise ValueError(
                f"There are duplicated functions in the list. Got {functions} instead."
            )

        if not isinstance(periods, int) or isinstance(periods, bool) or periods < 1:
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

    def fit(self, X: IntoDataFrame, y: Optional[IntoSeries] = None):
        """
        This transformer does not learn parameters.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training dataset.

        y: Series, default=None
            y is not needed in this transformer. You can pass None or y.
        """
        super().fit(X, y)
        check_window_parameters(
            check_X(X), self.window, self.functions, self.min_periods
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
        X = self._check_transform_input_and_state(X)

        if len(self.variables_) > 0:
            windows = self.window if isinstance(self.window, list) else [self.window]
            X = add_window_features(
                X,
                self.variables_,
                windows,
                self._functions_list(),
                self.min_periods,
                self.periods,
                self.freq,
                self._get_new_features_name(),
            )

        if self.drop_original is True:
            X = drop_variables_from_df(X, self.variables_)

        if self.drop_na is True:
            new_features = self._get_new_features_name()
            X = filter_rows(X, ~rows_with_missing_values(X, new_features))

        return nw.to_native(X, pass_through=True)

    def _functions_list(self) -> List[str]:
        """Return the functions as a list."""
        if isinstance(self.functions, list):
            return self.functions
        return [self.functions]

    def _get_new_features_name(self) -> List:
        """Get names of the window features."""

        functions_ = self._functions_list()

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
