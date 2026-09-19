# Author: Kishan Manani
# License: BSD 3 clause

from __future__ import annotations

from typing import List

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

# functions supported with dataframes other than pandas.
_FUNCTIONS = ["count", "sum", "mean", "median", "min", "max", "std", "var"]


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
class ExpandingWindowFeatures(BaseForecastTransformer):
    """
    ExpandingWindowFeatures adds new features to a dataframe based on expanding window
    operations. Expanding window operations are operations that perform an
    aggregation over an expanding window of all past values relative to the
    value of interest. An expanding window feature is, in other words, a feature
    created after computing statistics (e.g., mean, min, max, etc.) using a
    window over all the past data. For example, the mean value of all months
    prior to the month of interest is an expanding window feature.

    ExpandingWindowFeatures works with pandas and polars dataframes, and returns
    the same type of dataframe it receives.

    With pandas, ExpandingWindowFeatures uses the pandas' functions `expanding()`,
    `agg()` and `shift()`. With `expanding()`, it creates expanding windows. With
    `agg()` it applies multiple functions within those windows. With 'shift()' it
    allocates the values to the correct rows. For supported aggregation functions,
    see Expanding Window
    `Functions
    <https://pandas.pydata.org/docs/reference/window.html#expanding-window-functions>`_.
    The dataframe's index orders the rows in time, so it must have unique values
    and no NaN.

    With polars, ExpandingWindowFeatures supports the functions 'count', 'sum',
    'mean', 'median', 'min', 'max', 'std' and 'var', and returns the same values
    as pandas. polars dataframes have no index, so the rows must be sorted in time
    before using the transformer, and `freq` and `sort_index` are not used.

    ExpandingWindowFeatures works only with numerical variables. You can pass a
    list of variables to use as input for the expanding window. Alternatively,
    ExpandingWindowFeatures will automatically select all numerical variables
    in the training set.

    More details in the :ref:`User Guide <expanding_window_features>`.

    Parameters
    ----------
    {variables}

    {return_empty}

    min_periods: int, default None.
        Minimum number of observations in window required to have a value;
        otherwise, result is missing. See parameter `min_periods` in the pandas
        `expanding()` documentation for more details.

    functions: str, list of str, default = 'mean'
        The functions to apply within the window. With pandas, valid functions can be
        found `here <https://pandas.pydata.org/docs/reference/window.html>`_. With
        polars, the functions can be 'count', 'sum', 'mean', 'median', 'min', 'max',
        'std' and 'var'.

    periods: int, default=1
        Number of periods to shift. Can be zero or a positive integer. See param
        `periods` in pandas `shift`.

    freq: str, default=None
        Offset to use from the tseries module or time rule. See parameter `freq` in
        pandas `shift()`. Only supported with pandas dataframes.

    sort_index: bool, default=True
        Whether to order the index of the dataframe before creating the
        expanding window feature. Only used with pandas dataframes.

    {missing_values}

    {drop_original}

    drop_na: bool, default=False.
        Whether the NAN introduced in the created features should be removed.


    Attributes
    ----------
    variables_:
        The group of variables that will be used to create the expanding window
        features.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    transform:
        Add expanding window features.

    transform_x_y:
        Remove rows with missing data from X and y.

    {fit_transform}

    See Also
    --------
    pandas.expanding
    pandas.aggregate
    pandas.shift

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.timeseries.forecasting import ExpandingWindowFeatures
    >>> X = pd.DataFrame(dict(date = ["2022-09-18",
    >>>                               "2022-09-19",
    >>>                               "2022-09-20",
    >>>                               "2022-09-21",
    >>>                               "2022-09-22"],
    >>>                       x1 = [1,2,3,4,5],
    >>>                       x2 = [6,7,8,9,10]
    >>>                     ))
    >>> ewf = ExpandingWindowFeatures()
    >>> ewf.fit_transform(X)
             date  x1  x2  x1_expanding_mean  x2_expanding_mean
    0  2022-09-18   1   6                NaN                NaN
    1  2022-09-19   2   7                1.0                6.0
    2  2022-09-20   3   8                1.5                6.5
    3  2022-09-21   4   9                2.0                7.0
    4  2022-09-22   5  10                2.5                7.5

    With polars:

    >>> import polars as pl
    >>> from feature_engine.timeseries.forecasting import ExpandingWindowFeatures
    >>> X = pl.DataFrame(dict(date = ["2022-09-18",
    >>>                               "2022-09-19",
    >>>                               "2022-09-20",
    >>>                               "2022-09-21",
    >>>                               "2022-09-22"],
    >>>                       x1 = [1,2,3,4,5],
    >>>                       x2 = [6,7,8,9,10]
    >>>                     ))
    >>> ewf = ExpandingWindowFeatures()
    >>> ewf.fit_transform(X)
    shape: (5, 5)
    ┌────────────┬─────┬─────┬───────────────────┬───────────────────┐
    │ date       ┆ x1  ┆ x2  ┆ x1_expanding_mean ┆ x2_expanding_mean │
    │ ---        ┆ --- ┆ --- ┆ ---               ┆ ---               │
    │ str        ┆ i64 ┆ i64 ┆ f64               ┆ f64               │
    ╞════════════╪═════╪═════╪═══════════════════╪═══════════════════╡
    │ 2022-09-18 ┆ 1   ┆ 6   ┆ null              ┆ null              │
    │ 2022-09-19 ┆ 2   ┆ 7   ┆ 1.0               ┆ 6.0               │
    │ 2022-09-20 ┆ 3   ┆ 8   ┆ 1.5               ┆ 6.5               │
    │ 2022-09-21 ┆ 4   ┆ 9   ┆ 2.0               ┆ 7.0               │
    │ 2022-09-22 ┆ 5   ┆ 10  ┆ 2.5               ┆ 7.5               │
    └────────────┴─────┴─────┴───────────────────┴───────────────────┘
    """

    def __init__(
        self,
        variables: None | int | str | list[str | int] = None,
        return_empty: bool = False,
        min_periods: int | None = None,
        functions: str | list[str] = "mean",
        periods: int = 1,
        freq: str | None = None,
        sort_index: bool = True,
        missing_values: str = "raise",
        drop_original: bool = False,
        drop_na: bool = False,
    ) -> None:

        if (
            not isinstance(functions, (str, list))
            or len(functions) == 0
            or not all(isinstance(val, str) for val in functions)
        ):
            raise ValueError(
                "functions must be a list of strings or a string. "
                f"Got {functions} instead."
            )
        if isinstance(functions, list) and len(functions) != len(set(functions)):
            raise ValueError(f"There are duplicated functions in the list: {functions}")

        if not isinstance(periods, int) or periods < 0:
            raise ValueError(
                f"periods must be a non-negative integer. Got {periods} instead."
            )

        super().__init__(
            variables, return_empty, missing_values, drop_original, drop_na
        )

        self.min_periods = min_periods
        self.functions = functions
        self.periods = periods
        self.freq = freq
        self.sort_index = sort_index

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Adds expanding window features.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features + window_features]
            The dataframe with the original plus the new variables.
        """
        return super().transform(X)

    def _add_features(self, nw_X: nw.DataFrame) -> nw.DataFrame:
        """Add the expanding window features after the columns of nw_X."""
        X = nw_X.to_native()
        if nwd.is_pandas_dataframe(X) is True:
            # pandas is faster than narwhals, and freq shifts the pandas index.
            tmp = (
                X[self.variables_]
                .expanding(min_periods=self.min_periods)
                .agg(self.functions)
                .shift(periods=self.periods, freq=self.freq)
            )
            tmp.columns = self._get_new_features_name()
            X = X.merge(tmp, left_index=True, right_index=True, how="left")
            return nw.from_native(X, eager_only=True)

        functions = self._functions_list()
        unsupported = [func for func in functions if func not in _FUNCTIONS]
        if len(unsupported) > 0:
            raise NotImplementedError(
                "With dataframes other than pandas, ExpandingWindowFeatures supports "
                f"the functions {_FUNCTIONS}. Got {unsupported} instead."
            )

        # pandas treats min_periods=None as 0. narwhals' rolling functions need at
        # least 1, and 0 only changes the sum, done below.
        min_periods = 0 if self.min_periods is None else self.min_periods
        min_samples = max(min_periods, 1)
        # a rolling window as long as the dataframe is an expanding window.
        window = max(len(nw_X), min_samples)

        new_features = []
        for var in self.variables_:
            # pandas computes in float and treats NaN as missing data, and inf too,
            # except when counting.
            x = nw.col(var).cast(nw.Float64)
            if self.missing_values == "ignore":
                n_values = x.fill_nan(None).cum_count()
                x = nw.when(x.is_finite()).then(x)
            else:
                n_values = x.cum_count()
            count = x.cum_count()
            for func in functions:
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
                    values = nw_X.select(x).get_column(var).to_native()
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
                new_features.append(
                    feature.shift(self.periods).alias(f"{var}_expanding_{func}")
                )

        return nw_X.with_columns(new_features)

    def _functions_list(self) -> List[str]:
        """Return the functions as a list."""
        if isinstance(self.functions, list):
            return self.functions
        return [self.functions]

    def _get_new_features_name(self) -> List:
        """Get names of the window features."""

        feature_names = [
            f"{feature}_expanding_{agg}"
            for feature in self.variables_
            for agg in self._functions_list()
        ]

        return feature_names
