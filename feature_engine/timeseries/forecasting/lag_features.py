# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from collections.abc import Hashable
from typing import List, Union

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
class LagFeatures(BaseForecastTransformer):
    """
    LagFeatures adds lag features to the dataframe. A lag feature is a feature with
    information about a prior time step.

    LagFeatures works like pandas `shift()`, with the exception that only one of
    `periods` or `freq` can be indicated at a time. LagFeatures builds on top of
    `shift()` in that multiple lags can be created at the same time, and the features
    are added with names to the original dataframe.

    LagFeatures takes pandas and polars dataframes, among others. With pandas, the
    index gives the time order of the rows, so it must have unique values and no NaN.
    polars dataframes have no index: LagFeatures uses the rows in the order given, so
    sort them by time first.

    LagFeatures works only with numerical variables. You can pass a list of variables
    to lag. Alternatively, LagFeatures will automatically select and lag all numerical
    variables found in the training set.

    More details in the :ref:`User Guide <lag_features>`.

    Parameters
    ----------
    {variables}

    {return_empty}

    periods: int, list of ints, default=1
        Number of periods to shift. Can be a positive integer or list of positive
        integers. If list, features will be created for each one of the periods in the
        list. If the parameter `freq` is specified, `periods` will be ignored.

    freq: str, list of str, default=None
        Offset to use from the tseries module or time rule. See parameter `freq` in
        pandas `shift()`. It is the same functionality. If freq is a list, lag features
        will be created for each one of the frequency values in the list. If freq is not
        None, then this parameter overrides the parameter `periods`. `freq` lags the
        values based on the dataframe's DatetimeIndex, so it is only supported with
        pandas dataframes. With polars, use `periods` instead.

    fill_value: object, optional
        The scalar value to use for the missing values introduced by the lags. If None,
        the lag features show missing values (NaN in pandas, null in polars) in the
        rows that have no past values.

    sort_index: bool, default=True
        Whether to order the index of the dataframe before creating the lag features.
        Only applies to pandas dataframes. polars dataframes have no index, so their
        rows are used in the order given.

    {missing_values}

    {drop_original}

    drop_na: bool, default=False.
        Whether the NAN introduced in the lag features should be removed.

    Attributes
    ----------
    variables_:
        The group of variables that will be lagged.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    {fit}

    {fit_transform}

    transform:
        Add lag features.

    transform_x_y:
        Remove rows with missing data from X and y.

    See Also
    --------
    pandas.DataFrame.shift
    polars.Expr.shift

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.timeseries.forecasting import LagFeatures
    >>> X = pd.DataFrame(dict(date = ["2022-09-18",
    >>>                               "2022-09-19",
    >>>                               "2022-09-20",
    >>>                               "2022-09-21",
    >>>                               "2022-09-22"],
    >>>                       x1 = [1,2,3,4,5],
    >>>                       x2 = [6,7,8,9,10]
    >>>                     ))
    >>> lf = LagFeatures(periods=[1,2])
    >>> lf.fit_transform(X)
                date  x1  x2  x1_lag_1  x2_lag_1  x1_lag_2  x2_lag_2
    0  2022-09-18   1   6       NaN       NaN       NaN       NaN
    1  2022-09-19   2   7       1.0       6.0       NaN       NaN
    2  2022-09-20   3   8       2.0       7.0       1.0       6.0
    3  2022-09-21   4   9       3.0       8.0       2.0       7.0
    4  2022-09-22   5  10       4.0       9.0       3.0       8.0

    With polars:

    >>> import polars as pl
    >>> from feature_engine.timeseries.forecasting import LagFeatures
    >>> X = pl.DataFrame(dict(date = ["2022-09-18",
    >>>                               "2022-09-19",
    >>>                               "2022-09-20",
    >>>                               "2022-09-21",
    >>>                               "2022-09-22"],
    >>>                       x1 = [1,2,3,4,5],
    >>>                       x2 = [6,7,8,9,10]
    >>>                     ))
    >>> lf = LagFeatures(periods=[1,2])
    >>> lf.fit_transform(X)
    shape: (5, 7)
    ┌────────────┬─────┬─────┬──────────┬──────────┬──────────┬──────────┐
    │ date       ┆ x1  ┆ x2  ┆ x1_lag_1 ┆ x2_lag_1 ┆ x1_lag_2 ┆ x2_lag_2 │
    │ ---        ┆ --- ┆ --- ┆ ---      ┆ ---      ┆ ---      ┆ ---      │
    │ str        ┆ i64 ┆ i64 ┆ i64      ┆ i64      ┆ i64      ┆ i64      │
    ╞════════════╪═════╪═════╪══════════╪══════════╪══════════╪══════════╡
    │ 2022-09-18 ┆ 1   ┆ 6   ┆ null     ┆ null     ┆ null     ┆ null     │
    │ 2022-09-19 ┆ 2   ┆ 7   ┆ 1        ┆ 6        ┆ null     ┆ null     │
    │ 2022-09-20 ┆ 3   ┆ 8   ┆ 2        ┆ 7        ┆ 1        ┆ 6        │
    │ 2022-09-21 ┆ 4   ┆ 9   ┆ 3        ┆ 8        ┆ 2        ┆ 7        │
    │ 2022-09-22 ┆ 5   ┆ 10  ┆ 4        ┆ 9        ┆ 3        ┆ 8        │
    └────────────┴─────┴─────┴──────────┴──────────┴──────────┴──────────┘
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        periods: Union[int, List[int]] = 1,
        freq: Union[str, List[str], None] = None,
        fill_value: Hashable = None,
        sort_index: bool = True,
        missing_values: str = "raise",
        drop_original: bool = False,
        drop_na: bool = False,
    ) -> None:

        if not (
            isinstance(periods, int)
            and periods > 0
            or isinstance(periods, list)
            and all(isinstance(num, int) and num > 0 for num in periods)
        ):

            raise ValueError(
                "periods must be an integer or a list of positive integers. "
                f"Got {periods} instead."
            )
        if isinstance(periods, list) and len(periods) != len(set(periods)):
            raise ValueError(f"There are duplicated periods in the list: {periods}")

        if isinstance(freq, list) and len(freq) != len(set(freq)):
            raise ValueError(f"There are duplicated freq values in the list: {freq}")

        if not isinstance(sort_index, bool):
            raise ValueError(
                f"sort_index takes values True and False. Got {sort_index} instead."
            )

        super().__init__(
            variables, return_empty, missing_values, drop_original, drop_na
        )

        self.periods = periods
        self.freq = freq
        self.fill_value = fill_value
        self.sort_index = sort_index

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Adds lag features.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features + lag_features]
            The dataframe with the original plus the new variables. If
            `drop_na=True`, rows with missing data in the lag features are removed.
        """
        return super().transform(X)

    def _add_features(self, nw_X: nw.DataFrame) -> nw.DataFrame:
        """Adds the lag features after the columns of nw_X."""
        X = nw_X.to_native()
        new_features = self._get_new_features_name()

        periods = self.periods if isinstance(self.periods, list) else [self.periods]

        if nwd.is_pandas_dataframe(X) is True:
            # pandas is faster than narwhals, and freq needs the pandas index. The
            # suffixes are the ones merge() added when a lag name was already in X.
            if self.freq is None:
                lag = X[self.variables_].shift(
                    periods=periods, fill_value=self.fill_value
                )
                lag.columns = new_features
                return nw.from_native(
                    X.join(lag, lsuffix="_x", rsuffix="_y"), eager_only=True
                )

            freqs = self.freq if isinstance(self.freq, list) else [self.freq]
            # joining one lag at a time is faster than aligning all lags at once.
            for i, fr in enumerate(freqs):
                start, end = i * len(self.variables_), (i + 1) * len(self.variables_)
                lag = X[self.variables_].shift(freq=fr)
                lag.columns = new_features[start:end]
                X = X.join(lag, lsuffix="_x", rsuffix="_y")

            # shift() does not take fill_value together with freq.
            if self.fill_value is not None:
                X[new_features] = X[new_features].fillna(value=self.fill_value)

            return nw.from_native(X, eager_only=True)

        lags = [(var, pr) for pr in periods for var in self.variables_]
        if self.fill_value is None:
            return nw_X.with_columns(
                nw.col(var).shift(pr).alias(name)
                for (var, pr), name in zip(lags, new_features)
            )

        if nwd.is_polars_dataframe(X) is True:
            # polars is faster than narwhals: it fills the rows that shift()
            # introduces without copying the column.
            col = nw.get_native_namespace(nw_X).col
            return nw.from_native(
                X.with_columns(
                    col(var).shift(pr, fill_value=self.fill_value).alias(name)
                    for (var, pr), name in zip(lags, new_features)
                ),
                eager_only=True,
            )

        # narwhals' shift() has no fill_value. The mask is True in the first rows,
        # which have no past values, so missing data in X is not filled.
        return nw_X.with_columns(
            nw.when(nw.col(var).is_null().shift(pr).is_null())
            .then(nw.lit(self.fill_value))
            .otherwise(nw.col(var).shift(pr))
            .alias(name)
            for (var, pr), name in zip(lags, new_features)
        )

    def _get_new_features_name(self) -> List:
        """Get names of the lag features."""

        # create the names for the lag features
        if isinstance(self.freq, list):
            feature_names = [
                f"{feature}_lag_{fr}" for fr in self.freq for feature in self.variables_
            ]
        elif self.freq is not None:
            feature_names = [
                f"{feature}_lag_{self.freq}" for feature in self.variables_
            ]
        elif isinstance(self.periods, list):
            feature_names = [
                f"{feature}_lag_{pr}"
                for pr in self.periods
                for feature in self.variables_
            ]
        else:
            feature_names = [
                f"{feature}_lag_{self.periods}" for feature in self.variables_
            ]

        return feature_names
