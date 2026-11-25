# Authors: Morgan Sell <morganpsell@gmail.com>
# License: BSD 3 clause

from collections.abc import Hashable
from typing import List, Union

import narwhals as nw
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
from feature_engine.dataframe_helpers import (
    drop_variables_from_df,
    filter_rows,
    rows_with_missing_values,
)
from feature_engine.timeseries.forecasting._forecasting_helpers import (
    add_lag_features,
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

        periods_list = periods if isinstance(periods, list) else [periods]
        # bool is a subclass of int, but True is not a number of periods.
        if len(periods_list) == 0 or not all(
            isinstance(pr, int) and not isinstance(pr, bool) and pr > 0
            for pr in periods_list
        ):
            raise ValueError(
                "periods must be a positive integer or a non-empty list of positive "
                f"integers. Got {periods} instead."
            )
        if len(periods_list) != len(set(periods_list)):
            raise ValueError(
                f"There are duplicated periods in the list. Got {periods} instead."
            )

        if isinstance(freq, list) and len(freq) == 0:
            raise ValueError(
                "freq must be None, a frequency or a non-empty list of frequencies. "
                f"Got {freq} instead."
            )
        if isinstance(freq, list) and len(freq) != len(set(freq)):
            raise ValueError(
                f"There are duplicated freq values in the list. Got {freq} instead."
            )

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
        X = self._check_transform_input_and_state(X)

        periods = self.periods if isinstance(self.periods, list) else [self.periods]
        if self.freq is None or isinstance(self.freq, list):
            freq = self.freq
        else:
            freq = [self.freq]
        new_features = self._get_new_features_name()
        X = add_lag_features(
            X, self.variables_, periods, freq, self.fill_value, new_features
        )

        if self.drop_original is True:
            X = drop_variables_from_df(X, self.variables_)

        if self.drop_na is True:
            X = filter_rows(X, ~rows_with_missing_values(X, new_features))

        return nw.to_native(X, pass_through=True)

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
