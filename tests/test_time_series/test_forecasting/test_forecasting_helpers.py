import re

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from feature_engine.dataframe_checks import check_X
from feature_engine.timeseries.forecasting._forecasting_helpers import (
    add_expanding_window_features,
    add_lag_features,
    add_window_features,
    check_functions_supported,
    check_index,
    check_window_parameters,
    sort_by_index,
)
from tests.backend_helpers import frame_to_dict

DATES = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])
MSG_INDEX_NA = (
    "The dataframe's index contains NaN values or missing data. Only dataframes "
    "with complete indexes are compatible with this transformer."
)
MSG_INDEX_DUPLICATED = (
    "The dataframe's index does not contain unique values. Only dataframes with "
    "unique values in the index are compatible with this transformer."
)


@pytest.mark.parametrize("freq", [None, "1D"])
def test_check_index_accepts_complete_and_unique_pandas_index(freq):
    check_index(check_X(pd.DataFrame({"a": [1, 2, 3]}, index=DATES)), freq)


@pytest.mark.parametrize(
    "index, msg",
    [
        (list(DATES[:2]) + [pd.NaT], MSG_INDEX_NA),
        (list(DATES[:2]) + [DATES[0]], MSG_INDEX_DUPLICATED),
    ],
)
def test_check_index_raises_error_if_pandas_index_incomplete_or_duplicated(
    index, msg
):
    X = check_X(pd.DataFrame({"a": [1, 2, 3]}, index=index))
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        check_index(X)


def test_check_index_raises_error_if_freq_with_polars():
    X = check_X(pl.DataFrame({"a": [1, 2, 3]}))
    check_index(X)

    msg = (
        "freq is only supported with pandas dataframes, because it uses the "
        "dataframe's DatetimeIndex. With other dataframes, leave freq=None "
        "to shift the rows by periods. Got 1D instead."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        check_index(X, "1D")


def test_sort_by_index_sorts_pandas_rows_by_the_index():
    X = pd.DataFrame({"a": [3, 1, 2]}, index=DATES)
    Xt = sort_by_index(check_X(X))

    expected = pd.DataFrame({"a": [1, 2, 3]}, index=DATES.sort_values())
    pd.testing.assert_frame_equal(Xt, expected)
    assert X["a"].tolist() == [3, 1, 2]


def test_sort_by_index_keeps_the_row_order_with_polars():
    X = check_X(pl.DataFrame({"a": [3, 1, 2]}))
    assert frame_to_dict(sort_by_index(X).to_native()) == {"a": [3, 1, 2]}


def test_add_lag_features(make_df):
    X = check_X(make_df({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}))
    Xt = add_lag_features(X, ["a", "b"], [1, 2], None, None, ["a1", "b1", "a2", "b2"])

    assert frame_to_dict(nw.to_native(Xt, pass_through=True)) == {
        "a": [1.0, 2.0, 3.0],
        "b": [4.0, 5.0, 6.0],
        "a1": [None, 1.0, 2.0],
        "b1": [None, 4.0, 5.0],
        "a2": [None, None, 1.0],
        "b2": [None, None, 4.0],
    }


@pytest.mark.parametrize("backend", ["pandas", "polars", "narwhals"])
def test_add_lag_features_with_fill_value(backend):
    # a pandas dataframe in narwhals takes the path of the libraries other than
    # pandas and polars
    data = {"a": [1.0, None, 3.0]}
    X = {
        "pandas": check_X(pd.DataFrame(data)),
        "polars": check_X(pl.DataFrame(data)),
        "narwhals": nw.from_native(pd.DataFrame(data), eager_only=True),
    }[backend]
    Xt = add_lag_features(X, ["a"], [1], None, 0, ["a_lag"])

    # the missing value of a is lagged, not filled
    assert frame_to_dict(nw.to_native(Xt, pass_through=True))["a_lag"] == [
        0,
        1.0,
        None,
    ]


def test_add_window_features(make_df):
    X = check_X(make_df({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]}))
    names = ["a_max", "a_mean", "a_first", "b_max", "b_mean", "b_first"]
    functions = ["max", "mean", "first"]
    Xt = add_window_features(X, ["a", "b"], [2], functions, None, 1, None, names)

    # with polars, the functions come from different libraries, in the order of names
    assert list(Xt.columns) == ["a", "b"] + names
    assert frame_to_dict(nw.to_native(Xt, pass_through=True)) == {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [4.0, 3.0, 2.0, 1.0],
        "a_max": [None, None, 2.0, 3.0],
        "a_mean": [None, None, 1.5, 2.5],
        "a_first": [None, None, 1.0, 2.0],
        "b_max": [None, None, 4.0, 3.0],
        "b_mean": [None, None, 3.5, 2.5],
        "b_first": [None, None, 4.0, 3.0],
    }


@pytest.mark.parametrize(
    "window, functions, min_periods",
    [("3D", "mean", None), (3, ["quantile"], None), (3, "mean", 0)],
)
def test_check_window_parameters_accepts_everything_with_pandas(
    window, functions, min_periods
):
    X = check_X(pd.DataFrame({"a": [1, 2, 3]}, index=DATES))
    check_window_parameters(X, window, functions, min_periods)


def test_add_expanding_window_features(make_df):
    X = check_X(make_df({"a": [1.0, 3.0, 2.0, 4.0]}))
    names = ["a_max", "a_mean", "a_sum"]
    Xt = add_expanding_window_features(
        X, ["a"], ["max", "mean", "sum"], None, 1, None, False, names
    )

    assert frame_to_dict(nw.to_native(Xt, pass_through=True)) == {
        "a": [1.0, 3.0, 2.0, 4.0],
        "a_max": [None, 1.0, 3.0, 3.0],
        "a_mean": [None, 1.0, 2.0, 2.0],
        "a_sum": [None, 1.0, 4.0, 6.0],
    }


def test_check_functions_supported():
    check_functions_supported(check_X(pd.DataFrame({"a": [1]})), ["sem"], ["mean"])

    msg = (
        "With dataframes other than pandas, functions takes only ['max', 'mean']. "
        "Got sem instead."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        check_functions_supported(
            check_X(pl.DataFrame({"a": [1]})), "sem", ["mean", "max"]
        )
