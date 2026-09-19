import re

import numpy as np
import pandas as pd
import polars as pl
import pytest

from feature_engine.timeseries.forecasting import WindowFeatures
from tests.backend_helpers import frame_to_dict, make_series

DATA = {
    "x1": [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0],
    "x2": [10, 20, 30, 40, 50, 60, 70],
    "cat": ["a", "b", "c", "d", "e", "f", "g"],
}

MSG_POLARS_FUNCTIONS = (
    "With dataframes other than pandas, functions takes only ['count', 'kurt', "
    "'max', 'mean', 'median', 'min', 'skew', 'std', 'sum', 'var']. "
)


# init parameters
@pytest.mark.parametrize("window", [[2, 2, 3], ["45min", "45min", "30min"]])
def test_error_if_window_duplicated(window):
    msg = f"There are duplicated windows in the list: {window}"
    with pytest.raises(ValueError, match=re.escape(msg)):
        WindowFeatures(window=window)


@pytest.mark.parametrize("functions", [3.33, [1, "mean"], None, ("mean", "sum")])
def test_error_if_functions_not_string_or_list(functions):
    msg = f"functions must be a string or a list of strings. Got {functions} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        WindowFeatures(functions=functions)


def test_error_if_functions_duplicated():
    msg = "There are duplicated functions in the list: ['sum', 'sum', 'mean']"
    with pytest.raises(ValueError, match=re.escape(msg)):
        WindowFeatures(functions=["sum", "sum", "mean"])


@pytest.mark.parametrize("periods", ["pizza", 3.33, ["mate", "cumbia"], None, 0, -1])
def test_error_if_periods_not_positive_integer(periods):
    msg = f"periods must be a positive integer. Got {periods} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        WindowFeatures(periods=periods)


@pytest.mark.parametrize(
    "window, min_periods, functions, periods, freq, sort_index",
    [
        (3, None, "mean", 1, None, True),
        ([2, 3], 1, ["sum", "mean", "count"], 2, "15min", False),
        (["45min", "30min"], 2, ["std"], 3, "1h", True),
    ],
)
def test_init_param_assignment(
    window, min_periods, functions, periods, freq, sort_index
):
    transformer = WindowFeatures(
        window=window,
        min_periods=min_periods,
        functions=functions,
        periods=periods,
        freq=freq,
        sort_index=sort_index,
        missing_values="ignore",
        drop_original=True,
        drop_na=True,
    )
    assert transformer.window == window
    assert transformer.min_periods == min_periods
    assert transformer.functions == functions
    assert transformer.periods == periods
    assert transformer.freq == freq
    assert transformer.sort_index == sort_index
    assert transformer.missing_values == "ignore"
    assert transformer.drop_original is True
    assert transformer.drop_na is True


# fit and transform
def test_get_feature_names_out(make_df):
    X = make_df(DATA)
    original = ["x1", "x2", "cat"]

    transformer = WindowFeatures(window=3, functions=["mean", "sum"]).fit(X)
    expected = original + [
        "x1_window_3_mean",
        "x1_window_3_sum",
        "x2_window_3_mean",
        "x2_window_3_sum",
    ]
    assert transformer.get_feature_names_out() == expected
    assert transformer.get_feature_names_out(input_features=original) == expected

    transformer = WindowFeatures(window=[2, 3], functions="mean").fit(X)
    assert transformer.get_feature_names_out() == original + [
        "x1_window_2_mean",
        "x2_window_2_mean",
        "x1_window_3_mean",
        "x2_window_3_mean",
    ]

    transformer = WindowFeatures(variables="x2", drop_original=True).fit(X)
    assert transformer.get_feature_names_out() == ["x1", "cat", "x2_window_3_mean"]


@pytest.mark.parametrize("input_features", [["x1"], ["cat"]])
def test_error_if_input_features_differ_from_fit(make_df, input_features):
    transformer = WindowFeatures().fit(make_df(DATA))
    msg = "input_features is not equal to feature_names_in_"
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.get_feature_names_out(input_features=input_features)


def test_error_if_input_features_not_list(make_df):
    transformer = WindowFeatures().fit(make_df(DATA))
    msg = "input_features must be a list or an array. Got x1 instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.get_feature_names_out(input_features="x1")


def test_multiple_windows_and_functions(make_df):
    X = make_df(DATA)
    Xt = WindowFeatures(window=[2, 3], functions=["sum", "mean"]).fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        **DATA,
        "x1_window_2_sum": [None, None, 3.0, 6.0, 11.0, 18.0, 27.0],
        "x1_window_2_mean": [None, None, 1.5, 3.0, 5.5, 9.0, 13.5],
        "x2_window_2_sum": [None, None, 30, 50, 70, 90, 110],
        "x2_window_2_mean": [None, None, 15.0, 25.0, 35.0, 45.0, 55.0],
        "x1_window_3_sum": [None, None, None, 7.0, 13.0, 22.0, 34.0],
        "x1_window_3_mean": pytest.approx(
            [None, None, None, 7 / 3, 13 / 3, 22 / 3, 34 / 3]
        ),
        "x2_window_3_sum": [None, None, None, 60, 90, 120, 150],
        "x2_window_3_mean": [None, None, None, 20.0, 30.0, 40.0, 50.0],
    }


@pytest.mark.parametrize(
    "function, expected",
    [
        ("count", [4.0, 4.0, 4.0]),
        ("sum", [14.0, 24.0, 38.0]),
        ("mean", [3.5, 6.0, 9.5]),
        ("median", [3.0, 5.5, 9.0]),
        ("var", [7.0, 15.333333, 27.0]),
        ("std", [2.645751, 3.915780, 5.196152]),
        ("min", [1.0, 2.0, 4.0]),
        ("max", [7.0, 11.0, 16.0]),
        ("skew", [0.863919, 0.599581, 0.456178]),
        ("kurt", [-0.285714, -0.768431, -0.951989]),
    ],
)
def test_functions(make_df, function, expected):
    X = make_df(DATA)
    Xt = WindowFeatures(variables="x1", window=4, functions=function).fit_transform(X)

    assert isinstance(Xt, make_df)
    # the first window with 4 rows is shifted to the fifth row
    assert frame_to_dict(Xt)[f"x1_window_4_{function}"] == pytest.approx(
        [None, None, None, None] + expected, abs=1e-6
    )


def test_periods(make_df):
    X = make_df(DATA)
    Xt = WindowFeatures(window=2, functions="max", periods=3).fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        **DATA,
        "x1_window_2_max": [None, None, None, None, 2.0, 4.0, 7.0],
        "x2_window_2_max": [None, None, None, None, 20, 30, 40],
    }


def test_min_periods(make_df):
    X = make_df(DATA)
    Xt = WindowFeatures(window=3, min_periods=1).fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        **DATA,
        "x1_window_3_mean": pytest.approx(
            [None, 1.0, 1.5, 7 / 3, 13 / 3, 22 / 3, 34 / 3]
        ),
        "x2_window_3_mean": [None, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0],
    }


def test_min_periods_with_multiple_windows(make_df):
    X = make_df(DATA)
    Xt = WindowFeatures(variables="x2", window=[2, 3], min_periods=2).fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        **DATA,
        "x2_window_2_mean": [None, None, 15.0, 25.0, 35.0, 45.0, 55.0],
        "x2_window_3_mean": [None, None, 15.0, 20.0, 30.0, 40.0, 50.0],
    }


@pytest.mark.parametrize(
    "min_periods, mean, count",
    [
        (
            None,
            [None, None, None, None, None, None, 34 / 3],
            [None, None, None, 2.0, 2.0, 2.0, 3.0],
        ),
        (
            2,
            [None, None, 1.5, 1.5, 4.5, 9.0, 34 / 3],
            [None, None, 2.0, 2.0, 2.0, 2.0, 3.0],
        ),
    ],
)
def test_missing_values_ignored(make_df, min_periods, mean, count):
    X = make_df({**DATA, "x1": [1.0, 2.0, None, 7.0, 11.0, 16.0, 22.0]})
    transformer = WindowFeatures(
        variables="x1",
        functions=["mean", "count"],
        min_periods=min_periods,
        missing_values="ignore",
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    # windows with missing data need min_periods rows with data
    assert frame_to_dict(Xt)["x1_window_3_mean"] == pytest.approx(mean)
    assert frame_to_dict(Xt)["x1_window_3_count"] == count


def test_drop_original(make_df):
    X = make_df(DATA)
    Xt = WindowFeatures(window=2, drop_original=True).fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "cat": DATA["cat"],
        "x1_window_2_mean": [None, None, 1.5, 3.0, 5.5, 9.0, 13.5],
        "x2_window_2_mean": [None, None, 15.0, 25.0, 35.0, 45.0, 55.0],
    }


def test_drop_na(make_df):
    X = make_df(DATA)
    Xt = WindowFeatures(window=[2, 4], functions="min", drop_na=True).fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "x1": [11.0, 16.0, 22.0],
        "x2": [50, 60, 70],
        "cat": ["e", "f", "g"],
        "x1_window_2_min": [4.0, 7.0, 11.0],
        "x2_window_2_min": [30, 40, 50],
        "x1_window_4_min": [1.0, 2.0, 4.0],
        "x2_window_4_min": [10, 20, 30],
    }


def test_transform_x_y(make_df):
    X = make_df(DATA)
    y = make_series(make_df, [1, 2, 3, 4, 5, 6, 7], name="y")
    transformer = WindowFeatures(window=4, drop_na=True).fit(X)

    Xt, yt = transformer.transform_x_y(X, y)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == frame_to_dict(transformer.transform(X))
    assert list(yt) == [5, 6, 7]


def test_return_empty_without_numerical_variables(make_df):
    X = make_df({"cat": DATA["cat"]})
    transformer = WindowFeatures(
        window=[2, 3], functions=["mean", "max"], return_empty=True
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"cat": DATA["cat"]}


# pandas: the index orders the rows in time and allows time spans and freq
@pytest.fixture
def df_time_expected():
    return pd.DataFrame(
        {
            "ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62, 32.5, 32.52, 32.68],
            "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61, 47.01, 46.67, 47.52],
            "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42, 0.49, 0.57, 0.56],
            "color": ["blue"] * 8,
        },
        index=pd.date_range("2020-05-15 12:00:00", periods=8, freq="15min"),
    )


def test_freq_with_pandas(df_time, df_time_expected):
    transformer = WindowFeatures(
        variables=["ambient_temp", "irradiation"],
        window=2,
        functions="sum",
        freq="45min",
    )
    Xt = transformer.fit_transform(df_time).head(8)

    expected = df_time_expected.assign(
        ambient_temp_window_2_sum=[np.nan] * 4 + [62.82, 63.66, 64.54, 65.01],
        irradiation_window_2_sum=[np.nan] * 4 + [1.3, 1.44, 1.41, 1.18],
    )
    pd.testing.assert_frame_equal(Xt, expected, check_freq=False)


def test_time_span_windows_with_pandas(df_time, df_time_expected):
    transformer = WindowFeatures(
        variables="irradiation", window=["30min", "45min"], functions="max"
    )
    Xt = transformer.fit_transform(df_time).head(8)

    # windows of time spans take the rows available
    expected = df_time_expected.assign(
        irradiation_window_30min_max=[np.nan, 0.51, 0.79, 0.79, 0.76, 0.76, 0.49, 0.57],
        irradiation_window_45min_max=[np.nan, 0.51, 0.79, 0.79, 0.79, 0.76, 0.76, 0.57],
    )
    pd.testing.assert_frame_equal(Xt, expected, check_freq=False)


def test_functions_only_in_pandas(df_time):
    transformer = WindowFeatures(variables="irradiation", functions=["sem", "last"])
    Xt = transformer.fit_transform(df_time)

    expected = df_time["irradiation"].rolling(3).agg(["sem", "last"]).shift(1)
    expected.columns = ["irradiation_window_3_sem", "irradiation_window_3_last"]
    pd.testing.assert_frame_equal(Xt[expected.columns], expected)


def test_sort_index_with_pandas(df_time):
    X = df_time.sample(frac=1, random_state=1)

    Xt = WindowFeatures(sort_index=True).fit_transform(X)
    pd.testing.assert_frame_equal(Xt, WindowFeatures().fit_transform(df_time))

    transformer = WindowFeatures(sort_index=False)
    Xt = transformer.fit_transform(X)
    pd.testing.assert_frame_equal(Xt[df_time.columns], X)


def test_transform_x_y_aligns_y_on_the_index_with_pandas(df_time):
    X = df_time.sample(frac=1, random_state=1)
    y = pd.Series(np.arange(len(X)), index=X.index)
    transformer = WindowFeatures(window="45min", drop_na=True).fit(X)

    Xt, yt = transformer.transform_x_y(X, y)

    pd.testing.assert_frame_equal(Xt, transformer.transform(X))
    pd.testing.assert_series_equal(yt, y.loc[Xt.index])


def test_integer_column_names_with_pandas():
    X = pd.DataFrame({0: DATA["x1"], 1: DATA["x2"], "cat": DATA["cat"]})
    transformer = WindowFeatures(window=[2, 3], drop_original=True, drop_na=True)
    Xt = transformer.fit_transform(X)

    expected = pd.DataFrame(
        {
            "cat": ["d", "e", "f", "g"],
            "0_window_2_mean": [3.0, 5.5, 9.0, 13.5],
            "1_window_2_mean": [25.0, 35.0, 45.0, 55.0],
            "0_window_3_mean": [7 / 3, 13 / 3, 22 / 3, 34 / 3],
            "1_window_3_mean": [20.0, 30.0, 40.0, 50.0],
        },
        index=[3, 4, 5, 6],
    )
    # the columns started as a mix of integers and strings
    expected.columns = expected.columns.astype(object)
    pd.testing.assert_frame_equal(Xt, expected)


# polars: the rows are used in the order given and the windows are numbers of rows
def test_rows_keep_their_order_with_polars():
    X = pl.DataFrame({"x1": [4.0, 1.0, 7.0, 2.0]})
    Xt = WindowFeatures(window=2, functions="max").fit_transform(X)

    assert frame_to_dict(Xt)["x1_window_2_max"] == [None, None, 4.0, 7.0]


@pytest.mark.parametrize("window", ["3D", ["45min", "30min"], [2, "30min"]])
def test_error_if_time_span_window_with_polars(window):
    msg = (
        "Time spans in window, like '3D', are only supported with pandas "
        "dataframes, because they use the dataframe's DatetimeIndex. With other "
        "dataframes, window takes integers, the number of rows in each window. "
        f"Got {window} instead."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        WindowFeatures(window=window).fit(pl.DataFrame(DATA))

    # a transformer fitted on pandas can't transform polars with time spans either
    transformer = WindowFeatures(window=window).fit(
        pd.DataFrame(DATA, index=pd.date_range("2020-01-01", periods=7))
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        transformer.transform(pl.DataFrame(DATA))


@pytest.mark.parametrize("functions", ["sem", ["mean", "last"]])
def test_error_if_function_not_supported_with_polars(functions):
    msg = MSG_POLARS_FUNCTIONS + f"Got {functions} instead."
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        WindowFeatures(functions=functions).fit(pl.DataFrame(DATA))

    transformer = WindowFeatures(functions=functions).fit(pd.DataFrame(DATA))
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        transformer.transform(pl.DataFrame(DATA))


def test_error_if_min_periods_is_zero_with_polars():
    msg = (
        "min_periods=0 is only supported with pandas dataframes. With other "
        "dataframes, min_periods takes integers greater than 0 or None. "
        "Got 0 instead."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        WindowFeatures(min_periods=0).fit(pl.DataFrame(DATA))

    transformer = WindowFeatures(min_periods=0).fit(pd.DataFrame(DATA))
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        transformer.transform(pl.DataFrame(DATA))
