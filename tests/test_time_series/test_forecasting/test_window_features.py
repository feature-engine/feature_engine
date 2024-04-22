import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from feature_engine.timeseries.forecasting import WindowFeatures

_date_time = [
    pd.Timestamp("2020-05-15 12:00:00"),
    pd.Timestamp("2020-05-15 12:15:00"),
    pd.Timestamp("2020-05-15 12:30:00"),
    pd.Timestamp("2020-05-15 12:45:00"),
    pd.Timestamp("2020-05-15 13:00:00"),
    pd.Timestamp("2020-05-15 13:15:00"),
    pd.Timestamp("2020-05-15 13:30:00"),
    pd.Timestamp("2020-05-15 13:45:00"),
    pd.Timestamp("2020-05-15 14:00:00"),
]


@pytest.mark.parametrize("_window", [[2, 2, 3], ["45min", "45min", "30min"]])
def test_error_when_duplicated_windows(_window):
    with pytest.raises(ValueError):
        WindowFeatures(window=_window)


@pytest.mark.parametrize("_periods", [1, 2, 3])
def test_permitted_param_periods(_periods):
    transformer = WindowFeatures(periods=_periods)
    assert transformer.periods == _periods


@pytest.mark.parametrize("_periods", ["pizza", 3.33, ["mate", "cumbia"], None])
def test_error_when_non_permitted_param_periods(_periods):
    with pytest.raises(ValueError):
        WindowFeatures(periods=_periods)


@pytest.mark.parametrize(
    "_functions", ["mean", "std", ["sum", "mean"], ["sum", "mean", "count"]]
)
def test_permitted_param_functions(_functions):
    transformer = WindowFeatures(functions=_functions)
    assert transformer.functions == _functions


@pytest.mark.parametrize(
    "_functions", [3.33, [1, "mean"], None, ["sum", "sum", "mean"]]
)
def test_error_when_non_permitted_param_functions(_functions):
    with pytest.raises(ValueError):
        WindowFeatures(functions=_functions)


@pytest.mark.parametrize("_drop_or", [-1, [0], None, 7, "hola"])
def test_error_when_non_permitted_param_drop_original(_drop_or):
    with pytest.raises(ValueError):
        WindowFeatures(drop_original=_drop_or)


@pytest.mark.parametrize("_drop_na", [-1, [0], None, 7, "hola"])
def test_error_when_non_permitted_param_drop_na(_drop_na):
    with pytest.raises(ValueError):
        WindowFeatures(drop_na=_drop_na)


def test_get_feature_names_out(df_time):
    # input features
    original_features = ["ambient_temp", "module_temp", "irradiation", "color"]

    # case 1: 1 window, multiple variables, multiple functions
    tr = WindowFeatures(window=3, functions=["mean", "sum"])
    tr.fit(df_time)

    # expected
    output = [
        "ambient_temp_window_3_mean",
        "ambient_temp_window_3_sum",
        "module_temp_window_3_mean",
        "module_temp_window_3_sum",
        "irradiation_window_3_mean",
        "irradiation_window_3_sum",
    ]
    output = original_features + output
    assert tr.get_feature_names_out(input_features=None) == output
    assert tr.get_feature_names_out(input_features=original_features) == output

    with pytest.raises(ValueError):
        # get error when a user passes a string instead of list
        tr.get_feature_names_out(input_features=["ambient_temp"])

    with pytest.raises(ValueError):
        # assert error when uses passes features that were not transformed
        tr.get_feature_names_out(input_features=["color"])

    # case 2: 1 window, 1 variable, multiple functions
    tr = WindowFeatures(
        variables="ambient_temp", window=2, functions=["sum", "mean", "count"]
    )
    tr.fit(df_time)

    # expected
    output = [
        "ambient_temp_window_2_sum",
        "ambient_temp_window_2_mean",
        "ambient_temp_window_2_count",
    ]
    output = original_features + output
    assert tr.get_feature_names_out(input_features=None) == output
    assert tr.get_feature_names_out(input_features=df_time.columns) == output

    # case 3: multiple windows, multiple variables, multiple functions
    tr = WindowFeatures(window=[2, 3], functions=["mean", "sum"])
    tr.fit(df_time)

    # expected
    output = [
        "ambient_temp_window_2_mean",
        "ambient_temp_window_2_sum",
        "module_temp_window_2_mean",
        "module_temp_window_2_sum",
        "irradiation_window_2_mean",
        "irradiation_window_2_sum",
        "ambient_temp_window_3_mean",
        "ambient_temp_window_3_sum",
        "module_temp_window_3_mean",
        "module_temp_window_3_sum",
        "irradiation_window_3_mean",
        "irradiation_window_3_sum",
    ]
    output = original_features + output
    assert tr.get_feature_names_out(input_features=None) == output
    assert tr.get_feature_names_out(input_features=original_features) == output

    # case 4: 1 window, multiple variables, 1 function
    tr = WindowFeatures(window=2, functions=["mean"])
    tr.fit(df_time)

    # expected
    output = [
        "ambient_temp_window_2_mean",
        "module_temp_window_2_mean",
        "irradiation_window_2_mean",
    ]
    output = original_features + output
    assert tr.get_feature_names_out(input_features=None) == output

    # case 5: When function is a string
    tr = WindowFeatures(window=2, functions="mean")
    tr.fit(df_time)

    # expected
    output = [
        "ambient_temp_window_2_mean",
        "module_temp_window_2_mean",
        "irradiation_window_2_mean",
    ]
    output = original_features + output
    assert tr.get_feature_names_out(input_features=None) == output
    assert tr.get_feature_names_out(input_features=original_features) == output

    # case 6: drop_original is True
    tr = WindowFeatures(window=2, functions="mean", drop_original=True)
    tr.fit(df_time)

    # expected
    output = [
        "ambient_temp_window_2_mean",
        "module_temp_window_2_mean",
        "irradiation_window_2_mean",
    ]
    assert tr.get_feature_names_out(input_features=None) == ["color"] + output
    assert (
        tr.get_feature_names_out(input_features=original_features) == ["color"] + output
    )


def test_single_window_when_using_periods(df_time):

    expected_results = {
        "ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62, 32.5, 32.52, 32.68, 33.76],
        "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61, 47.01, 46.67, 47.52, 49.8],
        "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42, 0.49, 0.57, 0.56, 0.74],
        "color": [
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
        ],
        "ambient_temp_window_3_median": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            31.51,
            32.15,
            32.39,
            32.5,
            32.52,
        ],
        "module_temp_window_3_median": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            49.84,
            50.63,
            50.63,
            49.61,
            47.01,
        ],
        "irradiation_window_3_median": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.65,
            0.76,
            0.65,
            0.49,
            0.49,
        ],
    }
    expected_results_df = pd.DataFrame(data=expected_results, index=_date_time)

    # Case 1: automatically select variables
    transformer = WindowFeatures(window=3, functions=["median"], periods=2)
    df_tr = transformer.fit_transform(df_time)

    assert df_tr.head(9).equals(expected_results_df)

    # Case 2: when drop_original is true
    transformer = WindowFeatures(
        window=3, functions=["median"], periods=2, drop_original=True
    )
    df_tr = transformer.fit_transform(df_time)

    assert df_tr.head(9).equals(
        expected_results_df.drop(["ambient_temp", "module_temp", "irradiation"], axis=1)
    )

    # Case 3: user indicates multiple variables
    transformer = WindowFeatures(
        variables=["module_temp", "irradiation"],
        window=3,
        functions="median",
        periods=2,
    )
    transformer.fit(df_time)
    df_tr = transformer.transform(df_time)

    assert df_tr.head(9).equals(
        expected_results_df.drop(["ambient_temp_window_3_median"], axis=1)
    )

    # Case 4: user indicates 1 variable
    transformer = WindowFeatures(
        variables=["module_temp"], window=3, functions=["median"], periods=2
    )
    transformer.fit(df_time)
    df_tr = transformer.transform(df_time)

    assert df_tr.head(9).equals(
        expected_results_df.drop(
            ["ambient_temp_window_3_median", "irradiation_window_3_median"], axis=1
        )
    )


def test_single_window_when_using_freq(df_time):

    expected_results = {
        "ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62, 32.5, 32.52, 32.68, 33.76],
        "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61, 47.01, 46.67, 47.52, 49.8],
        "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42, 0.49, 0.57, 0.56, 0.74],
        "color": [
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
        ],
        "ambient_temp_window_2_sum": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            62.82,
            63.66,
            64.54,
            65.01,
            65.12,
        ],
        "module_temp_window_2_sum": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            99.02,
            102.19,
            102.98,
            100.24,
            96.62,
        ],
        "irradiation_window_2_sum": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            1.3,
            1.44,
            1.41,
            1.18,
            0.91,
        ],
    }

    expected_results_df = pd.DataFrame(data=expected_results, index=_date_time)

    # Case 1: automatically select variables
    transformer = WindowFeatures(window=2, functions=["sum"], freq="45min")
    df_tr = transformer.fit_transform(df_time)

    assert df_tr.head(9).round(3).equals(expected_results_df)

    # Case 2: when drop_original is true
    transformer = WindowFeatures(
        window=2, functions=["sum"], freq="45min", drop_original=True
    )
    df_tr = transformer.fit_transform(df_time)

    assert (
        df_tr.head(9)
        .round(3)
        .equals(
            expected_results_df.drop(
                ["ambient_temp", "module_temp", "irradiation"], axis=1
            )
        )
    )

    # Case 3: user indicates multiple variables
    transformer = WindowFeatures(
        variables=["ambient_temp", "irradiation"],
        window=2,
        functions="sum",
        freq="45min",
    )
    df_tr = transformer.fit_transform(df_time)

    assert (
        df_tr.head(9)
        .round(3)
        .equals(expected_results_df.drop(["module_temp_window_2_sum"], axis=1))
    )

    # Case 4: user indicates 1 variable
    transformer = WindowFeatures(
        variables=["irradiation"], window=2, functions=["sum"], freq="45min"
    )
    df_tr = transformer.fit_transform(df_time)

    assert (
        df_tr.head(9)
        .round(3)
        .equals(
            expected_results_df.drop(
                ["ambient_temp_window_2_sum", "module_temp_window_2_sum"], axis=1
            )
        )
    )


def test_multiple_windows(df_time):

    # Case 1: automatically select variables
    transformer = WindowFeatures(
        window=[2, 3], functions=["sum", "mean"], periods=15, freq="min"
    )
    df_time_tr = transformer.fit_transform(df_time)

    # Expected
    X = df_time.copy()
    num_vars = ["ambient_temp", "module_temp", "irradiation"]
    tmp = X[num_vars].rolling(2).agg(["sum", "mean"]).shift(periods=15, freq="min")
    tmp.columns = tmp.columns.droplevel()
    X_tr = X.merge(tmp, left_index=True, right_index=True, how="left")
    tmp = X[num_vars].rolling(3).agg(["sum", "mean"]).shift(periods=15, freq="min")
    tmp.columns = tmp.columns.droplevel()
    X_tr = X_tr.merge(tmp, left_index=True, right_index=True, how="left")
    X_tr.columns = transformer.get_feature_names_out()

    assert df_time_tr.equals(X_tr)

    # Case 2: user indicates multiple variables
    transformer = WindowFeatures(
        variables=["ambient_temp", "irradiation"],
        window=[2, 3],
        functions=["sum", "mean"],
        freq="30min",
    )
    df_time_tr = transformer.fit_transform(df_time)

    # Expected
    X = df_time.copy()
    tmp = (
        X[["ambient_temp", "irradiation"]]
        .rolling(2)
        .agg(["sum", "mean"])
        .shift(freq="30min")
    )
    tmp.columns = tmp.columns.droplevel()
    X_tr = X.merge(tmp, left_index=True, right_index=True, how="left")
    tmp = (
        X[["ambient_temp", "irradiation"]]
        .rolling(3)
        .agg(["sum", "mean"])
        .shift(freq="30min")
    )
    tmp.columns = tmp.columns.droplevel()
    X_tr = X_tr.merge(tmp, left_index=True, right_index=True, how="left")
    X_tr.columns = transformer.get_feature_names_out()

    assert df_time_tr.equals(X_tr)

    # Case 4: user indicates 1 variable
    transformer = WindowFeatures(
        variables=["irradiation"],
        window=[2, 3],
        functions="std",
        periods=1,
    )
    df_time_tr = transformer.fit_transform(df_time)

    # Expected
    X = df_time.copy()
    tmp = X["irradiation"].rolling(2).agg("std").shift(periods=1)
    X_tr = X.merge(tmp, left_index=True, right_index=True, how="left")
    tmp = X["irradiation"].rolling(3).agg("std").shift(periods=1)
    X_tr = X_tr.merge(tmp, left_index=True, right_index=True, how="left")
    X_tr.columns = transformer.get_feature_names_out()

    assert df_time_tr.equals(X_tr)


def test_sort_index(df_time):
    # Shuffle dataframe
    Xs = df_time.copy()
    Xs = Xs.sample(frac=1)

    transformer = WindowFeatures(sort_index=False)
    df_tr = transformer.fit_transform(Xs)
    assert_frame_equal(df_tr[transformer.variables_], Xs[transformer.variables_])

    transformer = WindowFeatures(sort_index=True)
    df_tr = transformer.fit_transform(Xs)
    assert_frame_equal(
        df_tr[transformer.variables_], Xs[transformer.variables_].sort_index()
    )


def test_drop_na(df_time):
    df = df_time.head(9).copy()

    expected_results = {
        "ambient_temp": [32.62, 32.5, 32.52, 32.68, 33.76],
        "module_temp": [49.61, 47.01, 46.67, 47.52, 49.8],
        "irradiation": [0.42, 0.49, 0.57, 0.56, 0.74],
        "color": [
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
        ],
        "ambient_temp_window_3_median": [
            31.51,
            32.15,
            32.39,
            32.5,
            32.52,
        ],
        "module_temp_window_3_median": [
            49.84,
            50.63,
            50.63,
            49.61,
            47.01,
        ],
        "irradiation_window_3_median": [
            0.65,
            0.76,
            0.65,
            0.49,
            0.49,
        ],
    }
    expected_results_df = pd.DataFrame(data=expected_results, index=_date_time[4:])

    # Case 1: automatically select variables
    transformer = WindowFeatures(
        window=3, functions=["median"], periods=2, drop_na=True
    )
    df_tr = transformer.fit_transform(df)

    assert df_tr.equals(expected_results_df)

    # Case 2: when drop_original is true
    transformer = WindowFeatures(
        window=3,
        functions=["median"],
        periods=2,
        drop_original=True,
        drop_na=True,
    )
    df_tr = transformer.fit_transform(df)

    assert df_tr.equals(
        expected_results_df.drop(["ambient_temp", "module_temp", "irradiation"], axis=1)
    )

    # Case 3: user indicates multiple variables
    transformer = WindowFeatures(
        variables=["module_temp", "irradiation"],
        window=3,
        functions="median",
        periods=2,
        drop_na=True,
    )
    transformer.fit(df)
    df_tr = transformer.transform(df)

    assert df_tr.equals(
        expected_results_df.drop(["ambient_temp_window_3_median"], axis=1)
    )


def test_transform_x_y(df_time):
    df = df_time.head(9).copy()
    y = pd.Series(np.zeros(len(df)), index=df.index)

    expected_results = {
        "ambient_temp": [32.62, 32.5, 32.52, 32.68, 33.76],
        "module_temp": [49.61, 47.01, 46.67, 47.52, 49.8],
        "irradiation": [0.42, 0.49, 0.57, 0.56, 0.74],
        "color": [
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
        ],
        "ambient_temp_window_3_median": [
            31.51,
            32.15,
            32.39,
            32.5,
            32.52,
        ],
        "module_temp_window_3_median": [
            49.84,
            50.63,
            50.63,
            49.61,
            47.01,
        ],
        "irradiation_window_3_median": [
            0.65,
            0.76,
            0.65,
            0.49,
            0.49,
        ],
    }
    expected_results_df = pd.DataFrame(data=expected_results, index=_date_time[4:])
    transformer = WindowFeatures(
        window=3, functions=["median"], periods=2, drop_na=True
    )

    df_tr = transformer.fit_transform(df)

    assert df_tr.equals(expected_results_df)
    assert len(df_tr) != len(y)

    Xt, yt = transformer.transform_x_y(df, y)
    assert len(Xt) == len(yt)
    assert len(y) != len(yt)
    assert (Xt.index == yt.index).all()
