import numpy as np
import pandas as pd
import pytest

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
    pd.Timestamp('2020-05-15 14:00:00'),
]

@pytest.mark.parametrize("_periods", [1, 2, 3])
def test_permitted_param_periods(_periods):
    transformer = WindowFeatures(periods=_periods)
    assert transformer.periods == _periods


@pytest.mark.parametrize("_periods", ["pizza", 3.33, ["mate", "cumbia"], None])
def test_error_when_non_permitted_param_periods(_periods):
    with pytest.raises(ValueError):
        WindowFeatures(periods=_periods)


def test_get_feature_names_out(df_time):
    # input features
    input_features = ["ambient_temp", "module_temp", "irradiation"]
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
    assert tr.get_feature_names_out(input_features=None) == original_features + output
    assert tr.get_feature_names_out(input_features=input_features) == output
    assert tr.get_feature_names_out(input_features=input_features[0:2]) == output[0:4]
    assert tr.get_feature_names_out(input_features=[input_features[0]]) == output[0:2]

    with pytest.raises(ValueError):
        # get error when a user passes a string instead of list
        tr.get_feature_names_out(input_features=input_features[0])

    with pytest.raises(ValueError):
        # assert error when uses passes features that were not transformed
        tr.get_feature_names_out(input_features=["color"])

    # case 2: 1 window, 1 variable, multiple functions
    tr = WindowFeatures(variables="ambient_temp", window=2, functions=["sum", "mean", "count"])
    tr.fit(df_time)

    # expected
    output = [
        "ambient_temp_window_2_sum",
        "ambient_temp_window_2_mean",
        "ambient_temp_window_2_count",
    ]
    assert tr.get_feature_names_out(input_features=None) == original_features + output
    assert tr.get_feature_names_out(input_features=["ambient_temp"]) == output

    # case 3: multiple windows, multiple variables, multiple functions
    tr = WindowFeatures(window=[2,3], functions=["mean", "sum"])
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
    assert tr.get_feature_names_out(input_features=None) == original_features + output
    assert tr.get_feature_names_out(input_features=input_features) == output
    assert tr.get_feature_names_out(input_features=input_features[0:2]) == output[0:4]+output[6:10]
    assert tr.get_feature_names_out(input_features=[input_features[0]]) == output[0:2]+output[6:8]


def test_single_window_when_using_periods(df_time):

    expected_results = {
        "ambient_temp": [
            31.31, 31.51, 32.15, 32.39, 32.62, 32.5, 32.52, 32.68, 33.76
        ],
        "module_temp": [
            49.18, 49.84, 52.35, 50.63, 49.61, 47.01, 46.67, 47.52, 49.8
        ],
        "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42, 0.49, 0.57, 0.56, 0.74],
        "color": [
            'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        ,
        "ambient_temp_window_3_median": [
            np.nan, np.nan, np.nan, np.nan, 31.51, 32.15, 32.39, 32.5, 32.52
        ],
        "module_temp_window_3_median": [
            np.nan, np.nan, np.nan, np.nan, 49.84, 50.63, 50.63, 49.61, 47.01
        ],
        "irradiation_window_3_median": [
            np.nan, np.nan, np.nan, np.nan, 0.65, 0.76, 0.65, 0.49, 0.49
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
    transformer = WindowFeatures(variables=["module_temp", "irradiation"],
                                 window=3,
                                 functions=["median"],
                                 periods=2
                                 )
    transformer.fit(df_time)
    df_tr = transformer.transform(df_time)

    assert df_tr.head(9).equals(
        expected_results_df.drop(["ambient_temp_window_3_median"], axis=1)
    )

    # Case 4: user indicates 1 variable
    transformer = WindowFeatures(variables=["module_temp"],
                                 window=3,
                                 functions=["median"],
                                 periods=2
                                 )
    transformer.fit(df_time)
    df_tr = transformer.transform(df_time)

    assert df_tr.head(9).equals(
        expected_results_df.drop(["ambient_temp_window_3_median", "irradiation_window_3_median"], axis=1)
    )


def test_single_window_when_using_freq(df_time):

    expected_results = {
        "ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62, 32.5, 32.52, 32.68, 33.76],
        "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61, 47.01, 46.67, 47.52, 49.8],
        "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42, 0.49, 0.57, 0.56, 0.74],
        "color": ['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue'],
        "ambient_temp_window_2_sum": [
            np.nan, np.nan, np.nan, np.nan, 62.82, 63.66, 64.54, 65.01, 65.12
        ],
        "module_temp_window_2_sum": [
            np.nan, np.nan, np.nan, np.nan, 99.02, 102.19, 102.98, 100.24, 96.62
        ],
        "irradiation_window_2_sum": [
            np.nan, np.nan, np.nan, np.nan, 1.3, 1.44, 1.41, 1.18, 0.91]
        ,
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

    assert df_tr.head(9).round(3).equals(
        expected_results_df.drop(["ambient_temp", "module_temp", "irradiation"], axis=1)
    )

    # Case 3: user indicates multiple variables
    transformer = WindowFeatures(
        variables=["ambient_temp", "irradiation"],
        window=2,
        functions=["sum"],
        freq="45min"
    )
    df_tr = transformer.fit_transform(df_time)

    assert df_tr.head(9).round(3).equals(
        expected_results_df.drop(["module_temp_window_2_sum"], axis=1)
    )

    # Case 4: user indicates 1 variable
    transformer = WindowFeatures(
        variables=["irradiation"],
        window=2,
        functions=["sum"],
        freq="45min"
    )
    df_tr = transformer.fit_transform(df_time)

    assert df_tr.head(9).round(3).equals(
        expected_results_df.drop(["ambient_temp_window_2_sum", "module_temp_window_2_sum"], axis=1)
    )

# TODO: complete the values for this test
def test_multiple_windows(df_time):
    expected_results = {
        "ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62, 32.5, 32.52, 32.68, 33.76],
        "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61, 47.01, 46.67, 47.52, 49.8],
        "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42, 0.49, 0.57, 0.56, 0.74],
        "color": ['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue'],
        "ambient_temp_window_2_sum": [
            np.nan, np.nan, np.nan, np.nan, 62.82, 63.66, 64.54, 65.01, 65.12
        ],
        "ambient_temp_window_2_mean": [
            np.nan, np.nan, np.nan, np.nan, xxx, xxx, xxx, xxx, xxx
        ],
        "module_temp_window_2_sum": [
            np.nan, np.nan, np.nan, np.nan, 99.02, 102.19, 102.98, 100.24, 96.62
        ],
        "module_temp_window_2_mean": [
            np.nan, np.nan, np.nan, np.nan, xxx, xxx, xxx, xxx, xxx
        ],
        "irradiation_window_2_sum": [
            np.nan, np.nan, np.nan, np.nan, 1.3, 1.44, 1.41, 1.18, 0.91]
        ,
        "irradiation_window_2_mean": [
            np.nan, np.nan, np.nan, np.nan, xxx, xxx, xxx, xxx, xxx]
        ,
        "ambient_temp_window_3_sum": [
            np.nan, np.nan, np.nan, np.nan, xxx, xxx, xxx, xxx, xxx
        ],
        "ambient_temp_window_3_mean": [
            np.nan, np.nan, np.nan, np.nan, xxx, xxx, xxx, xxx, xxx
        ],
        "module_temp_window_3_sum": [
            np.nan, np.nan, np.nan, np.nan, xxx, xxx, xxx, xxx, xxx
        ],
        "module_temp_window_3_mean": [
            np.nan, np.nan, np.nan, np.nan, xxx, xxx, xxx, xxx, xxx
        ],
        "irradiation_window_3_sum": [
            np.nan, np.nan, np.nan, np.nan, xxx, xxx, xxx, xxx, xxx]
        ,
        "irradiation_window_3_mean": [
            np.nan, np.nan, np.nan, np.nan, xxx, xxx, xxx, xxx, xxx]
        ,
    }

    expected_results_df = pd.DataFrame(data=expected_results, index=_date_time)

    # Case 1: automatically select variables
    transformer = WindowFeatures(window=[2,3], functions=["sum","mean"], periods=45, freq="min")
    df_tr = transformer.fit_transform(df_time)

    assert df_tr.head(9).round(3).equals(expected_results_df)

    # Case 2: when drop_original is true
    transformer = WindowFeatures(
        window=[2,3], functions=["sum","mean"], periods=45, freq="min", drop_original=True
    )
    df_tr = transformer.fit_transform(df_time)

    assert df_tr.head(9).round(3).equals(
        expected_results_df.drop(["ambient_temp", "module_temp", "irradiation"], axis=1)
    )

    # Case 3: user indicates multiple variables
    transformer = WindowFeatures(
        variables=["ambient_temp", "irradiation"],
        window=[2,3], functions=["sum","mean"], periods=45, freq="min"
    )
    df_tr = transformer.fit_transform(df_time)

    assert df_tr.head(9).round(3).equals(
        expected_results_df.drop([xxx,xxx,], axis=1)
    )

    # Case 4: user indicates 1 variable
    transformer = WindowFeatures(
        variables=["irradiation"],
        window=[2,3], functions=["sum","mean"], periods=45, freq="min"
    )
    df_tr = transformer.fit_transform(df_time)

    assert df_tr.head(9).round(3).equals(
        expected_results_df.drop([xxx,xxx,xxx], axis=1)
    )
