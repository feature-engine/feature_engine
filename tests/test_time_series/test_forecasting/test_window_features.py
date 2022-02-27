import numpy as np
import pandas as pd
import pytest

from feature_engine.timeseries.forecasting import WindowFeatures


@pytest.mark.parametrize("_windows", [3, "1H", "15min", 5])
def test_permitted_param_windows(_windows):
    transformer = WindowFeatures(window=_windows)
    assert transformer.window == _windows


@pytest.mark.parametrize("_windows", [True, [4, "hola"], 4.3])
def test_error_when_non_permitted_param_windows(_windows):
    with pytest.raises(ValueError):
        WindowFeatures(window=_windows)


@pytest.mark.parametrize("_functions", [np.mean, np.std, np.max, np.median])
def test_permitted_param_functions(_functions):
    transformer = WindowFeatures(function=_functions)
    assert transformer.function == _functions


@pytest.mark.parametrize("_functions", ["che", True, np.sin, 1984])
def test_error_when_non_permitted_param_functions(_functions):
    with pytest.raises(ValueError):
        WindowFeatures(function=_functions)


@pytest.mark.parametrize("_periods", [1, 2, 3])
def test_permitted_param_periods(_periods):
    transformer = WindowFeatures(periods=_periods)
    assert transformer.periods == _periods


@pytest.mark.parametrize("_periods", ["pizza", 3.33, True, None])
def test_error_when_non_permitted_param_periods(_periods):
    with pytest.raises(ValueError):
        WindowFeatures(periods=_periods)


@pytest.mark.parametrize("_freqs", ["1h", "90min"])
def test_permitted_param_freq(_freqs):
    transformer = WindowFeatures(freq=_freqs)
    assert transformer.freq == _freqs


@pytest.mark.parametrize("_freqs", [5, "asado", [1, 2, 4], False])
def test_error_when_non_permitted_param_freq(_freqs):
    with pytest.raises(ValueError):
        WindowFeatures(periods=None, freq=_freqs)


def test_get_feature_names_out(df_time):
    # input features
    input_features = ["ambient_temp", "module_temp", "irradiation"]
    original_features = ["ambient_temp", "module_temp", "irradiation", "color"]

    # when freq is a string
    tr = WindowFeatures(window=3, freq="3D")
    tr.fit(df_time)

    # expected
    output = [
        "ambient_temp_window_3_freq_3D",
        "module_temp_window_3_freq_3D",
        "irradiation_window_3_freq_3D",
    ]
    assert tr.get_feature_names_out(input_features=None) == original_features + output
    assert tr.get_feature_names_out(input_features=input_features) == output
    assert tr.get_feature_names_out(input_features=input_features[0:2]) == output[0:2]
    assert tr.get_feature_names_out(input_features=[input_features[0]]) == [output[0]]

    with pytest.raises(ValueError):
        # get error when a user passes a string instead of list
        tr.get_feature_names_out(input_features=input_features[0])

    with pytest.raises(ValueError):
        # assert error when uses passes features that were not transformed
        tr.get_feature_names_out(input_features=["lamp"])

    # when period is an int
    tr = WindowFeatures(window=2, periods=7)
    tr.fit(df_time)

    # expected
    output = [
        "ambient_temp_window_2_periods_7",
        "module_temp_window_2_periods_7",
        "irradiation_window_2_periods_7",
    ]
    assert tr.get_feature_names_out(input_features=None) == original_features + output
    assert tr.get_feature_names_out(input_features=input_features) == output
    assert tr.get_feature_names_out(input_features=input_features[0:2]) == output[0:2]
    assert tr.get_feature_names_out(input_features=[input_features[0]]) == [output[0]]


def test_correct_window_when_using_periods(df_time):
    date_time = [
        "2020-05-15 12:00:00",
        "2020-05-15 12:15:00",
        "2020-05-15 12:30:00",
        "2020-05-15 12:45:00",
        "2020-05-15 13:00:00",
        "2020-05-15 13:15:00",
        "2020-05-15 13:30:00",
        "2020-05-15 13:45:00",
        '2020-05-15 14:00:00',
    ]

    expected_results = {
        "ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62, 32.5, 32.52, 32.68, 33.76],
        "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61, 47.01, 46.67, 47.52, 49.8],
        "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42, 0.49, 0.57, 0.56, 0.74],
        "color": ['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue'],
        "ambient_temp_window_3_periods_2": [np.nan, np.nan, np.nan, np.nan, 31.51, 32.15, 32.39, 32.5, 32.52],
        "module_temp_window_3_periods_2": [np.nan, np.nan, np.nan, np.nan, 49.84, 50.63, 50.63, 49.61, 47.01],
        "irradiation_window_3_periods_2": [np.nan, np.nan, np.nan, np.nan, 0.65, 0.76, 0.65, 0.49, 0.49],
    }
    expected_results_df = pd.DataFrame(data=expected_results, index=date_time)

    transformer = WindowFeatures(window=3, function=np.median, periods=2)
    transformer.fit(df_time)
    df_tr = transformer.transform(df_time)
    print(df_tr.head())
    # assert df_tr.columns.tolist() == expected_results_df.columns.tolist()
    assert df_tr.head(9).equals(expected_results_df)