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
    assert tr.get_feature_names_out(input_features=input_features[0]) == [output[0]]