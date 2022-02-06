import numpy as np
import pandas as pd
import pytest

from feature_engine.timeseries.forecasting import LagFeatures

_false_input_params = [
    ([3, 2], "pizza", ["tango", "empanada"], "tigre", 9),
    ("mate", ["bombilla", "limpiado"], True, False, "cocinado"),
    (True, -3, "asado", 4, [1984, 1999]),
    ("alfajor", [3, -8, 10], ["jamon", "merienda"], ["mate", "bombilla"], 10),
    ("asado", False, "empanada", None, ["tango", "empanada"]),
]

_test_types_init_params = [
    ("empanada", 3, "1d", "raise", True),
    (["che", "medialuna"], [2, 4, 6], ["1h", "30m", "4h"], "ignore", True),
    (None, None, None, "raise", False),
]


@pytest.mark.parametrize(
    "_variables, _periods, _freq, _missing_values, _drop_original", _false_input_params
)
def test_raises_error_when_wrong_input_params(
    _variables, _periods, _freq, _missing_values, _drop_original, df_time
):
    with pytest.raises(KeyError):
        transformer1 = LagFeatures(variables=_variables)
        transformer1.fit(df_time)
        transformer1.transform(df_time)
    with pytest.raises(ValueError):
        transformer2 = LagFeatures(periods=_periods)
        transformer2.fit(df_time)
        transformer2.transform(df_time)
    with pytest.raises((ValueError, TypeError)):
        transformer3 = LagFeatures(freq=_freq)
        transformer3.fit(df_time)
        transformer3.transform(df_time)
    with pytest.raises(ValueError):
        transformer4 = LagFeatures(missing_values=_missing_values)
        transformer4.fit(df_time)
        transformer4.transform(df_time)
    with pytest.raises(ValueError):
        transformer5 = LagFeatures(drop_original=_drop_original)
        transformer5.fit(df_time)
        transformer5.transform(df_time)


@pytest.mark.parametrize(
    "_variables, _periods, _freq, _missing_values, _drop_original", _test_types_init_params
)
def test_different_types_for_init_params(
        _variables, _periods, _freq, _missing_values, _drop_original
):
    transformer = LagFeatures(
        variables=_variables,
        periods=_periods,
        freq=_freq,
        missing_values=_missing_values,
        drop_original=_drop_original
    )
    assert transformer.variables == _variables
    assert transformer.periods == _periods
    assert transformer.freq == _freq
    assert transformer.missing_values == _missing_values
    assert transformer.drop_original == _drop_original


def test_time_lag_period_shift_and_keep_original_data(df_time):
    # The lag is correctly performed using the 'period' param.
    transformer = LagFeatures(
        variables=["ambient_temp", "module_temp"],
        periods=3,
        drop_original=False,
    )
    transformer.fit(df_time)
    df_tr = transformer.transform(df_time)

    date_time = [
        pd.Timestamp("2020-05-15 12:00:00"),
        pd.Timestamp("2020-05-15 12:15:00"),
        pd.Timestamp("2020-05-15 12:30:00"),
        pd.Timestamp("2020-05-15 12:45:00"),
        pd.Timestamp("2020-05-15 13:00:00"),
    ]
    expected_results = {
        "ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62],
        "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61],
        "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42],
        "ambient_temp_lag_3": [np.nan, np.nan, np.nan, 31.31, 31.51],
        "module_temp_lag_3": [np.nan, np.nan, np.nan, 49.18, 49.84],
    }
    expected_results_df = pd.DataFrame(data=expected_results, index=date_time)

    assert df_tr.head(5).equals(expected_results_df)


def test_time_lag_frequency_shift_and_drop_original_data(df_time):
    # Data is properly transformed using the 'freq' param.
    transformer = LagFeatures(freq="1h", drop_original=True)
    transformer.fit(df_time)
    df_tr = transformer.transform(df_time)

    date_time = [
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
    expected_results = {
        "ambient_temp_lag_1h": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            31.31,
            31.51,
            32.15,
            32.39,
            32.62,
        ],
        "module_temp_lag_1h": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            49.18,
            49.84,
            52.35,
            50.63,
            49.61,
        ],
        "irradiation_lag_1h": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.51,
            0.79,
            0.65,
            0.76,
            0.42,
        ],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time,
    )

    assert df_tr.head(9).equals(expected_results_df)


def test_time_lag_periods_drop_original_value(df_time):
    transformer = LagFeatures(
        periods=2,
        drop_original=True,
    )
    transformer.fit(df_time)
    df_tr = transformer.transform(df_time)

    date_time = [
        pd.Timestamp("2020-05-15 12:00:00"),
        pd.Timestamp("2020-05-15 12:15:00"),
        pd.Timestamp("2020-05-15 12:30:00"),
        pd.Timestamp("2020-05-15 12:45:00"),
        pd.Timestamp("2020-05-15 13:00:00"),
    ]
    expected_results = {
        "ambient_temp_lag_2": [np.nan, np.nan, 31.31, 31.51, 32.15],
        "module_temp_lag_2": [np.nan, np.nan, 49.18, 49.84, 52.35],
        "irradiation_lag_2": [np.nan, np.nan, 0.51, 0.79, 0.65],
    }
    expected_results_df = pd.DataFrame(data=expected_results, index=date_time)

    assert df_tr.head(5).equals(expected_results_df)


def test_get_feature_names_out(df_time):
    transformer = LagFeatures(
        freq="1h",
        drop_original=True
    )
    transformer.fit(df_time)
    feature_names = transformer.get_feature_names_out()

    assert feature_names == (
        ["ambient_temp_lag_1h", "module_temp_lag_1h", "irradiation_lag_1h"]
    )
