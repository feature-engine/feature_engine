import numpy as np
import pandas as pd
import pytest

from feature_engine.timeseries.forecasting import LagFeatures
_false_input_params = [
    ([3, 2], "pizza", ["tango", "empanada"], 9),
    ("mate", ["bombilla", "limpiado"], True, "cocinado"),
    (True, False, "asado", [1984, 1999]),
]


@pytest.mark.parametrize(
    "_variables, _periods, _freq, _drop_original",
    _false_input_params
)
def test_raises_error_when_wrong_input_params(
        _variables, _periods, _freq, _drop_original, df_time
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
        transformer4 = LagFeatures(drop_original=_drop_original)
        transformer4.fit(df_time)
        transformer4.transform(df_time)


_test_init_params = [
    ("empanada", 3, "1d", True),
    (["che", "medialuna"], [2, 4, 6], ["1h", "30m", "4h"], True)
    (None, None, None, False)
]

@pytest.mark.parametrize(
    "_variables, _periods, _freq, _drop_original", _test_init_params
)
def test_class_initiation_params(
        _variables, _periods, _freq, _drop_original
):
    transformer = LagFeatures(
        variables=_variables,
        periods=_periods,
        freq=_freq,
        drop_original=_drop_original
    )
    assert transformer.variables == _variables
    assert transformer.periods == _periods
    assert transformer.freq == _freq
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
        pd.Timestamp('2020-05-15 12:00:00'),
        pd.Timestamp('2020-05-15 12:15:00'),
        pd.Timestamp('2020-05-15 12:30:00'),
        pd.Timestamp('2020-05-15 12:45:00'),
        pd.Timestamp('2020-05-15 13:00:00'),
    ]
    expected_results = {
        "ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62],
        "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61],
        "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42],
        "ambient_temp_lag_3": [np.nan, np.nan, np.nan, 31.31, 31.51],
        "module_temp_lag_3": [np.nan, np.nan, np.nan, 49.18, 49.84],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time
    )

    assert df_tr.head(5).equals(expected_results_df)


def test_time_lag_frequency_shift_and_drop_original_data(df_time):
    # Data is properly transformed using the 'freq' param.
    transformer = LagFeatures(
        freq="1h",
        drop_original=True
    )
    transformer.fit(df_time)
    df_tr = transformer.transform(df_time)

    date_time = [
        pd.Timestamp('2020-05-15 12:00:00'),
        pd.Timestamp('2020-05-15 12:15:00'),
        pd.Timestamp('2020-05-15 12:30:00'),
        pd.Timestamp('2020-05-15 12:45:00'),
        pd.Timestamp('2020-05-15 13:00:00'),
        pd.Timestamp('2020-05-15 13:15:00'),
        pd.Timestamp('2020-05-15 13:30:00'),
        pd.Timestamp('2020-05-15 13:45:00'),
        pd.Timestamp('2020-05-15 14:00:00')
    ]
    expected_results = {
        "ambient_temp_lag_1h": [np.nan, np.nan, np.nan, np.nan,
                                31.31, 31.51, 32.15, 32.39, 32.62],
        "module_temp_lag_1h": [np.nan, np.nan, np.nan, np.nan,
                               49.18, 49.84, 52.35, 50.63, 49.61],
        "irradiation_lag_1h": [np.nan, np.nan, np.nan, np.nan,
                               0.51, 0.79, 0.65, 0.76, 0.42],
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
        pd.Timestamp('2020-05-15 12:00:00'),
        pd.Timestamp('2020-05-15 12:15:00'),
        pd.Timestamp('2020-05-15 12:30:00'),
        pd.Timestamp('2020-05-15 12:45:00'),
        pd.Timestamp('2020-05-15 13:00:00'),
    ]
    expected_results = {
        "ambient_temp_lag_2": [np.nan, np.nan, 31.31, 31.51, 32.15],
        "module_temp_lag_2": [np.nan, np.nan, 49.18, 49.84, 52.35],
        "irradiation_lag_2": [np.nan, np.nan, 0.51, 0.79, 0.65],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time
    )

    assert df_tr.head(5).equals(expected_results_df)


def test_error_when_df_in_transform_is_not_a_dataframe(df_time):
    # case 7: return error if 'df' is not a dataframe
    msg = "X is not a pandas dataframe. The dataset should be a pandas dataframe."
    with pytest.raises(TypeError) as record:
        transformer = LagFeatures(periods=5)
        transformer.transform(df_time["module_temp"])

    # check that error message matches
    assert str(record.value) == msg
