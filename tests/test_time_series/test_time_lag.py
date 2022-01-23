import pandas as pd
import pytest

from feature_engine.timeseries.forecasting import TimeSeriesLagTransformer


def test_time_lag_period_shift_and_keep_original_data(df_time):
    # case 1: The lag is correctly performed using the 'period' param.
    transformer = TimeSeriesLagTransformer(
        variables=["ambient_temp", "module_temp"],
        periods=3,
        keep_original=True,
    )

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
        "ambient_temp_lag_3pds": [nan, nan, nan, 31.31, 31.51],
        "module_temp_lag_3pds": [nan, nan, nan, 49.18, 49.84],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time
    )

    assert df_tr.head(5) == expected_results_df


def test_time_lag_frequency_shift_and_ignore_original_data(df_time):
    # case 2: Data is properly transformed using the 'freq' param.
    transformer = TimeSeriesLagTransformer(
        freq="1h",
        keep_original=False
    )
    df_tr = transformer.transform(df_time)

    date_time = [
        pd.Timestamp('2020-05-15 13:00:00'),
        pd.Timestamp('2020-05-15 13:15:00'),
        pd.Timestamp('2020-05-15 13:30:00'),
        pd.Timestamp('2020-05-15 13:45:00'),
        pd.Timestamp('2020-05-15 14:00:00')
    ]
    expected_results = {
        "ambient_temp_lag_1h": [31.31, 31.51, 32.15, 32.39, 32.62],
        "module_temp_lag_1h": [49.18, 49.84, 52.35, 50.63, 49.61],
        "irradiation_lag_1h": [0.51, 0.79, 0.65, 0.76, 0.42],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time,
    )

    assert df_tr.head(5) == expected_results_df


def test_time_lag_fill_value(df_time):
    # case 3: test that the NaN values are correctly filled
    transformer = TimeSeriesLagTransformer(
        periods=2,
        fill_value="test_fill",
        keep_original=True
    )
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
        "ambient_temp_lag_2pds": ['test_fill', 'test_fill', 31.31, 31.51, 32.15],
        "module_temp_lag_2pds": ['test_fill', 'test_fill', 49.18, 49.84, 52.35],
        "irradiation_lag_2pds": ['test_fill', 'test_fill', 0.51, 0.79, 0.65],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time
    )

    assert df_tr.head(5) == expected_results_df


def test_incorrect_periods_during_installation(df_time):
    # case 4: test warning when inappropriate value has been inputted for
    # the 'periods' param
    with pytest.raises(ValueError):
        transformer = TimeSeriesLagTransformer(periods="cumbia")


def test_incorrect_axis_during_installation(df_time):
    # case 4: test warning when inappropriate value has been inputted for
    # the 'periods' param
    with pytest.raises(ValueError):
        transformer = TimeSeriesLagTransformer(axis="cumbia")


def test_incorrect_keep_original_during_installation(df_time):
    # case 4: test warning when inappropriate value has been inputted for
    # the 'periods' param
    with pytest.raises(ValueError):
        transformer = TimeSeriesLagTransformer(keep_original="cumbia")