import pandas as pd
import pytest

from feature_engine.timeseries.forecasting import TimeSeriesLagTransformer


def test_time_lag_period_shift(df_time):
    # Case 1: The lag is correctly performed using the 'period' param.
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
