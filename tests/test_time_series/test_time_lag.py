import pandas as pd
import pytest

from feature_engine.timeseries.forecasting import TimeSeriesLagTransformer
# TODO: Need to 'freq' variable
_false_input_params = [
    ([3, 2], "pizza", 9),
    ("mate", ["bombilla", "limpiado"], "cocinado"),
    (True, False, [1984, 1999]),
]

# TODO: check test
@pytest.mark.parametrize(
    "_variables, _periods, _drop_original", _false_input_params
)
def test_raises_error_when_wrong_input_params(
        _variables, _periods, _drop_original
):
    #TODO: Add test with 'freq' param
    with pytest.raises(ValueError):
        assert TimeSeriesLagTransformer(variables=_variables)
    with pytest.raises(ValueError):
        assert TimeSeriesLagTransformer(periods=_periods)
    with pytest.raises(ValueError):
        assert TimeSeriesLagTransformer(drop_original=_drop_original)


def test_default_params():
    transformer = TimeSeriesLagTransformer()
    assert transformer.variables is None
    assert transformer.periods == 1
    assert transformer.freq is None
    assert transformer.drop_original is False


def test_time_lag_period_shift_and_keep_original_data(df_time):
    # case 1: The lag is correctly performed using the 'period' param.
    transformer = TimeSeriesLagTransformer(
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
        "ambient_temp_lag_3": ['nan', 'nan', 'nan', 31.31, 31.51],
        "module_temp_lag_3": ['nan', 'nan', 'nan', 49.18, 49.84],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time
    )

    assert df_tr.head(5) == expected_results_df


def test_time_lag_frequency_shift_and_drop_original_data(df_time):
    # case 2: Data is properly transformed using the 'freq' param.
    transformer = TimeSeriesLagTransformer(
        freq="1h",
        drop_original=True
    )
    transformer.fit(df_time)
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
        "ambient_temp_lag_2": ['nan', 'nan', 31.31, 31.51, 32.15],
        "module_temp_lag_2": ['nan', 'nan', 49.18, 49.84, 52.35],
        "irradiation_lag_2": ['nan', 'nan', 0.51, 0.79, 0.65],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time
    )

    assert df_tr.head(5) == expected_results_df


def test_error_when_df_in_transform_is_not_a_dataframe(df_time):
    # case 7: return error if 'df' is not a dataframe
    msg = "X is not a pandas dataframe. The dataset should be a pandas dataframe."
    with pytest.raises(TypeError) as record:
        transformer = TimeSeriesLagTransformer(periods=5)
        transformer.transform(df_time["module_temp"])

    # check that error message matches
    assert str(record.value) == msg


def test_incorrect_freq_during_installation(df_time):
    # case 6: return error when inappropriate value has been inputted for
    # the 'freq' param
    # ADD TEST
    pass