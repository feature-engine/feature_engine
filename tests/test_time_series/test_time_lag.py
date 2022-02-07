import numpy as np
import pandas as pd
import pytest

from feature_engine.timeseries.forecasting import LagFeatures


@pytest.mark.parametrize('_periods', [1, [1,2,3]])
def test_permitted_param_periods(_periods):
    transformer = LagFeatures(periods=_periods)
    assert transformer.periods == _periods


@pytest.mark.parametrize('_periods', [-1, None, [-1,2,3], [0.1, 1, 2], 0.5])
def test_error_when_non_permitted_param_periods(_periods):
    with pytest.raises(ValueError):
        LagFeatures(periods=_periods)


def test_get_feature_names_out(df_time):

    # input features
    input_features = ["ambient_temp", "module_temp", "irradiation"]
    original_features = ["ambient_temp", "module_temp", "irradiation", "color"]

    # When freq is a string:
    tr = LagFeatures(freq="1D")
    tr.fit(df_time)

    # Expected
    out = ["ambient_temp_lag_1D", "module_temp_lag_1D", "irradiation_lag_1D"]
    assert tr.get_feature_names_out(input_features=None) == original_features + out
    assert tr.get_feature_names_out(input_features=input_features) == out
    assert tr.get_feature_names_out(input_features=input_features[0:1]) == out[0:1]
    assert tr.get_feature_names_out(input_features=[input_features[0]]) == [out[0]]

    with pytest.raises(ValueError):
        # assert error when user passes a string instead of list
        tr.get_feature_names_out(input_features=input_features[0])

    # When period is an int:
    tr = LagFeatures(periods=2)
    tr.fit(df_time)

    # Expected
    out = ["ambient_temp_lag_2", "module_temp_lag_2", "irradiation_lag_2"]
    assert tr.get_feature_names_out(input_features=None) == original_features + out
    assert tr.get_feature_names_out(input_features=input_features) == out
    assert tr.get_feature_names_out(input_features=input_features[0:1]) == out[0:1]
    assert tr.get_feature_names_out(input_features=[input_features[0]]) == [out[0]]

    # When freq is a list:
    tr = LagFeatures(freq=["3D", "2D"])
    tr.fit(df_time)

    # Expected
    out = [
        "ambient_temp_lag_3D", "module_temp_lag_3D", "irradiation_lag_3D",
        "ambient_temp_lag_2D", "module_temp_lag_2D", "irradiation_lag_2D",
    ]

    assert tr.get_feature_names_out(input_features=None) == original_features + out
    assert tr.get_feature_names_out(input_features=input_features) == out
    assert tr.get_feature_names_out(input_features=input_features[0:1]) == out[0:1] + out[3:4]
    assert tr.get_feature_names_out(input_features=[input_features[0]]) == [out[0]] + [out[3]]

    # When periods is a list:
    tr = LagFeatures(periods=[2, 3])
    tr.fit(df_time)

    # Expected
    out = [
        "ambient_temp_lag_2", "module_temp_lag_2", "irradiation_lag_2",
        "ambient_temp_lag_3", "module_temp_lag_3", "irradiation_lag_3",
    ]

    assert tr.get_feature_names_out(input_features=None) == original_features + out
    assert tr.get_feature_names_out(input_features=input_features) == out
    assert tr.get_feature_names_out(input_features=input_features[0:1]) == out[0:1] + out[3:4]
    assert tr.get_feature_names_out(input_features=[input_features[0]]) == [out[0]] + [out[3]]

    # When drop original is true.
    tr = LagFeatures(freq="1D", drop_original=True)
    tr.fit(df_time)

    # Expected
    out = ["ambient_temp_lag_1D", "module_temp_lag_1D", "irradiation_lag_1D"]
    assert tr.get_feature_names_out(input_features=None) == ["color"] + out
    assert tr.get_feature_names_out(input_features=input_features) == out

# def test_time_lag_period_shift_and_keep_original_data(df_time):
#     # The lag is correctly performed using the 'period' param.
#     transformer = LagFeatures(
#         variables=["ambient_temp", "module_temp"],
#         periods=3,
#         drop_original=False,
#     )
#     transformer.fit(df_time)
#     df_tr = transformer.transform(df_time)
#
#     date_time = [
#         pd.Timestamp('2020-05-15 12:00:00'),
#         pd.Timestamp('2020-05-15 12:15:00'),
#         pd.Timestamp('2020-05-15 12:30:00'),
#         pd.Timestamp('2020-05-15 12:45:00'),
#         pd.Timestamp('2020-05-15 13:00:00'),
#     ]
#     expected_results = {
#         "ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62],
#         "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61],
#         "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42],
#         "ambient_temp_lag_3": [np.nan, np.nan, np.nan, 31.31, 31.51],
#         "module_temp_lag_3": [np.nan, np.nan, np.nan, 49.18, 49.84],
#     }
#     expected_results_df = pd.DataFrame(
#         data=expected_results,
#         index=date_time
#     )
#
#     assert df_tr.head(5).equals(expected_results_df)
#
#
# def test_time_lag_frequency_shift_and_drop_original_data(df_time):
#     # Data is properly transformed using the 'freq' param.
#     transformer = LagFeatures(
#         freq="1h",
#         drop_original=True
#     )
#     transformer.fit(df_time)
#     df_tr = transformer.transform(df_time)
#
#     date_time = [
#         pd.Timestamp('2020-05-15 12:00:00'),
#         pd.Timestamp('2020-05-15 12:15:00'),
#         pd.Timestamp('2020-05-15 12:30:00'),
#         pd.Timestamp('2020-05-15 12:45:00'),
#         pd.Timestamp('2020-05-15 13:00:00'),
#         pd.Timestamp('2020-05-15 13:15:00'),
#         pd.Timestamp('2020-05-15 13:30:00'),
#         pd.Timestamp('2020-05-15 13:45:00'),
#         pd.Timestamp('2020-05-15 14:00:00')
#     ]
#     expected_results = {
#         "ambient_temp_lag_1h": [np.nan, np.nan, np.nan, np.nan,
#                                 31.31, 31.51, 32.15, 32.39, 32.62],
#         "module_temp_lag_1h": [np.nan, np.nan, np.nan, np.nan,
#                                49.18, 49.84, 52.35, 50.63, 49.61],
#         "irradiation_lag_1h": [np.nan, np.nan, np.nan, np.nan,
#                                0.51, 0.79, 0.65, 0.76, 0.42],
#     }
#     expected_results_df = pd.DataFrame(
#         data=expected_results,
#         index=date_time,
#     )
#
#     assert df_tr.head(9).equals(expected_results_df)
#
#
# def test_time_lag_periods_drop_original_value(df_time):
#     transformer = LagFeatures(
#         periods=2,
#         drop_original=True,
#     )
#     transformer.fit(df_time)
#     df_tr = transformer.transform(df_time)
#
#     date_time = [
#         pd.Timestamp('2020-05-15 12:00:00'),
#         pd.Timestamp('2020-05-15 12:15:00'),
#         pd.Timestamp('2020-05-15 12:30:00'),
#         pd.Timestamp('2020-05-15 12:45:00'),
#         pd.Timestamp('2020-05-15 13:00:00'),
#     ]
#     expected_results = {
#         "ambient_temp_lag_2": [np.nan, np.nan, 31.31, 31.51, 32.15],
#         "module_temp_lag_2": [np.nan, np.nan, 49.18, 49.84, 52.35],
#         "irradiation_lag_2": [np.nan, np.nan, 0.51, 0.79, 0.65],
#     }
#     expected_results_df = pd.DataFrame(
#         data=expected_results,
#         index=date_time
#     )
#
#     assert df_tr.head(5).equals(expected_results_df)
#
#
# def test_error_when_df_in_transform_is_not_a_dataframe(df_time):
#     # case 7: return error if 'df' is not a dataframe
#     msg = "X is not a pandas dataframe. The dataset should be a pandas dataframe."
#     with pytest.raises(TypeError) as record:
#         transformer = LagFeatures(periods=5)
#         transformer.transform(df_time["module_temp"])
#
#     # check that error message matches
#     assert str(record.value) == msg
