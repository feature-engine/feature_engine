import numpy as np
import pandas as pd
import pytest

from feature_engine.timeseries.forecasting import LagFeatures


@pytest.mark.parametrize("_periods", [1, [1, 2, 3]])
def test_permitted_param_periods(_periods):
    transformer = LagFeatures(periods=_periods)
    assert transformer.periods == _periods


@pytest.mark.parametrize("_periods", [-1, None, [-1, 2, 3], [0.1, 1, 2], 0.5])
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
        "ambient_temp_lag_3D",
        "module_temp_lag_3D",
        "irradiation_lag_3D",
        "ambient_temp_lag_2D",
        "module_temp_lag_2D",
        "irradiation_lag_2D",
    ]

    assert tr.get_feature_names_out(input_features=None) == original_features + out
    assert tr.get_feature_names_out(input_features=input_features) == out
    assert (
        tr.get_feature_names_out(input_features=input_features[0:1])
        == out[0:1] + out[3:4]
    )
    assert tr.get_feature_names_out(input_features=[input_features[0]]) == [out[0]] + [
        out[3]
    ]

    # When periods is a list:
    tr = LagFeatures(periods=[2, 3])
    tr.fit(df_time)

    # Expected
    out = [
        "ambient_temp_lag_2",
        "module_temp_lag_2",
        "irradiation_lag_2",
        "ambient_temp_lag_3",
        "module_temp_lag_3",
        "irradiation_lag_3",
    ]

    assert tr.get_feature_names_out(input_features=None) == original_features + out
    assert tr.get_feature_names_out(input_features=input_features) == out
    assert (
        tr.get_feature_names_out(input_features=input_features[0:1])
        == out[0:1] + out[3:4]
    )
    assert tr.get_feature_names_out(input_features=[input_features[0]]) == [out[0]] + [
        out[3]
    ]

    # When drop original is true.
    tr = LagFeatures(freq="1D", drop_original=True)
    tr.fit(df_time)

    # Expected
    out = ["ambient_temp_lag_1D", "module_temp_lag_1D", "irradiation_lag_1D"]
    assert tr.get_feature_names_out(input_features=None) == ["color"] + out
    assert tr.get_feature_names_out(input_features=input_features) == out


def test_correct_lag_when_using_periods(df_time):
    # Expected
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
        "color": ["blue"] * 5,
        "ambient_temp_lag_3": [np.nan, np.nan, np.nan, 31.31, 31.51],
        "module_temp_lag_3": [np.nan, np.nan, np.nan, 49.18, 49.84],
        "ambient_temp_lag_2": [np.nan, np.nan, 31.31, 31.51, 32.15],
        "module_temp_lag_2": [np.nan, np.nan, 49.18, 49.84, 52.35],
    }
    expected_results_df = pd.DataFrame(data=expected_results, index=date_time)

    # When period is an int.
    transformer = LagFeatures(variables=["ambient_temp", "module_temp"], periods=3)
    df_tr = transformer.fit_transform(df_time)
    assert df_tr.head(5).equals(
        expected_results_df.drop(["ambient_temp_lag_2", "module_temp_lag_2"], axis=1)
    )

    # When period is list.
    transformer = LagFeatures(variables=["ambient_temp", "module_temp"], periods=[3, 2])
    df_tr = transformer.fit_transform(df_time)
    assert df_tr.head(5).equals(expected_results_df)

    # When drop original is True
    transformer = LagFeatures(
        variables=["ambient_temp", "module_temp"], periods=[3, 2], drop_original=True
    )
    df_tr = transformer.fit_transform(df_time)
    assert df_tr.head(5).equals(
        expected_results_df.drop(["ambient_temp", "module_temp"], axis=1)
    )


def test_correct_lag_when_using_freq(df_time):
    # Expected
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
        "color": ["blue"] * 5,
        "ambient_temp_lag_1h": [np.nan, np.nan, np.nan, np.nan, 31.31],
        "module_temp_lag_1h": [np.nan, np.nan, np.nan, np.nan, 49.18],
        "irradiation_lag_1h": [np.nan, np.nan, np.nan, np.nan, 0.51],
        "ambient_temp_lag_15min": [np.nan, 31.31, 31.51, 32.15, 32.39],
        "module_temp_lag_15min": [np.nan, 49.18, 49.84, 52.35, 50.63],
        "irradiation_lag_15min": [np.nan, 0.51, 0.79, 0.65, 0.76],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time,
    )

    # When freq is a string
    transformer = LagFeatures(freq="1h")
    df_tr = transformer.fit_transform(df_time)
    assert df_tr.head(5).equals(
        expected_results_df.drop(
            [
                "ambient_temp_lag_15min",
                "module_temp_lag_15min",
                "irradiation_lag_15min",
            ],
            axis=1,
        )
    )

    # When freq is a list
    transformer = LagFeatures(freq=["1h", "15min"])
    df_tr = transformer.fit_transform(df_time)
    assert df_tr.head(5).equals(expected_results_df)

    # When drop original is True
    transformer = LagFeatures(freq=["1h"], drop_original=True)
    df_tr = transformer.fit_transform(df_time)
    assert df_tr.head(5).equals(
        expected_results_df[
            ["color", "ambient_temp_lag_1h", "module_temp_lag_1h", "irradiation_lag_1h"]
        ]
    )
