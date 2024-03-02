import numpy as np
import pandas as pd
import pytest

from feature_engine.timeseries.forecasting import LagFeatures


@pytest.mark.parametrize("_periods", [1, [1, 2, 3]])
def test_permitted_param_periods(_periods):
    transformer = LagFeatures(periods=_periods)
    assert transformer.periods == _periods


@pytest.mark.parametrize(
    "_periods", [-1, 0, None, [-1, 2, 3], [0.1, 1], 0.5, [0, 1], [1, 1, 2]]
)
def test_error_when_non_permitted_param_periods(_periods):
    with pytest.raises(ValueError):
        LagFeatures(periods=_periods)


def test_error_when_non_permitted_param_freq():
    with pytest.raises(ValueError):
        LagFeatures(freq=["2h", "2h", "3h"])


@pytest.mark.parametrize("_sort_index", [True, False])
def test_permitted_param_sort_index(_sort_index):
    transformer = LagFeatures(sort_index=_sort_index)
    assert transformer.sort_index == _sort_index


@pytest.mark.parametrize("_sort_index", [-1, None, "hola"])
def test_error_when_non_permitted_param_sort_index(_sort_index):
    with pytest.raises(ValueError):
        LagFeatures(sort_index=_sort_index)


def test_get_feature_names_out(df_time):
    # input features
    original_features = ["ambient_temp", "module_temp", "irradiation", "color"]

    # When freq is a string:
    tr = LagFeatures(freq="1D")
    tr.fit(df_time)

    # Expected
    out = ["ambient_temp_lag_1D", "module_temp_lag_1D", "irradiation_lag_1D"]
    out = original_features + out
    assert tr.get_feature_names_out(input_features=None) == out
    assert tr.get_feature_names_out(input_features=original_features) == out
    assert tr.get_feature_names_out(input_features=df_time.columns) == out

    with pytest.raises(ValueError):
        # assert error when user passes a string instead of list
        tr.get_feature_names_out(input_features=original_features[0])

    with pytest.raises(ValueError):
        # assert error when uses passes features that were not lagged
        tr.get_feature_names_out(input_features=["color"])

    # When period is an int:
    tr = LagFeatures(periods=2)
    tr.fit(df_time)

    # Expected
    out = ["ambient_temp_lag_2", "module_temp_lag_2", "irradiation_lag_2"]
    out = original_features + out
    assert tr.get_feature_names_out(input_features=None) == out
    assert tr.get_feature_names_out(input_features=original_features) == out
    assert tr.get_feature_names_out(input_features=df_time.columns) == out

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
    out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == out
    assert tr.get_feature_names_out(input_features=df_time.columns) == out

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
    out = original_features + out

    assert tr.get_feature_names_out(input_features=None) == out
    assert tr.get_feature_names_out(input_features=original_features) == out

    # When drop original is true.
    tr = LagFeatures(freq="1D", drop_original=True)
    tr.fit(df_time)

    # Expected
    out = ["ambient_temp_lag_1D", "module_temp_lag_1D", "irradiation_lag_1D"]
    assert tr.get_feature_names_out(input_features=None) == ["color"] + out
    assert tr.get_feature_names_out(input_features=original_features) == ["color"] + out


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


def test_sort_index(df_time):
    X = df_time.copy()

    # Shuffle dataframe
    Xs = X.sample(len(df_time)).copy()

    transformer = LagFeatures(sort_index=True)
    X_tr = transformer.fit_transform(Xs)

    A = X[transformer.variables_].iloc[0:4].values
    B = X_tr[transformer._get_new_features_name()].iloc[1:5].values
    assert (A == B).all()

    transformer = LagFeatures(sort_index=False)
    X_tr = transformer.fit_transform(Xs)

    A = Xs[transformer.variables_].iloc[0:4].values
    B = X_tr[transformer._get_new_features_name()].iloc[1:5].values
    assert (A == B).all()


def test_correct_groupby_lag_when_using_periods(df_time):
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
        pd.Timestamp("2020-05-15 14:15:00"),
        pd.Timestamp("2020-05-15 14:30:00"),
        pd.Timestamp("2020-05-15 14:45:00"),
        pd.Timestamp("2020-05-15 15:00:00"),
        pd.Timestamp("2020-05-15 15:15:00"),
        pd.Timestamp("2020-05-15 15:30:00"),
    ]
    expected_results = {
        "ambient_temp": [
            31.31,
            31.51,
            32.15,
            32.39,
            32.62,
            32.5,
            32.52,
            32.68,
            33.76,
            34.13,
            34.08,
            33.7,
            33.89,
            34.04,
            34.4,
        ],
        "module_temp": [
            49.18,
            49.84,
            52.35,
            50.63,
            49.61,
            47.01,
            46.67,
            47.52,
            49.8,
            55.03,
            54.52,
            47.62,
            46.03,
            44.29,
            46.74,
        ],
        "irradiation": [
            0.51,
            0.79,
            0.65,
            0.76,
            0.42,
            0.49,
            0.57,
            0.56,
            0.74,
            0.89,
            0.47,
            0.54,
            0.4,
            0.45,
            0.57,
        ],
        "color": [
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "green",
            "green",
            "green",
            "green",
            "green",
        ],
        "ambient_temp_lag_3": [
            np.nan,
            np.nan,
            np.nan,
            31.31,
            31.51,
            32.15,
            32.39,
            32.62,
            32.5,
            32.52,
            np.nan,
            np.nan,
            np.nan,
            34.08,
            33.7,
        ],
        "irradiation_lag_3": [
            np.nan,
            np.nan,
            np.nan,
            0.51,
            0.79,
            0.65,
            0.76,
            0.42,
            0.49,
            0.57,
            np.nan,
            np.nan,
            np.nan,
            0.47,
            0.54,
        ],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time,
    )
    # When setting group_by to color
    transformer = LagFeatures(
        variables=["ambient_temp", "irradiation"], periods=3, group_by="color"
    )
    df_tr = transformer.fit_transform(df_time)
    assert df_tr.equals(expected_results_df)


def test_multiple_periods_with_groupby(df_time):
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
        pd.Timestamp("2020-05-15 14:15:00"),
        pd.Timestamp("2020-05-15 14:30:00"),
        pd.Timestamp("2020-05-15 14:45:00"),
        pd.Timestamp("2020-05-15 15:00:00"),
        pd.Timestamp("2020-05-15 15:15:00"),
        pd.Timestamp("2020-05-15 15:30:00"),
    ]
    expected_results = {
        "ambient_temp": [
            31.31,
            31.51,
            32.15,
            32.39,
            32.62,
            32.5,
            32.52,
            32.68,
            33.76,
            34.13,
            34.08,
            33.7,
            33.89,
            34.04,
            34.4,
        ],
        "module_temp": [
            49.18,
            49.84,
            52.35,
            50.63,
            49.61,
            47.01,
            46.67,
            47.52,
            49.8,
            55.03,
            54.52,
            47.62,
            46.03,
            44.29,
            46.74,
        ],
        "irradiation": [
            0.51,
            0.79,
            0.65,
            0.76,
            0.42,
            0.49,
            0.57,
            0.56,
            0.74,
            0.89,
            0.47,
            0.54,
            0.4,
            0.45,
            0.57,
        ],
        "color": [
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "green",
            "green",
            "green",
            "green",
            "green",
        ],
        "ambient_temp_lag_2": [
            np.nan,
            np.nan,
            31.31,
            31.51,
            32.15,
            32.39,
            32.62,
            32.5,
            32.52,
            32.68,
            np.nan,
            np.nan,
            34.08,
            33.7,
            33.89,
        ],
        "irradiation_lag_2": [
            np.nan,
            np.nan,
            0.51,
            0.79,
            0.65,
            0.76,
            0.42,
            0.49,
            0.57,
            0.56,
            np.nan,
            np.nan,
            0.47,
            0.54,
            0.4,
        ],
        "ambient_temp_lag_3": [
            np.nan,
            np.nan,
            np.nan,
            31.31,
            31.51,
            32.15,
            32.39,
            32.62,
            32.5,
            32.52,
            np.nan,
            np.nan,
            np.nan,
            34.08,
            33.7,
        ],
        "irradiation_lag_3": [
            np.nan,
            np.nan,
            np.nan,
            0.51,
            0.79,
            0.65,
            0.76,
            0.42,
            0.49,
            0.57,
            np.nan,
            np.nan,
            np.nan,
            0.47,
            0.54,
        ],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time,
    )
    # When setting group_by to color
    transformer = LagFeatures(
        variables=["ambient_temp", "irradiation"], periods=[2, 3], group_by="color"
    )
    df_tr = transformer.fit_transform(df_time)
    assert df_tr.equals(expected_results_df)


def test_correct_groupby_lag_when_using_freq(df_time):
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
        pd.Timestamp("2020-05-15 14:15:00"),
        pd.Timestamp("2020-05-15 14:30:00"),
        pd.Timestamp("2020-05-15 14:45:00"),
        pd.Timestamp("2020-05-15 15:00:00"),
        pd.Timestamp("2020-05-15 15:15:00"),
        pd.Timestamp("2020-05-15 15:30:00"),
    ]
    expected_results = {
        "ambient_temp": [
            31.31,
            31.51,
            32.15,
            32.39,
            32.62,
            32.5,
            32.52,
            32.68,
            33.76,
            34.13,
            34.08,
            33.7,
            33.89,
            34.04,
            34.4,
        ],
        "module_temp": [
            49.18,
            49.84,
            52.35,
            50.63,
            49.61,
            47.01,
            46.67,
            47.52,
            49.8,
            55.03,
            54.52,
            47.62,
            46.03,
            44.29,
            46.74,
        ],
        "irradiation": [
            0.51,
            0.79,
            0.65,
            0.76,
            0.42,
            0.49,
            0.57,
            0.56,
            0.74,
            0.89,
            0.47,
            0.54,
            0.4,
            0.45,
            0.57,
        ],
        "color": [
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "green",
            "green",
            "green",
            "green",
            "green",
        ],
        "irradiation_lag_15min": [
            np.nan,
            0.51,
            0.79,
            0.65,
            0.76,
            0.42,
            0.49,
            0.57,
            0.56,
            0.74,
            np.nan,
            0.47,
            0.54,
            0.4,
            0.45,
        ],
        "ambient_temp_lag_15min": [
            np.nan,
            31.31,
            31.51,
            32.15,
            32.39,
            32.62,
            32.5,
            32.52,
            32.68,
            33.76,
            np.nan,
            34.08,
            33.7,
            33.89,
            34.04,
        ],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time,
    )
    # When setting group_by to color
    transformer = LagFeatures(
        variables=["irradiation", "ambient_temp"], freq="15min", group_by="color"
    )
    df_tr = transformer.fit_transform(df_time)
    assert df_tr.equals(expected_results_df)


def test_multiple_freq_with_groupby(df_time):
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
        pd.Timestamp("2020-05-15 14:15:00"),
        pd.Timestamp("2020-05-15 14:30:00"),
        pd.Timestamp("2020-05-15 14:45:00"),
        pd.Timestamp("2020-05-15 15:00:00"),
        pd.Timestamp("2020-05-15 15:15:00"),
        pd.Timestamp("2020-05-15 15:30:00"),
    ]
    expected_results = {
        "ambient_temp": [
            31.31,
            31.51,
            32.15,
            32.39,
            32.62,
            32.5,
            32.52,
            32.68,
            33.76,
            34.13,
            34.08,
            33.7,
            33.89,
            34.04,
            34.4,
        ],
        "module_temp": [
            49.18,
            49.84,
            52.35,
            50.63,
            49.61,
            47.01,
            46.67,
            47.52,
            49.8,
            55.03,
            54.52,
            47.62,
            46.03,
            44.29,
            46.74,
        ],
        "irradiation": [
            0.51,
            0.79,
            0.65,
            0.76,
            0.42,
            0.49,
            0.57,
            0.56,
            0.74,
            0.89,
            0.47,
            0.54,
            0.4,
            0.45,
            0.57,
        ],
        "color": [
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "blue",
            "green",
            "green",
            "green",
            "green",
            "green",
        ],
        "irradiation_lag_15min": [
            np.nan,
            0.51,
            0.79,
            0.65,
            0.76,
            0.42,
            0.49,
            0.57,
            0.56,
            0.74,
            np.nan,
            0.47,
            0.54,
            0.4,
            0.45,
        ],
        "ambient_temp_lag_15min": [
            np.nan,
            31.31,
            31.51,
            32.15,
            32.39,
            32.62,
            32.5,
            32.52,
            32.68,
            33.76,
            np.nan,
            34.08,
            33.7,
            33.89,
            34.04,
        ],
        "irradiation_lag_30min": [
            np.nan,
            np.nan,
            0.51,
            0.79,
            0.65,
            0.76,
            0.42,
            0.49,
            0.57,
            0.56,
            np.nan,
            np.nan,
            0.47,
            0.54,
            0.4,
        ],
        "ambient_temp_lag_30min": [
            np.nan,
            np.nan,
            31.31,
            31.51,
            32.15,
            32.39,
            32.62,
            32.5,
            32.52,
            32.68,
            np.nan,
            np.nan,
            34.08,
            33.7,
            33.89,
        ],
    }
    expected_results_df = pd.DataFrame(
        data=expected_results,
        index=date_time,
    )
    # When setting group_by to color
    transformer = LagFeatures(
        variables=["irradiation", "ambient_temp"],
        freq=["15min", "30min"],
        group_by="color",
    )
    df_tr = transformer.fit_transform(df_time)
    assert df_tr.equals(expected_results_df)
