import re

import numpy as np
import pandas as pd
import pytest

from feature_engine.timeseries.forecasting import LagFeatures
from tests.backend_helpers import frame_to_dict, make_series

DATA = {
    "ambient_temp": [31.31, 31.51, 32.15, 32.39, 32.62],
    "module_temp": [49.18, 49.84, 52.35, 50.63, 49.61],
    "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42],
    "color": ["blue"] * 5,
}

DATES = pd.date_range("2020-05-15 12:00:00", periods=5, freq="15min")


# init parameters
# the errors of the parameters from BaseForecastTransformer are tested in
# test_base_forecast_transformer.py
@pytest.mark.parametrize(
    "periods", [-1, 0, None, [-1, 2, 3], [0.1, 1], 0.5, [0, 1], "1", (1, 2)]
)
def test_error_if_periods_not_permitted(periods):
    msg = (
        "periods must be an integer or a list of positive integers. "
        f"Got {periods} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        LagFeatures(periods=periods)


def test_error_if_periods_duplicated():
    msg = "There are duplicated periods in the list: [1, 1, 2]"
    with pytest.raises(ValueError, match=re.escape(msg)):
        LagFeatures(periods=[1, 1, 2])


def test_error_if_freq_duplicated():
    msg = "There are duplicated freq values in the list: ['2h', '2h', '3h']"
    with pytest.raises(ValueError, match=re.escape(msg)):
        LagFeatures(freq=["2h", "2h", "3h"])


@pytest.mark.parametrize("sort_index", [-1, 1, None, "hola", [True]])
def test_error_if_sort_index_not_bool(sort_index):
    msg = f"sort_index takes values True and False. Got {sort_index} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        LagFeatures(sort_index=sort_index)


@pytest.mark.parametrize(
    "periods, freq, fill_value, sort_index, missing_values, drop_original, drop_na",
    [
        (1, None, None, True, "raise", False, False),
        ([1, 2, 3], None, 0, False, "ignore", True, True),
        (1, ["1h", "2h"], -1.5, True, "raise", False, True),
    ],
)
def test_init_param_assignment(
    periods, freq, fill_value, sort_index, missing_values, drop_original, drop_na
):
    transformer = LagFeatures(
        periods=periods,
        freq=freq,
        fill_value=fill_value,
        sort_index=sort_index,
        missing_values=missing_values,
        drop_original=drop_original,
        drop_na=drop_na,
    )
    assert transformer.periods == periods
    assert transformer.freq == freq
    assert transformer.fill_value == fill_value
    assert transformer.sort_index == sort_index
    assert transformer.missing_values == missing_values
    assert transformer.drop_original == drop_original
    assert transformer.drop_na == drop_na


# fit and transform
def test_lags_all_numerical_variables_by_default(make_df):
    X = make_df(DATA)
    Xt = LagFeatures().fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        **DATA,
        "ambient_temp_lag_1": [None, 31.31, 31.51, 32.15, 32.39],
        "module_temp_lag_1": [None, 49.18, 49.84, 52.35, 50.63],
        "irradiation_lag_1": [None, 0.51, 0.79, 0.65, 0.76],
    }


def test_lag_with_one_period(make_df):
    X = make_df(DATA)
    transformer = LagFeatures(variables=["ambient_temp", "module_temp"], periods=3)
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        **DATA,
        "ambient_temp_lag_3": [None, None, None, 31.31, 31.51],
        "module_temp_lag_3": [None, None, None, 49.18, 49.84],
    }


def test_lag_with_list_of_periods(make_df):
    X = make_df(DATA)
    transformer = LagFeatures(variables=["ambient_temp", "module_temp"], periods=[3, 2])
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        **DATA,
        "ambient_temp_lag_3": [None, None, None, 31.31, 31.51],
        "module_temp_lag_3": [None, None, None, 49.18, 49.84],
        "ambient_temp_lag_2": [None, None, 31.31, 31.51, 32.15],
        "module_temp_lag_2": [None, None, 49.18, 49.84, 52.35],
    }


def test_drop_original(make_df):
    X = make_df(DATA)
    transformer = LagFeatures(
        variables=["ambient_temp", "module_temp"], periods=[3, 2], drop_original=True
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42],
        "color": ["blue"] * 5,
        "ambient_temp_lag_3": [None, None, None, 31.31, 31.51],
        "module_temp_lag_3": [None, None, None, 49.18, 49.84],
        "ambient_temp_lag_2": [None, None, 31.31, 31.51, 32.15],
        "module_temp_lag_2": [None, None, 49.18, 49.84, 52.35],
    }


@pytest.mark.parametrize("fill_value", [-1, 0, 15.5])
def test_fill_value(make_df, fill_value):
    X = make_df(DATA)
    transformer = LagFeatures(
        variables=["ambient_temp", "module_temp"], periods=[3, 2], fill_value=fill_value
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        **DATA,
        "ambient_temp_lag_3": [fill_value, fill_value, fill_value, 31.31, 31.51],
        "module_temp_lag_3": [fill_value, fill_value, fill_value, 49.18, 49.84],
        "ambient_temp_lag_2": [fill_value, fill_value, 31.31, 31.51, 32.15],
        "module_temp_lag_2": [fill_value, fill_value, 49.18, 49.84, 52.35],
    }


def test_fill_value_does_not_fill_missing_data_of_the_variables(make_df):
    X = make_df({"a": [1.0, None, 3.0, 4.0], "b": [10, 20, 30, 40]})
    transformer = LagFeatures(periods=[1, 2], fill_value=0, missing_values="ignore")
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "a": [1.0, None, 3.0, 4.0],
        "b": [10, 20, 30, 40],
        "a_lag_1": [0, 1.0, None, 3.0],
        "b_lag_1": [0, 10, 20, 30],
        "a_lag_2": [0, 0, 1.0, None],
        "b_lag_2": [0, 0, 10, 20],
    }


def test_drop_na(make_df):
    X = make_df(DATA)
    transformer = LagFeatures(
        variables=["ambient_temp", "module_temp"], periods=[3, 2], drop_na=True
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "ambient_temp": [32.39, 32.62],
        "module_temp": [50.63, 49.61],
        "irradiation": [0.76, 0.42],
        "color": ["blue"] * 2,
        "ambient_temp_lag_3": [31.31, 31.51],
        "module_temp_lag_3": [49.18, 49.84],
        "ambient_temp_lag_2": [31.51, 32.15],
        "module_temp_lag_2": [49.84, 52.35],
    }


def test_transform_x_y(make_df):
    X = make_df(DATA)
    y = make_series(make_df, [1, 2, 3, 4, 5], name="y")
    transformer = LagFeatures(variables="ambient_temp", periods=2, drop_na=True)
    transformer.fit(X)

    Xt, yt = transformer.transform_x_y(X, y)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "ambient_temp": [32.15, 32.39, 32.62],
        "module_temp": [52.35, 50.63, 49.61],
        "irradiation": [0.65, 0.76, 0.42],
        "color": ["blue"] * 3,
        "ambient_temp_lag_2": [31.31, 31.51, 32.15],
    }
    assert list(yt) == [3, 4, 5]


def test_return_empty_without_numerical_variables(make_df):
    X = make_df({"color": DATA["color"]})
    transformer = LagFeatures(return_empty=True, periods=[1, 2]).fit(X)
    Xt = transformer.transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"color": DATA["color"]}
    assert transformer.get_feature_names_out() == ["color"]


@pytest.mark.parametrize(
    "periods, drop_original, expected",
    [
        (
            2,
            False,
            [
                "ambient_temp",
                "module_temp",
                "irradiation",
                "color",
                "ambient_temp_lag_2",
                "module_temp_lag_2",
                "irradiation_lag_2",
            ],
        ),
        (
            [2, 3],
            True,
            [
                "color",
                "ambient_temp_lag_2",
                "module_temp_lag_2",
                "irradiation_lag_2",
                "ambient_temp_lag_3",
                "module_temp_lag_3",
                "irradiation_lag_3",
            ],
        ),
    ],
)
def test_get_feature_names_out(make_df, periods, drop_original, expected):
    X = make_df(DATA)
    transformer = LagFeatures(periods=periods, drop_original=drop_original).fit(X)

    assert transformer.get_feature_names_out() == expected
    assert transformer.get_feature_names_out(input_features=list(DATA)) == expected
    assert list(transformer.transform(X).columns) == expected


def test_get_feature_names_out_raises_error_if_input_features_wrong(make_df):
    transformer = LagFeatures().fit(make_df(DATA))

    msg = "input_features must be a list or an array. Got color instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.get_feature_names_out(input_features="color")

    msg = "input_features is not equal to feature_names_in_"
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.get_feature_names_out(input_features=["color"])


# pandas: freq lags the rows based on the DatetimeIndex
def test_lag_with_freq_with_pandas(df_time):
    transformer = LagFeatures(freq=["1h", "15min"])
    Xt = transformer.fit_transform(df_time)

    expected = pd.DataFrame(
        {
            **DATA,
            "ambient_temp_lag_1h": [np.nan, np.nan, np.nan, np.nan, 31.31],
            "module_temp_lag_1h": [np.nan, np.nan, np.nan, np.nan, 49.18],
            "irradiation_lag_1h": [np.nan, np.nan, np.nan, np.nan, 0.51],
            "ambient_temp_lag_15min": [np.nan, 31.31, 31.51, 32.15, 32.39],
            "module_temp_lag_15min": [np.nan, 49.18, 49.84, 52.35, 50.63],
            "irradiation_lag_15min": [np.nan, 0.51, 0.79, 0.65, 0.76],
        },
        index=DATES,
    )
    pd.testing.assert_frame_equal(Xt.head(5), expected, check_freq=False)
    assert len(Xt) == len(df_time)


def test_lag_with_freq_on_irregular_index_with_pandas():
    # rows with no data one hour before get missing values
    index = pd.to_datetime(
        ["2020-05-15 12:00", "2020-05-15 13:00", "2020-05-15 13:30", "2020-05-15 15:00"]
    )
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=index)
    Xt = LagFeatures(freq="1h", drop_original=True).fit_transform(X)

    expected = pd.DataFrame({"a_lag_1h": [np.nan, 1.0, np.nan, np.nan]}, index=index)
    pd.testing.assert_frame_equal(Xt, expected)


@pytest.mark.parametrize("fill_value", [-1, 0, 15])
def test_fill_value_with_freq_with_pandas(df_time, fill_value):
    transformer = LagFeatures(
        variables=["ambient_temp", "module_temp"],
        freq=["45min", "30min"],
        fill_value=fill_value,
        drop_original=True,
    )
    Xt = transformer.fit_transform(df_time)

    expected = pd.DataFrame(
        {
            "irradiation": [0.51, 0.79, 0.65, 0.76, 0.42],
            "color": ["blue"] * 5,
            "ambient_temp_lag_45min": [fill_value] * 3 + [31.31, 31.51],
            "module_temp_lag_45min": [fill_value] * 3 + [49.18, 49.84],
            "ambient_temp_lag_30min": [fill_value] * 2 + [31.31, 31.51, 32.15],
            "module_temp_lag_30min": [fill_value] * 2 + [49.18, 49.84, 52.35],
        },
        index=DATES,
    )
    pd.testing.assert_frame_equal(Xt.head(5), expected, check_freq=False)


def test_get_feature_names_out_with_freq_with_pandas(df_time):
    transformer = LagFeatures(freq=["3D", "2D"], drop_original=True).fit(df_time)

    assert transformer.get_feature_names_out() == [
        "color",
        "ambient_temp_lag_3D",
        "module_temp_lag_3D",
        "irradiation_lag_3D",
        "ambient_temp_lag_2D",
        "module_temp_lag_2D",
        "irradiation_lag_2D",
    ]


@pytest.mark.parametrize(
    "sort_index, expected_index, expected_lag",
    [
        (True, DATES, [np.nan, 31.31, 31.51, 32.15, 32.39]),
        (False, DATES[[2, 0, 4, 1, 3]], [np.nan, 32.15, 31.31, 32.62, 31.51]),
    ],
)
def test_sort_index_with_pandas(sort_index, expected_index, expected_lag):
    X = pd.DataFrame(DATA, index=DATES).iloc[[2, 0, 4, 1, 3]]
    transformer = LagFeatures(variables="ambient_temp", sort_index=sort_index)
    Xt = transformer.fit_transform(X)

    expected = pd.DataFrame(DATA, index=DATES).loc[expected_index]
    expected["ambient_temp_lag_1"] = expected_lag
    pd.testing.assert_frame_equal(Xt, expected, check_freq=False)


def test_integer_column_names_with_pandas():
    X = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [4, 5, 6], "c": ["x", "y", "z"]})
    transformer = LagFeatures(periods=[1, 2], fill_value=0, drop_original=True)
    Xt = transformer.fit(X).transform(X[["c", 1, 0]])

    expected = pd.DataFrame(
        {
            "c": ["x", "y", "z"],
            "0_lag_1": [0.0, 1.0, 2.0],
            "1_lag_1": [0, 4, 5],
            "0_lag_2": [0.0, 0.0, 1.0],
            "1_lag_2": [0, 0, 4],
        }
    )
    # the columns started as a mix of integers and strings
    expected.columns = expected.columns.astype(object)
    pd.testing.assert_frame_equal(Xt, expected)
