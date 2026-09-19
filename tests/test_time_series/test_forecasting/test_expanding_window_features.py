import re

import numpy as np
import pandas as pd
import polars as pl
import pytest

from feature_engine.timeseries.forecasting import ExpandingWindowFeatures
from tests.backend_helpers import frame_to_dict, make_series

DATA = {
    "x1": [3, 1, 4, 1, 5, 9],
    "x2": [2.0, 7.0, 1.0, 8.0, 2.0, 8.0],
    "color": ["a", "b", "c", "d", "e", "f"],
}

DATES = pd.date_range("2020-05-15 12:00:00", periods=6, freq="15min")


# init parameters
# the errors of the parameters from BaseForecastTransformer are tested in
# test_base_forecast_transformer.py
@pytest.mark.parametrize(
    "functions", [[np.min, np.max], np.min, 1, None, [], "", ["mean", 1], ("mean",)]
)
def test_error_if_functions_not_permitted(functions):
    msg = f"functions must be a list of strings or a string. Got {functions} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        ExpandingWindowFeatures(functions=functions)


def test_error_if_functions_duplicated():
    msg = "There are duplicated functions in the list: ['sum', 'mean', 'sum']"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ExpandingWindowFeatures(functions=["sum", "mean", "sum"])


@pytest.mark.parametrize("periods", [-1, 1.5, "1", None, [1]])
def test_error_if_periods_not_permitted(periods):
    msg = f"periods must be a non-negative integer. Got {periods} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        ExpandingWindowFeatures(periods=periods)


@pytest.mark.parametrize(
    "min_periods, functions, periods, freq, sort_index, missing_values, "
    "drop_original, drop_na",
    [
        (None, "mean", 1, None, True, "raise", False, False),
        (3, ["sum", "std"], 0, "15min", False, "ignore", True, True),
        (1, ["max"], 2, "1h", True, "raise", False, True),
    ],
)
def test_init_param_assignment(
    min_periods,
    functions,
    periods,
    freq,
    sort_index,
    missing_values,
    drop_original,
    drop_na,
):
    transformer = ExpandingWindowFeatures(
        min_periods=min_periods,
        functions=functions,
        periods=periods,
        freq=freq,
        sort_index=sort_index,
        missing_values=missing_values,
        drop_original=drop_original,
        drop_na=drop_na,
    )
    assert transformer.min_periods == min_periods
    assert transformer.functions == functions
    assert transformer.periods == periods
    assert transformer.freq == freq
    assert transformer.sort_index == sort_index
    assert transformer.missing_values == missing_values
    assert transformer.drop_original == drop_original
    assert transformer.drop_na == drop_na


# fit and transform
@pytest.mark.parametrize(
    "variables, functions, expected",
    [
        (None, "mean", ["x1_expanding_mean", "x2_expanding_mean"]),
        (
            None,
            ["mean", "sum"],
            [
                "x1_expanding_mean",
                "x1_expanding_sum",
                "x2_expanding_mean",
                "x2_expanding_sum",
            ],
        ),
        (
            "x2",
            ["sum", "mean", "count"],
            ["x2_expanding_sum", "x2_expanding_mean", "x2_expanding_count"],
        ),
    ],
)
def test_get_feature_names_out(make_df, variables, functions, expected):
    transformer = ExpandingWindowFeatures(variables=variables, functions=functions)
    transformer.fit(make_df(DATA))

    assert transformer.get_feature_names_out() == ["x1", "x2", "color"] + expected
    assert (
        transformer.get_feature_names_out(["x1", "x2", "color"])
        == [
            "x1",
            "x2",
            "color",
        ]
        + expected
    )


def test_get_feature_names_out_raises_error_if_input_features_not_list(make_df):
    transformer = ExpandingWindowFeatures().fit(make_df(DATA))
    msg = "input_features must be a list or an array. Got x1 instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.get_feature_names_out(input_features="x1")


def test_get_feature_names_out_raises_error_if_input_features_not_in_fit(make_df):
    transformer = ExpandingWindowFeatures().fit(make_df(DATA))
    msg = "input_features is not equal to feature_names_in_"
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.get_feature_names_out(input_features=["color"])


def test_default_expanding_mean_of_all_numerical_variables(make_df):
    X = make_df(DATA)
    Xt = ExpandingWindowFeatures().fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        **DATA,
        "x1_expanding_mean": pytest.approx(
            [None, 3.0, 2.0, 2.6666666666666665, 2.25, 2.8]
        ),
        "x2_expanding_mean": pytest.approx(
            [None, 2.0, 4.5, 3.3333333333333335, 4.5, 4.0]
        ),
    }


def test_all_functions_supported_with_polars(make_df):
    functions = ["count", "sum", "mean", "median", "min", "max", "std", "var"]
    X = make_df(DATA)
    Xt = ExpandingWindowFeatures(variables="x2", functions=functions).fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        **DATA,
        "x2_expanding_count": [None, 1.0, 2.0, 3.0, 4.0, 5.0],
        "x2_expanding_sum": [None, 2.0, 9.0, 10.0, 18.0, 20.0],
        "x2_expanding_mean": pytest.approx(
            [None, 2.0, 4.5, 3.3333333333333335, 4.5, 4.0]
        ),
        "x2_expanding_median": [None, 2.0, 4.5, 2.0, 4.5, 2.0],
        "x2_expanding_min": [None, 2.0, 2.0, 1.0, 1.0, 1.0],
        "x2_expanding_max": [None, 2.0, 7.0, 7.0, 8.0, 8.0],
        "x2_expanding_std": pytest.approx(
            [
                None,
                None,
                3.5355339059327378,
                3.214550253664318,
                3.5118845842842465,
                3.24037034920393,
            ]
        ),
        "x2_expanding_var": pytest.approx(
            [None, None, 12.5, 10.333333333333332, 12.333333333333334, 10.5]
        ),
    }


@pytest.mark.parametrize(
    "periods, expected",
    [
        (0, [2.0, 9.0, 10.0, 18.0, 20.0, 28.0]),
        (1, [None, 2.0, 9.0, 10.0, 18.0, 20.0]),
        (3, [None, None, None, 2.0, 9.0, 10.0]),
    ],
)
def test_periods(make_df, periods, expected):
    X = make_df(DATA)
    transformer = ExpandingWindowFeatures(
        variables="x2", functions="sum", periods=periods
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt)["x2_expanding_sum"] == expected


# NaN and inf are missing data for the statistics, but count includes inf.
DATA_NA = {"x": [None, 1.0, None, 4.0, np.inf, 9.0]}


@pytest.mark.parametrize(
    "min_periods, expected",
    [
        (
            None,
            {
                "x_expanding_count": [None, 0.0, 1.0, 1.0, 2.0, 3.0],
                "x_expanding_sum": [None, 0.0, 1.0, 1.0, 5.0, 5.0],
                "x_expanding_mean": [None, None, 1.0, 1.0, 2.5, 2.5],
                "x_expanding_min": [None, None, 1.0, 1.0, 1.0, 1.0],
                "x_expanding_std": pytest.approx(
                    [None, None, None, None, 2.1213203435596424, 2.1213203435596424]
                ),
            },
        ),
        (
            2,
            {
                "x_expanding_count": [None, None, 1.0, 1.0, 2.0, 3.0],
                "x_expanding_sum": [None, None, None, None, 5.0, 5.0],
                "x_expanding_mean": [None, None, None, None, 2.5, 2.5],
                "x_expanding_min": [None, None, None, None, 1.0, 1.0],
                "x_expanding_std": pytest.approx(
                    [None, None, None, None, 2.1213203435596424, 2.1213203435596424]
                ),
            },
        ),
    ],
)
def test_min_periods_and_missing_data(make_df, min_periods, expected):
    X = make_df(DATA_NA)
    transformer = ExpandingWindowFeatures(
        functions=["count", "sum", "mean", "min", "std"],
        min_periods=min_periods,
        missing_values="ignore",
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {**DATA_NA, **expected}


def test_nan_is_missing_data(make_df):
    # polars keeps NaN apart from null, pandas treats both as missing data.
    X = make_df({"x": [1.0, np.nan, 3.0, 5.0]})
    transformer = ExpandingWindowFeatures(
        functions=["count", "mean", "max"], missing_values="ignore"
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "x": [1.0, None, 3.0, 5.0],
        "x_expanding_count": [None, 1.0, 1.0, 2.0],
        "x_expanding_mean": [None, 1.0, 1.0, 2.0],
        "x_expanding_max": [None, 1.0, 1.0, 3.0],
    }


def test_drop_original(make_df):
    X = make_df(DATA)
    transformer = ExpandingWindowFeatures(functions=["min", "max"], drop_original=True)
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "color": DATA["color"],
        "x1_expanding_min": [None, 3.0, 1.0, 1.0, 1.0, 1.0],
        "x1_expanding_max": [None, 3.0, 3.0, 4.0, 4.0, 5.0],
        "x2_expanding_min": [None, 2.0, 2.0, 1.0, 1.0, 1.0],
        "x2_expanding_max": [None, 2.0, 7.0, 7.0, 8.0, 8.0],
    }


def test_drop_na(make_df):
    X = make_df(DATA)
    transformer = ExpandingWindowFeatures(
        variables="x1", functions=["sum", "std"], drop_na=True
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "x1": [4, 1, 5, 9],
        "x2": [1.0, 8.0, 2.0, 8.0],
        "color": ["c", "d", "e", "f"],
        "x1_expanding_sum": [4.0, 8.0, 9.0, 14.0],
        "x1_expanding_std": pytest.approx(
            [1.4142135623730951, 1.5275252316519468, 1.5, 1.7888543819998317]
        ),
    }


@pytest.mark.parametrize(
    "drop_na, expected", [(True, [3, 4, 5, 6]), (False, [1, 2, 3, 4, 5, 6])]
)
def test_transform_x_y(make_df, drop_na, expected):
    X = make_df(DATA)
    y = make_series(make_df, [1, 2, 3, 4, 5, 6], name="y")
    transformer = ExpandingWindowFeatures(
        variables="x1", functions=["sum", "std"], drop_na=drop_na
    ).fit(X)

    Xt, yt = transformer.transform_x_y(X, y)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == frame_to_dict(transformer.transform(X))
    assert list(yt) == expected


@pytest.mark.parametrize("y", [[1, 2, 3, 4, 5, 6], np.array([1, 2, 3, 4, 5, 6])])
def test_fit_accepts_y_as_list_and_array(make_df, y):
    X = make_df(DATA)
    Xt = ExpandingWindowFeatures(variables="x1").fit(X, y).transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt)["x1_expanding_mean"] == pytest.approx(
        [None, 3.0, 2.0, 2.6666666666666665, 2.25, 2.8]
    )


def test_no_numerical_variables_with_return_empty(make_df):
    X = make_df({"color": DATA["color"]})
    Xt = ExpandingWindowFeatures(return_empty=True).fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"color": DATA["color"]}


# pandas: the index orders the rows in time, and freq shifts the index
@pytest.mark.parametrize(
    "periods, freq",
    [(1, "30min"), (2, "15min"), (2, None)],
)
def test_freq_with_pandas(periods, freq):
    X = pd.DataFrame(DATA, index=DATES)
    transformer = ExpandingWindowFeatures(
        variables="x2", functions="sum", periods=periods, freq=freq
    )
    Xt = transformer.fit_transform(X)

    expected = X.assign(x2_expanding_sum=[np.nan, np.nan, 2.0, 9.0, 10.0, 18.0])
    pd.testing.assert_frame_equal(Xt, expected)


def test_sort_index_with_pandas():
    X = pd.DataFrame(DATA, index=DATES).iloc[[3, 0, 5, 1, 4, 2]]

    Xt = ExpandingWindowFeatures(variables="x2", functions="max").fit_transform(X)
    expected = X.sort_index().assign(x2_expanding_max=[np.nan, 2.0, 7.0, 7.0, 8.0, 8.0])
    pd.testing.assert_frame_equal(Xt, expected)

    transformer = ExpandingWindowFeatures(
        variables="x2", functions="max", sort_index=False
    )
    Xt = transformer.fit_transform(X)
    expected = X.assign(x2_expanding_max=[np.nan, 8.0, 8.0, 8.0, 8.0, 8.0])
    pd.testing.assert_frame_equal(Xt, expected)


def test_functions_not_supported_by_polars_with_pandas():
    X = pd.DataFrame(DATA)
    transformer = ExpandingWindowFeatures(variables="x2", functions=["skew", "sem"])
    Xt = transformer.fit_transform(X)

    expected = X.assign(
        x2_expanding_skew=[
            np.nan,
            np.nan,
            np.nan,
            1.545392525695021,
            0.0,
            0.5878222855698739,
        ],
        x2_expanding_sem=[
            np.nan,
            np.nan,
            2.5,
            1.855921454276674,
            1.7559422921421233,
            1.4491376746189437,
        ],
    )
    pd.testing.assert_frame_equal(Xt, expected)


def test_integer_column_names_with_pandas():
    X = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [4, 5, 6], "c": ["x", "y", "z"]})
    transformer = ExpandingWindowFeatures(
        functions=["mean", "max"], drop_original=True, drop_na=True
    ).fit(X)
    Xt = transformer.transform(X[["c", 1, 0]])

    expected = pd.DataFrame(
        {
            "c": ["y", "z"],
            "0_expanding_mean": [1.0, 1.5],
            "0_expanding_max": [1.0, 2.0],
            "1_expanding_mean": [4.0, 4.5],
            "1_expanding_max": [4.0, 5.0],
        },
        index=[1, 2],
    )
    # the columns started as a mix of integers and strings
    expected.columns = expected.columns.astype(object)
    pd.testing.assert_frame_equal(Xt, expected)


# polars: the rows are used in the order given
def test_rows_are_used_in_the_order_given_with_polars():
    X = pl.DataFrame({"x": [8.0, 2.0, 7.0, 1.0]})
    Xt = ExpandingWindowFeatures(functions="min", sort_index=True).fit_transform(X)

    assert frame_to_dict(Xt)["x_expanding_min"] == [None, 8.0, 2.0, 2.0]


def test_error_if_function_not_supported_with_polars():
    X = pl.DataFrame(DATA)
    transformer = ExpandingWindowFeatures(functions=["mean", "skew", "sem"]).fit(X)
    msg = (
        "With dataframes other than pandas, ExpandingWindowFeatures supports the "
        "functions ['count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var']. "
        "Got ['skew', 'sem'] instead."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        transformer.transform(X)
