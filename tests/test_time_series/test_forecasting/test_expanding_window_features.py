import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from feature_engine.timeseries.forecasting import ExpandingWindowFeatures


def test_get_feature_names_out_raises_when_input_features_is_string(df_time):

    tr = ExpandingWindowFeatures(functions=["mean", "sum"])
    tr.fit(df_time)

    with pytest.raises(ValueError):
        # get error when user does not pass a list
        tr.get_feature_names_out(input_features="ambient_temp")


def test_get_feature_names_out_raises_when_input_features_not_transformed(df_time):

    tr = ExpandingWindowFeatures(functions=["mean", "sum"])
    tr.fit(df_time)

    with pytest.raises(ValueError):
        # assert error when uses passes features that were not transformed
        tr.get_feature_names_out(input_features=["color"])


@pytest.mark.parametrize("_periods", [1, 2, 3])
def test_permitted_param_periods(_periods):
    transformer = ExpandingWindowFeatures(periods=_periods)
    assert transformer.periods == _periods


@pytest.mark.parametrize("_drop_or", [-1, [0], None, 7, "hola"])
def test_error_when_non_permitted_param_drop_original(_drop_or):
    with pytest.raises(ValueError):
        ExpandingWindowFeatures(drop_original=_drop_or)


@pytest.mark.parametrize("_drop_na", [-1, [0], None, 7, "hola"])
def test_error_when_non_permitted_param_drop_na(_drop_na):
    with pytest.raises(ValueError):
        ExpandingWindowFeatures(drop_na=_drop_na)


def test_get_feature_names_out_multiple_variables_and_functions(df_time):
    # input features
    original_features = ["ambient_temp", "module_temp", "irradiation", "color"]

    tr = ExpandingWindowFeatures(functions=["mean", "sum"])
    tr.fit(df_time)

    # expected
    output = [
        "ambient_temp_expanding_mean",
        "ambient_temp_expanding_sum",
        "module_temp_expanding_mean",
        "module_temp_expanding_sum",
        "irradiation_expanding_mean",
        "irradiation_expanding_sum",
    ]
    output = original_features + output
    assert tr.get_feature_names_out(input_features=None) == output
    assert tr.get_feature_names_out(input_features=original_features) == output


def test_get_feature_names_out_single_variable_and_multiple_functions(df_time):
    # input features
    original_features = ["ambient_temp", "module_temp", "irradiation", "color"]

    tr = ExpandingWindowFeatures(
        variables="ambient_temp", functions=["sum", "mean", "count"]
    )
    tr.fit(df_time)

    # expected
    output = [
        "ambient_temp_expanding_sum",
        "ambient_temp_expanding_mean",
        "ambient_temp_expanding_count",
    ]
    output = original_features + output
    assert tr.get_feature_names_out(input_features=None) == output


def test_get_feature_names_out_single_variable_and_single_function(df_time):
    # input features
    original_features = ["ambient_temp", "module_temp", "irradiation", "color"]

    tr = ExpandingWindowFeatures(variables="ambient_temp", functions="sum")
    tr.fit(df_time)

    # expected
    output = [
        "ambient_temp_expanding_sum",
    ]
    output = original_features + output
    assert tr.get_feature_names_out(input_features=None) == output


def test_expanding_sum_single_var(df_time):
    expected_results = {
        "ambient_temp_expanding_sum": [
            np.nan,
            31.31,
            62.82,
            94.97,
            127.36,
            159.98,
            192.48,
            225.00,
            257.68,
            291.44,
            325.57,
            359.65,
            393.35,
            427.24,
            461.28,
        ],
    }
    expected_df = df_time.copy()
    expected_df["ambient_temp_expanding_sum"] = expected_results[
        "ambient_temp_expanding_sum"
    ]

    transformer = ExpandingWindowFeatures(variables=["ambient_temp"], functions="sum")
    df_tr = transformer.fit_transform(df_time)
    assert_frame_equal(df_tr, expected_df)


def test_expanding_sum_multiple_vars(df_time):
    expected_results = {
        "ambient_temp_expanding_sum": [
            np.nan,
            31.31,
            62.82,
            94.97,
            127.36,
            159.98,
            192.48,
            225.00,
            257.68,
            291.44,
            325.57,
            359.65,
            393.35,
            427.24,
            461.28,
        ],
        "irradiation_expanding_sum": [
            np.nan,
            0.51,
            1.3,
            1.95,
            2.71,
            3.13,
            3.62,
            4.19,
            4.75,
            5.49,
            6.38,
            6.85,
            7.39,
            7.79,
            8.24,
        ],
    }
    expected_df = df_time.copy()
    expected_df["ambient_temp_expanding_sum"] = expected_results[
        "ambient_temp_expanding_sum"
    ]
    expected_df["irradiation_expanding_sum"] = expected_results[
        "irradiation_expanding_sum"
    ]

    transformer = ExpandingWindowFeatures(
        variables=["ambient_temp", "irradiation"], functions="sum"
    )
    df_tr = transformer.fit_transform(df_time)
    assert_frame_equal(df_tr, expected_df)


def test_expanding_sum_and_mean_single_var(df_time):
    expected_results = {
        "ambient_temp_expanding_sum": [
            np.nan,
            31.31,
            62.82,
            94.97,
            127.36,
            159.98,
            192.48,
            225.00,
            257.68,
            291.44,
            325.57,
            359.65,
            393.35,
            427.24,
            461.28,
        ],
        "ambient_temp_expanding_mean": [
            np.nan,
            31.3100,
            31.4100,
            31.6567,
            31.8400,
            31.9960,
            32.0800,
            32.1429,
            32.2100,
            32.3822,
            32.5570,
            32.6955,
            32.7792,
            32.8646,
            32.9486,
        ],
    }
    expected_df = df_time.copy()
    expected_df["ambient_temp_expanding_sum"] = expected_results[
        "ambient_temp_expanding_sum"
    ]
    expected_df["ambient_temp_expanding_mean"] = expected_results[
        "ambient_temp_expanding_mean"
    ]

    transformer = ExpandingWindowFeatures(
        variables=["ambient_temp"], functions=["sum", "mean"]
    )
    df_tr = transformer.fit_transform(df_time)
    assert_frame_equal(df_tr, expected_df)


def test_expanding_sum_and_mean_multiple_vars(df_time):
    expected_results = {
        "ambient_temp_expanding_sum": [
            np.nan,
            31.31,
            62.82,
            94.97,
            127.36,
            159.98,
            192.48,
            225.00,
            257.68,
            291.44,
            325.57,
            359.65,
            393.35,
            427.24,
            461.28,
        ],
        "ambient_temp_expanding_mean": [
            np.nan,
            31.3100,
            31.4100,
            31.6567,
            31.8400,
            31.9960,
            32.0800,
            32.1429,
            32.2100,
            32.3822,
            32.5570,
            32.6955,
            32.7792,
            32.8646,
            32.9486,
        ],
        "irradiation_expanding_sum": [
            np.nan,
            0.51,
            1.3,
            1.95,
            2.71,
            3.13,
            3.62,
            4.19,
            4.75,
            5.49,
            6.38,
            6.85,
            7.39,
            7.79,
            8.24,
        ],
        "irradiation_expanding_mean": [
            np.nan,
            0.51000,
            0.65000,
            0.65000,
            0.67750,
            0.62600,
            0.60333,
            0.59857,
            0.59375,
            0.61000,
            0.63800,
            0.62273,
            0.61583,
            0.59923,
            0.58857,
        ],
    }
    expected_df = df_time.copy()
    expected_df["ambient_temp_expanding_sum"] = expected_results[
        "ambient_temp_expanding_sum"
    ]
    expected_df["ambient_temp_expanding_mean"] = expected_results[
        "ambient_temp_expanding_mean"
    ]
    expected_df["irradiation_expanding_sum"] = expected_results[
        "irradiation_expanding_sum"
    ]
    expected_df["irradiation_expanding_mean"] = expected_results[
        "irradiation_expanding_mean"
    ]

    transformer = ExpandingWindowFeatures(
        variables=["ambient_temp", "irradiation"], functions=["sum", "mean"]
    )
    df_tr = transformer.fit_transform(df_time)
    assert_frame_equal(df_tr, expected_df)


def test_expanding_sum_single_var_periods(df_time):
    expected_results = {
        "ambient_temp_expanding_sum": [
            np.nan,
            np.nan,
            31.31,
            62.82,
            94.97,
            127.36,
            159.98,
            192.48,
            225.00,
            257.68,
            291.44,
            325.57,
            359.65,
            393.35,
            427.24,
        ],
    }
    expected_df = df_time.copy()
    expected_df["ambient_temp_expanding_sum"] = expected_results[
        "ambient_temp_expanding_sum"
    ]

    transformer = ExpandingWindowFeatures(
        variables=["ambient_temp"], functions="sum", periods=2
    )
    df_tr = transformer.fit_transform(df_time)
    assert_frame_equal(df_tr, expected_df)


def test_expanding_sum_single_var_freqs(df_time):
    expected_results = {
        "ambient_temp_expanding_sum": [
            np.nan,
            np.nan,
            31.31,
            62.82,
            94.97,
            127.36,
            159.98,
            192.48,
            225.00,
            257.68,
            291.44,
            325.57,
            359.65,
            393.35,
            427.24,
        ],
    }
    expected_df = df_time.copy()
    expected_df["ambient_temp_expanding_sum"] = expected_results[
        "ambient_temp_expanding_sum"
    ]

    transformer = ExpandingWindowFeatures(
        variables=["ambient_temp"], functions="sum", freq="30min"
    )
    df_tr = transformer.fit_transform(df_time)
    assert_frame_equal(df_tr, expected_df)


def test_expanding_sum_single_var_periods_and_freqs(df_time):
    expected_results = {
        "ambient_temp_expanding_sum": [
            np.nan,
            np.nan,
            31.31,
            62.82,
            94.97,
            127.36,
            159.98,
            192.48,
            225.00,
            257.68,
            291.44,
            325.57,
            359.65,
            393.35,
            427.24,
        ],
    }
    expected_df = df_time.copy()
    expected_df["ambient_temp_expanding_sum"] = expected_results[
        "ambient_temp_expanding_sum"
    ]

    transformer = ExpandingWindowFeatures(
        variables=["ambient_temp"], functions="sum", periods=2, freq="15min"
    )
    df_tr = transformer.fit_transform(df_time)
    assert_frame_equal(df_tr, expected_df)


def test_sort_index(df_time):
    # Shuffle dataframe
    Xs = df_time.sample(frac=1)

    transformer = ExpandingWindowFeatures(sort_index=False)
    df_tr = transformer.fit_transform(Xs)
    assert_frame_equal(df_tr[transformer.variables_], Xs[transformer.variables_])

    transformer = ExpandingWindowFeatures(sort_index=True)
    df_tr = transformer.fit_transform(Xs)
    assert_frame_equal(
        df_tr[transformer.variables_], Xs[transformer.variables_].sort_index()
    )


def test_expanding_window_raises_when_periods_negative():
    with pytest.raises(
        ValueError, match="periods must be a non-negative integer. Got -1 instead."
    ):
        ExpandingWindowFeatures(periods=-1)


def test_drop_na(df_time):
    expected_results = {
        "ambient_temp_expanding_sum": [
            31.31,
            62.82,
            94.97,
            127.36,
            159.98,
            192.48,
            225.00,
            257.68,
            291.44,
            325.57,
            359.65,
            393.35,
            427.24,
            461.28,
        ],
        "irradiation_expanding_sum": [
            0.51,
            1.3,
            1.95,
            2.71,
            3.13,
            3.62,
            4.19,
            4.75,
            5.49,
            6.38,
            6.85,
            7.39,
            7.79,
            8.24,
        ],
    }
    expected_df = df_time.iloc[1:].copy()
    expected_df["ambient_temp_expanding_sum"] = expected_results[
        "ambient_temp_expanding_sum"
    ]
    expected_df["irradiation_expanding_sum"] = expected_results[
        "irradiation_expanding_sum"
    ]

    transformer = ExpandingWindowFeatures(
        variables=["ambient_temp", "irradiation"],
        functions="sum",
        drop_na=True,
    )
    df_tr = transformer.fit_transform(df_time)
    assert_frame_equal(df_tr, expected_df)


def test_transform_x_y(df_time):
    y = pd.Series(np.zeros(len(df_time)), index=df_time.index)
    expected_results = {
        "ambient_temp_expanding_sum": [
            31.31,
            62.82,
            94.97,
            127.36,
            159.98,
            192.48,
            225.00,
            257.68,
            291.44,
            325.57,
            359.65,
            393.35,
            427.24,
            461.28,
        ],
        "irradiation_expanding_sum": [
            0.51,
            1.3,
            1.95,
            2.71,
            3.13,
            3.62,
            4.19,
            4.75,
            5.49,
            6.38,
            6.85,
            7.39,
            7.79,
            8.24,
        ],
    }
    expected_df = df_time.iloc[1:].copy()
    expected_df["ambient_temp_expanding_sum"] = expected_results[
        "ambient_temp_expanding_sum"
    ]
    expected_df["irradiation_expanding_sum"] = expected_results[
        "irradiation_expanding_sum"
    ]

    transformer = ExpandingWindowFeatures(
        variables=["ambient_temp", "irradiation"],
        functions="sum",
        drop_na=True,
    )
    transformer.fit(df_time)

    Xt, yt = transformer.transform_x_y(df_time, y)
    assert len(Xt) == len(yt)
    assert len(y) != len(yt)
    assert (Xt.index == yt.index).all()


def test_error_duplicate_functions(df_time):
    msg = "There are duplicated functions in the list: ['sum', 'sum']"
    with pytest.raises(ValueError) as record:
        ExpandingWindowFeatures(
            variables=["ambient_temp"], functions=["sum", "sum"], periods=2, freq="15T"
        )

    # check that error message matches
    assert str(record.value) == msg


@pytest.mark.parametrize("functions", [[np.min, np.max], np.min])
def test_error_native_functions(df_time, functions):
    msg = "functions must be a list of strings or a string." f"Got {functions} instead."
    with pytest.raises(ValueError) as record:
        ExpandingWindowFeatures(
            variables=["ambient_temp"],
            functions=functions,
            periods=2,
            freq="15T",
        )

    # check that error message matches
    assert str(record.value) == msg
