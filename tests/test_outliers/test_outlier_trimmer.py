# Authors: Soledad Galli <solegalli@gprotonmail.com>
# License: BSD 3 clause

import numpy as np
import pytest

from feature_engine.outliers import OutlierTrimmer
from tests.backend_helpers import make_series, frame_to_dict

# row 0 is an outlier in both variables, row 1 only in var_b, row 4 only in var_a
DATA_TWO_VARS = {"var_a": [1, 2, 3, 4, 100], "var_b": [1000, 6, 7, 8, 9]}


# init parameters
# the errors come from WinsorizerBase and are tested in test_base_outlier.py
@pytest.mark.parametrize(
    "capping_method, tail, fold, missing_values",
    [
        ("gaussian", "right", "auto", "raise"),
        ("iqr", "left", 2, "ignore"),
        ("mad", "both", 1.5, "raise"),
        ("quantiles", "both", 0.1, "ignore"),
    ],
)
def test_init_param_assignment(capping_method, tail, fold, missing_values):
    transformer = OutlierTrimmer(
        capping_method=capping_method,
        tail=tail,
        fold=fold,
        missing_values=missing_values,
    )
    assert transformer.capping_method == capping_method
    assert transformer.tail == tail
    assert transformer.fold == fold
    assert transformer.missing_values == missing_values


# fit and transform
def test_gaussian_right_tail_capping_when_fold_is_1(make_df, data_normal_dist):
    transformer = OutlierTrimmer(capping_method="gaussian", tail="right", fold=1)
    X = transformer.fit_transform(make_df(data_normal_dist))

    cap = transformer.right_tail_caps_["var"]
    expected = [v for v in data_normal_dist["var"] if v <= cap]

    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {"var": pytest.approx(expected)}
    assert X.shape[0] == 83


def test_gaussian_both_tails_capping_with_fold_2(make_df, data_normal_dist):
    transformer = OutlierTrimmer(capping_method="gaussian", tail="both", fold=2)
    X = transformer.fit_transform(make_df(data_normal_dist))

    lower = transformer.left_tail_caps_["var"]
    upper = transformer.right_tail_caps_["var"]
    expected = [v for v in data_normal_dist["var"] if lower <= v <= upper]

    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {"var": pytest.approx(expected)}
    assert X.shape[0] == 96


def test_iqr_left_tail_capping_with_fold_0_8(make_df, data_normal_dist):
    transformer = OutlierTrimmer(capping_method="iqr", tail="left", fold=0.8)
    X = transformer.fit_transform(make_df(data_normal_dist))

    lower = transformer.left_tail_caps_["var"]
    expected = [v for v in data_normal_dist["var"] if v >= lower]

    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {"var": pytest.approx(expected)}
    assert X.shape[0] == 98


def test_mad_right_tail_capping_with_fold_1(make_df, data_normal_dist):
    transformer = OutlierTrimmer(capping_method="mad", tail="right", fold=1)
    X = transformer.fit_transform(make_df(data_normal_dist))

    cap = transformer.right_tail_caps_["var"]
    expected = [v for v in data_normal_dist["var"] if v <= cap]

    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {"var": pytest.approx(expected)}
    assert X.shape[0] == 83


def test_transformer_ignores_na_in_df(make_df, data_na):
    transformer = OutlierTrimmer(
        capping_method="gaussian",
        tail="right",
        fold=1,
        variables=["Age"],
        missing_values="ignore",
    )
    X = transformer.fit_transform(make_df(data_na))

    assert transformer.right_tail_caps_["Age"] == pytest.approx(38.04494616731882)
    assert isinstance(X, make_df)
    # rows with missing values are removed too
    assert frame_to_dict(X)["Age"] == [20, 21, 19, 23, 37]


def test_rows_are_removed_if_any_variable_is_an_outlier(make_df):
    transformer = OutlierTrimmer(capping_method="quantiles", tail="both", fold=0.2)
    X = transformer.fit_transform(make_df(DATA_TWO_VARS))

    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {"var_a": [3, 4], "var_b": [7, 8]}


def test_transform_x_y(make_df, data_normal_dist):
    df = make_df(data_normal_dist)
    y = make_series(make_df, [0.0] * len(data_normal_dist["var"]))
    transformer = OutlierTrimmer(capping_method="mad", tail="right", fold=1)
    X = transformer.fit_transform(df)
    assert X.shape[0] != len(y)

    Xt, yt = transformer.transform_x_y(df, y)
    assert isinstance(Xt, make_df)
    assert isinstance(yt, type(y))
    assert frame_to_dict(Xt) == frame_to_dict(X)
    assert Xt.shape[0] == len(yt)
    assert Xt.shape[0] != len(data_normal_dist["var"])


@pytest.mark.parametrize(
    "capping_method, expected",
    [("gaussian", 3), ("iqr", 1.5), ("mad", 3.29), ("quantiles", 0.05)],
)
def test_auto_fold_default_value(capping_method, expected, make_df, data_normal_dist):
    transformer = OutlierTrimmer(capping_method=capping_method, fold="auto")
    transformer.fit(make_df(data_normal_dist))
    assert transformer.fold_ == expected


def test_variables_without_variation_are_left_untouched(make_df, data_normal_dist):
    data = {"var": [v // 10 for v in data_normal_dist["var"]]}
    transformer = OutlierTrimmer(capping_method="mad", tail="both")
    Xt = transformer.fit_transform(make_df(data))

    assert transformer.right_tail_caps_ == {"var": np.inf}
    assert transformer.left_tail_caps_ == {"var": -np.inf}
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == data
