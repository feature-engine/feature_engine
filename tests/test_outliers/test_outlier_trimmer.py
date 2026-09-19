# Authors: Soledad Galli <solegalli@gprotonmail.com>
# License: BSD 3 clause

import numpy as np
import pytest

from feature_engine.outliers import OutlierTrimmer
from tests.backend_helpers import make_series, frame_to_dict

# var_a and var_b each push a different row past their own bounds (row 0 fails
# both, row 1 fails only var_b, row 4 fails only var_a) - exercises that the
# combined filter() keeps a row only when every variable's condition holds.
DATA_TWO_VARS = {"var_a": [1, 2, 3, 4, 100], "var_b": [1000, 6, 7, 8, 9]}


def test_gaussian_right_tail_capping_when_fold_is_1(make_df, data_normal_dist):
    # test case 1: mean and std, right tail
    transformer = OutlierTrimmer(capping_method="gaussian", tail="right", fold=1)
    X = transformer.fit_transform(make_df(data_normal_dist))

    cap = transformer.right_tail_caps_["var"]
    expected = [v for v in data_normal_dist["var"] if v <= cap]

    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {"var": pytest.approx(expected)}
    assert X.shape[0] == 83


def test_gaussian_both_tails_capping_with_fold_2(make_df, data_normal_dist):
    # test case 2: mean and std, both tails, different fold value
    transformer = OutlierTrimmer(capping_method="gaussian", tail="both", fold=2)
    X = transformer.fit_transform(make_df(data_normal_dist))

    lower = transformer.left_tail_caps_["var"]
    upper = transformer.right_tail_caps_["var"]
    expected = [v for v in data_normal_dist["var"] if lower <= v <= upper]

    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {"var": pytest.approx(expected)}
    assert X.shape[0] == 96


def test_iqr_left_tail_capping_with_fold_2(make_df, data_normal_dist):
    # test case 3: IQR, left tail, fold 2
    transformer = OutlierTrimmer(capping_method="iqr", tail="left", fold=0.8)
    X = transformer.fit_transform(make_df(data_normal_dist))

    lower = transformer.left_tail_caps_["var"]
    expected = [v for v in data_normal_dist["var"] if v >= lower]

    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {"var": pytest.approx(expected)}
    assert X.shape[0] == 98


def test_mad_right_tail_capping_with_fold_1(make_df, data_normal_dist):
    # test case 4: MAD, right tail, fold 1
    transformer = OutlierTrimmer(capping_method="mad", tail="right", fold=1)
    X = transformer.fit_transform(make_df(data_normal_dist))

    cap = transformer.right_tail_caps_["var"]
    expected = [v for v in data_normal_dist["var"] if v <= cap]

    assert isinstance(X, make_df)
    assert frame_to_dict(X) == {"var": pytest.approx(expected)}
    assert X.shape[0] == 83


def test_transformer_ignores_na_in_df(make_df, data_na):
    # test case 5: dataset contains na, and transformer is asked to ignore
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
    assert X.shape[0] == 5


def test_multiple_variables_combine_bounds_with_and(make_df):
    # each variable's condition independently drops a different row; only
    # rows passing every variable's bounds should survive the combined filter
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
    "strings,expected",
    [("gaussian", 3), ("iqr", 1.5), ("mad", 3.29), ("quantiles", 0.05)],
)
def test_auto_fold_default_value(strings, expected, make_df, data_normal_dist):
    transformer = OutlierTrimmer(capping_method=strings, fold="auto")
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
