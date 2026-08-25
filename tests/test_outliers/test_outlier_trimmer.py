# Authors: Soledad Galli <solegalli@gprotonmail.com>
# License: BSD 3 clause

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from feature_engine.outliers import OutlierTrimmer

# same seed/params as the pandas-only df_normal_dist fixture in tests/conftest.py,
# reproduced here as a plain dict so it can be built with either backend.
np.random.seed(0)
_NORMAL_VALUES = np.random.normal(0, 0.1, 100).tolist()
DATA_NORMAL = {"var": _NORMAL_VALUES}

DATA_NA = {
    "Age": [20, 21, 19, None, 23, 40, 41, 37],
    "Marks": [0.9, 0.8, 0.7, None, 0.3, None, 0.8, 0.6],
}

# var_a and var_b each push a different row past their own bounds (row 0 fails
# both, row 1 fails only var_b, row 4 fails only var_a) - exercises that the
# combined filter() keeps a row only when every variable's condition holds.
DATA_TWO_VARS = {"var_a": [1, 2, 3, 4, 100], "var_b": [1000, 6, 7, 8, 9]}


def _cols(X, columns):
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    return {c: result[c] for c in columns}


def _to_list(y):
    return nw.from_native(y, series_only=True).to_list()


def _make_series(make_df, values):
    return pd.Series(values) if make_df is pd.DataFrame else pl.Series(values)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_gaussian_right_tail_capping_when_fold_is_1(make_df):
    # test case 1: mean and std, right tail
    df = make_df(DATA_NORMAL)
    transformer = OutlierTrimmer(capping_method="gaussian", tail="right", fold=1)
    X = transformer.fit_transform(df)

    cap = transformer.right_tail_caps_["var"]
    expected = [v for v in DATA_NORMAL["var"] if v <= cap]

    assert _cols(X, ["var"])["var"] == pytest.approx(expected)
    assert X.shape[0] == 83


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_gaussian_both_tails_capping_with_fold_2(make_df):
    # test case 2: mean and std, both tails, different fold value
    df = make_df(DATA_NORMAL)
    transformer = OutlierTrimmer(capping_method="gaussian", tail="both", fold=2)
    X = transformer.fit_transform(df)

    lower = transformer.left_tail_caps_["var"]
    upper = transformer.right_tail_caps_["var"]
    expected = [v for v in DATA_NORMAL["var"] if lower <= v <= upper]

    assert _cols(X, ["var"])["var"] == pytest.approx(expected)
    assert X.shape[0] == 96


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_iqr_left_tail_capping_with_fold_2(make_df):
    # test case 3: IQR, left tail, fold 2
    df = make_df(DATA_NORMAL)
    transformer = OutlierTrimmer(capping_method="iqr", tail="left", fold=0.8)
    X = transformer.fit_transform(df)

    lower = transformer.left_tail_caps_["var"]
    expected = [v for v in DATA_NORMAL["var"] if v >= lower]

    assert _cols(X, ["var"])["var"] == pytest.approx(expected)
    assert X.shape[0] == 98


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_mad_right_tail_capping_with_fold_1(make_df):
    # test case 4: MAD, right tail, fold 1
    df = make_df(DATA_NORMAL)
    transformer = OutlierTrimmer(capping_method="mad", tail="right", fold=1)
    X = transformer.fit_transform(df)

    cap = transformer.right_tail_caps_["var"]
    expected = [v for v in DATA_NORMAL["var"] if v <= cap]

    assert _cols(X, ["var"])["var"] == pytest.approx(expected)
    assert X.shape[0] == 83


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transformer_ignores_na_in_df(make_df):
    # test case 5: dataset contains na, and transformer is asked to ignore
    df = make_df(DATA_NA)
    transformer = OutlierTrimmer(
        capping_method="gaussian",
        tail="right",
        fold=1,
        variables=["Age"],
        missing_values="ignore",
    )
    X = transformer.fit_transform(df)

    assert transformer.right_tail_caps_["Age"] == pytest.approx(38.04494616731882)
    assert X.shape[0] == 5


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_multiple_variables_combine_bounds_with_and(make_df):
    # each variable's condition independently drops a different row; only
    # rows passing every variable's bounds should survive the combined filter
    df = make_df(DATA_TWO_VARS)
    transformer = OutlierTrimmer(capping_method="quantiles", tail="both", fold=0.2)
    X = transformer.fit_transform(df)

    assert _cols(X, ["var_a", "var_b"]) == {"var_a": [3, 4], "var_b": [7, 8]}


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_x_y(make_df):
    df = make_df(DATA_NORMAL)
    y = _make_series(make_df, list(np.zeros(len(DATA_NORMAL["var"]))))
    transformer = OutlierTrimmer(capping_method="mad", tail="right", fold=1)
    X = transformer.fit_transform(df)
    assert X.shape[0] != len(y)

    Xt, yt = transformer.transform_x_y(df, y)
    assert Xt.shape[0] == len(_to_list(yt))
    assert Xt.shape[0] != len(DATA_NORMAL["var"])


@pytest.mark.parametrize(
    "strings,expected",
    [("gaussian", 3), ("iqr", 1.5), ("mad", 3.29), ("quantiles", 0.05)],
)
@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_auto_fold_default_value(strings, expected, make_df):
    df = make_df(DATA_NORMAL)
    transformer = OutlierTrimmer(capping_method=strings, fold="auto")
    transformer.fit(df)
    assert transformer.fold_ == expected


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_low_variation(make_df):
    low_variation_data = {"var": [v // 10 for v in DATA_NORMAL["var"]]}
    df = make_df(low_variation_data)
    transformer = OutlierTrimmer(capping_method="mad")
    with pytest.raises(
        ValueError, match="have low variation for method 'mad'"
    ):
        transformer.fit(df)
