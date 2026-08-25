import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from feature_engine.outliers import ArbitraryOutlierCapper

DATA = {"var": list(np.random.RandomState(0).normal(0, 0.1, 20))}

DATA_NA = {
    "Name": ["tom", "nick", "krish", "jack", "tom", "eric"],
    "City": ["London", "Manchester", "Liverpool", "Bristol", "Manchester", "Liverpool"],
    "Age": [20.0, 21.0, 19.0, 18.0, np.nan, 41.0],
    "Marks": [0.9, 0.8, 0.7, 0.6, 0.5, 0.6],
    "dob": pd.date_range("2020-02-24", periods=6, freq="min"),
}


def _to_dict(X):
    return nw.from_native(X, eager_only=True).to_dict(as_series=False)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_right_end_capping(make_df):
    X = make_df(DATA)
    transformer = ArbitraryOutlierCapper(
        max_capping_dict={"var": 0.10727677848029868}, min_capping_dict=None
    )
    Xt = transformer.fit_transform(X)

    # expected output
    expected = [min(v, 0.10727677848029868) for v in DATA["var"]]

    # test init params
    assert np.round(transformer.max_capping_dict["var"], 3) == np.round(
        0.10727677848029868, 3
    )
    assert transformer.min_capping_dict is None
    assert transformer.variables_ == ["var"]
    # test fit attrs
    assert np.round(transformer.right_tail_caps_["var"], 3) == np.round(
        0.10727677848029868, 3
    )
    assert transformer.left_tail_caps_ == {}
    assert transformer.n_features_in_ == 1
    # test transform output
    result = _to_dict(Xt)
    assert result["var"] == pytest.approx(expected)
    assert max(result["var"]) <= 0.10727677848029868 + 1e-8


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_both_ends_capping(make_df):
    X = make_df(DATA)
    transformer = ArbitraryOutlierCapper(
        max_capping_dict={"var": 0.20857275540714884},
        min_capping_dict={"var": -0.19661115230025186},
    )
    Xt = transformer.fit_transform(X)

    # expected output
    expected = [
        min(max(v, -0.19661115230025186), 0.20857275540714884) for v in DATA["var"]
    ]

    # test fit params
    assert np.round(transformer.right_tail_caps_["var"], 3) == np.round(
        0.20857275540714884, 3
    )
    assert np.round(transformer.left_tail_caps_["var"], 3) == np.round(
        -0.19661115230025186, 3
    )
    # test transform output
    result = _to_dict(Xt)
    assert result["var"] == pytest.approx(expected)
    assert max(result["var"]) <= 0.20857275540714884 + 1e-8
    assert min(result["var"]) >= -0.19661115230025186 - 1e-8


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_left_tail_capping(make_df):
    X = make_df(DATA)
    transformer = ArbitraryOutlierCapper(
        max_capping_dict=None, min_capping_dict={"var": -0.17486039103044}
    )
    Xt = transformer.fit_transform(X)

    # expected output
    expected = [max(v, -0.17486039103044) for v in DATA["var"]]

    # test init param
    assert transformer.max_capping_dict is None
    assert np.round(transformer.min_capping_dict["var"], 3) == np.round(
        -0.17486039103044, 3
    )
    # test fit attr
    assert transformer.right_tail_caps_ == {}
    assert np.round(transformer.left_tail_caps_["var"], 3) == np.round(
        -0.17486039103044, 3
    )
    # test transform output
    result = _to_dict(Xt)
    assert result["var"] == pytest.approx(expected)
    assert min(result["var"]) >= -0.17486039103044 - 1e-8


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_ignores_na_in_input_df(make_df):
    X = make_df(DATA_NA)
    transformer = ArbitraryOutlierCapper(
        max_capping_dict=None, min_capping_dict={"Age": 20}, missing_values="ignore"
    )
    Xt = transformer.fit_transform(X)

    # expected output
    expected = [
        v if np.isnan(v) else max(v, 20) for v in DATA_NA["Age"]
    ]

    # test fit params
    assert transformer.max_capping_dict is None
    assert transformer.min_capping_dict == {"Age": 20}
    assert transformer.n_features_in_ == 5
    # test transform output
    result = _to_dict(Xt)
    assert result["Age"] == pytest.approx(expected, nan_ok=True)
    assert np.nanmin(result["Age"]) >= 20


def test_error_if_max_capping_dict_wrong_input():
    with pytest.raises(TypeError):
        ArbitraryOutlierCapper(max_capping_dict="other")
    with pytest.raises(ValueError):
        ArbitraryOutlierCapper(max_capping_dict={"a": "a"})


def test_error_if_min_capping_dict_wrong_input():
    with pytest.raises(TypeError):
        ArbitraryOutlierCapper(min_capping_dict="other")
    with pytest.raises(ValueError):
        ArbitraryOutlierCapper(min_capping_dict={"a": "a"})


def test_error_if_both_capping_dicts_are_none():
    with pytest.raises(ValueError):
        ArbitraryOutlierCapper(min_capping_dict=None, max_capping_dict=None)


def test_error_if_missing_values_not_bool():
    with pytest.raises(ValueError):
        ArbitraryOutlierCapper(missing_values="other")


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_and_transform_raise_error_if_df_contains_na(make_df):
    X = make_df(DATA)
    data_na = dict(DATA)
    var_na = list(DATA["var"])
    var_na[1] = np.nan
    data_na["var"] = var_na
    X_na = make_df(data_na)

    # test case: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = ArbitraryOutlierCapper(
            min_capping_dict={"var": -0.17486039103044}
        )
        transformer.fit(X_na)

    # test case: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = ArbitraryOutlierCapper(
            min_capping_dict={"var": -0.17486039103044}
        )
        transformer.fit(X)
        transformer.transform(X_na)


@pytest.mark.parametrize(
    "missing_values",
    ["HOLA", 1, True, {"key1": "value1", "key2": "value2", "key3": "value3"}],
)
def test_error_if_missing_values_wrong_type(missing_values):
    msg = "missing_values takes only values 'raise' or 'ignore'"
    with pytest.raises(ValueError, match=msg):
        ArbitraryOutlierCapper(
            min_capping_dict={"var": -0.17486039103044}, missing_values="missing_values"
        )
