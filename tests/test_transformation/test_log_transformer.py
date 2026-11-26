import re

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import LogTransformer

DATA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
}
DATA_NA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20.0, 21.0, 19.0, np.nan],
    "Marks": [0.9, 0.8, 0.7, np.nan],
}
DATA_C = {
    "vara": [0, 1, 2, 3],
    "varb": [5, 5, 6, 7],
    "varc": [-2, -1, 0, 4],
    "vard": [-3, -2, -1, -5],
    "vare": ["a", "b", "c", "d"],
}
DATA_C_VARS = ["vara", "varb", "varc", "vard"]
DATA_C_AUTO = {"vara": 1, "varb": 0, "varc": 3, "vard": 6}


def _to_dict(X):
    return nw.from_native(X, eager_only=True).to_dict(as_series=False)


def _expected_log(c, base):
    fn = np.log if base == "e" else np.log10
    out = {}
    for var in DATA_C_VARS:
        c_var = c[var] if isinstance(c, dict) else c
        out[var] = [fn(x + c_var) for x in DATA_C[var]]
    return out


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transforming_int_vars(make_df):
    X = make_df({"var1": [1, 2, 3], "var2": [4, 5, 3]})
    transformer = LogTransformer(base="e", variables=None)
    Xt = transformer.fit_transform(X)
    result = _to_dict(Xt)
    assert result["var1"] == pytest.approx(list(np.log([1, 2, 3])))
    assert result["var2"] == pytest.approx(list(np.log([4, 5, 3])))


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_log_base_e_plus_automatically_find_variables(make_df):
    X = make_df(DATA)
    transformer = LogTransformer(base="e", variables=None)
    Xt = transformer.fit_transform(X)

    # test init params
    assert transformer.base == "e"
    assert transformer.variables is None
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 4

    # test transform output
    result = _to_dict(Xt)
    assert result["Age"] == pytest.approx(
        [2.99573, 3.04452, 2.94444, 2.89037], abs=1e-5
    )
    assert result["Marks"] == pytest.approx(
        [-0.105361, -0.223144, -0.356675, -0.510826], abs=1e-5
    )

    # test inverse_transform
    Xit = transformer.inverse_transform(Xt)
    result_it = _to_dict(Xit)
    assert [round(v) for v in result_it["Age"]] == DATA["Age"]
    assert [round(v, 1) for v in result_it["Marks"]] == DATA["Marks"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_log_base_10_plus_user_passes_var_list(make_df):
    X = make_df(DATA)
    transformer = LogTransformer(base="10", variables="Age")
    Xt = transformer.fit_transform(X)

    # test init params
    assert transformer.base == "10"
    assert transformer.variables == "Age"
    # test fit attr
    assert transformer.variables_ == ["Age"]
    assert transformer.n_features_in_ == 4

    # test transform output
    result = _to_dict(Xt)
    assert result["Age"] == pytest.approx(
        [1.30103, 1.32222, 1.27875, 1.25527], abs=1e-5
    )

    # test inverse_transform
    Xit = transformer.inverse_transform(Xt)
    result_it = _to_dict(Xit)
    assert [round(v) for v in result_it["Age"]] == DATA["Age"]


def test_error_if_base_value_not_allowed():
    msg = "base can take only '10' or 'e' as values. Got other instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        LogTransformer(base="other")


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_na_in_df(make_df):
    X = make_df(DATA_NA)
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_raises_error_if_na_in_df(make_df):
    X = make_df(DATA)
    X_na = make_df(DATA_NA)
    transformer = LogTransformer()
    transformer.fit(X)
    with pytest.raises(ValueError):
        transformer.transform(X_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_df_contains_negative_values(make_df):
    data_neg = dict(DATA)
    data_neg["Age"] = [20, -1, 19, 18]
    X = make_df(DATA)
    X_neg = make_df(data_neg)

    # when variable contains negative value, fit
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(X_neg)

    # when variable contains negative value, transform
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(X)
        transformer.transform(X_neg)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_non_fitted_error(make_df):
    X = make_df(DATA)
    with pytest.raises(NotFittedError):
        transformer = LogTransformer()
        transformer.transform(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_e_plus_user_passes_var_list(make_df):
    X = make_df(DATA)
    transformer = LogTransformer(variables="Age")
    Xt = transformer.fit_transform(X)
    Xit = transformer.inverse_transform(Xt)

    # test init params
    assert transformer.base == "e"
    assert transformer.variables == "Age"
    # test fit attr
    assert transformer.variables_ == ["Age"]
    assert transformer.n_features_in_ == 4
    # test transform output
    result_it = _to_dict(Xit)
    assert [round(v) for v in result_it["Age"]] == DATA["Age"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_default_C_preserves_original_fail_fast_behavior(make_df):
    """LogTransformer()'s default C=0 must raise at fit() time, with the
    original exact message, matching pre-merge behavior. See #957."""
    X = make_df({"x": [1, 2, 0, 4]})
    tr = LogTransformer()

    assert tr.C == 0

    msg = "Some variables contain zero or negative values, can't apply log"
    with pytest.raises(ValueError, match=re.escape(msg)):
        tr.fit(X)


@pytest.mark.parametrize("c", [1, 0.1, {"var1": 1, "var2": 2}, "auto"])
def test_c_parameter(c):
    tr = LogTransformer(C=c)
    assert tr.C == c


@pytest.mark.parametrize("c", ["string", [1, 2]])
def test_c_raises_error(c):
    msg = f"C can take only 'auto', integers, floats or dictionaries. Got {c} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        LogTransformer(C=c)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_C_when_auto(make_df):
    X = make_df(DATA_C)
    tr = LogTransformer(C="auto")
    tr.fit(X)
    assert tr.C_ == DATA_C_AUTO


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_C_when_dict(make_df):
    X = make_df(DATA_C)
    tr = LogTransformer(C=DATA_C_AUTO)
    tr.fit(X)
    assert tr.C_ == DATA_C_AUTO


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_C_when_int(make_df):
    X = make_df(DATA_C)
    tr = LogTransformer(C=10)
    tr.fit(X)
    assert tr.C_ == 10


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_raises_error_when_transformed_data_has_negative_values_with_C(make_df):
    X = make_df(DATA_C)
    tr = LogTransformer(C="auto")
    tr.fit(X)

    data_shifted = dict(DATA_C)
    data_shifted["vara"] = [v - 2 for v in DATA_C["vara"]]
    Xt = make_df(data_shifted)

    msg = (
        "Some variables contain zero or negative values after adding constant C, "
        "can't apply log."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        tr.transform(Xt)


@pytest.mark.parametrize("base", ["e", "10"])
@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_log_with_C(make_df, base):
    X = make_df(DATA_C)

    dft = LogTransformer(C="auto", base=base).fit_transform(X)
    result = _to_dict(dft)
    expected = _expected_log(DATA_C_AUTO, base)
    for var in DATA_C_VARS:
        assert result[var] == pytest.approx(expected[var], abs=1e-6)
    assert result["vare"] == DATA_C["vare"]

    dft = LogTransformer(C=10, base=base).fit_transform(X)
    result = _to_dict(dft)
    expected = _expected_log(10, base)
    for var in DATA_C_VARS:
        assert result[var] == pytest.approx(expected[var], abs=1e-6)
    assert result["vare"] == DATA_C["vare"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_with_C(make_df):
    X = make_df(DATA_C)

    tr = LogTransformer(C="auto", base="10")
    dft = tr.fit_transform(X)
    orig = tr.inverse_transform(dft)
    result = _to_dict(orig)
    for var in DATA_C_VARS:
        assert result[var] == pytest.approx(DATA_C[var], abs=0.1)

    tr = LogTransformer(C=10, base="e")
    dft = tr.fit_transform(X)
    orig = tr.inverse_transform(dft)
    result = _to_dict(orig)
    for var in DATA_C_VARS:
        assert result[var] == pytest.approx(DATA_C[var], abs=0.1)
