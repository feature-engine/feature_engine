import re

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import LogCpTransformer

DATA = {
    "vara": [0, 1, 2, 3],
    "varb": [5, 5, 6, 7],
    "varc": [-2, -1, 0, 4],
    "vard": [-3, -2, -1, -5],
    "vare": ["a", "b", "c", "d"],
}
DATA_VARS = ["vara", "varb", "varc", "vard"]
DATA_AUTO_C = {"vara": 1, "varb": 0, "varc": 3, "vard": 6}

DATA_VARTYPES = {
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


def _to_dict(X):
    return nw.from_native(X, eager_only=True).to_dict(as_series=False)


def _expected_log(c, base):
    fn = np.log if base == "e" else np.log10
    out = {}
    for var in DATA_VARS:
        c_var = c[var] if isinstance(c, dict) else c
        out[var] = [fn(x + c_var) for x in DATA[var]]
    return out


@pytest.mark.parametrize("base", ["e", "10"])
def test_base_parameter(base):
    tr = LogCpTransformer(base=base)
    assert tr.base == base


@pytest.mark.parametrize("base", [False, 1, 10])
def test_base_raises_error(base):
    msg = f"base can take only '10' or 'e' as values. Got {base} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        LogCpTransformer(base=base)


@pytest.mark.parametrize("c", [1, 0.1, {"var1": 1, "var2": 2}, "auto"])
def test_c_parameter(c):
    tr = LogCpTransformer(C=c)
    assert tr.C == c


@pytest.mark.parametrize("c", ["string", [1, 2]])
def test_c_raises_error(c):
    msg = f"C can take only 'auto', integers, floats or dictionaries. Got {c} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        LogCpTransformer(C=c)


def test_instantiation_raises_future_warning():
    msg = (
        "LogCpTransformer was deprecated in version 2.0.0 in favour of "
        "LogTransformer and will be removed in version 2.1.0. "
        'Use LogTransformer(C="auto") instead.'
    )
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        LogCpTransformer()


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_C_when_auto(make_df):
    X = make_df(DATA)
    tr = LogCpTransformer(C="auto")
    tr.fit(X)
    assert tr.C_ == DATA_AUTO_C


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_C_when_dict(make_df):
    X = make_df(DATA)
    tr = LogCpTransformer(C=DATA_AUTO_C)
    tr.fit(X)
    assert tr.C_ == DATA_AUTO_C


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_C_when_int(make_df):
    X = make_df(DATA)
    tr = LogCpTransformer(C=10)
    tr.fit(X)
    assert tr.C_ == 10


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_raises_error_when_transformed_data_has_negative_values(make_df):
    X = make_df(DATA)
    tr = LogCpTransformer(C="auto")
    tr.fit(X)

    data_shifted = dict(DATA)
    data_shifted["vara"] = [v - 2 for v in DATA["vara"]]
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
    X = make_df(DATA)

    dft = LogCpTransformer(C="auto", base=base).fit_transform(X)
    result = _to_dict(dft)
    expected = _expected_log(DATA_AUTO_C, base)
    for var in DATA_VARS:
        assert result[var] == pytest.approx(expected[var], abs=1e-6)
    assert result["vare"] == DATA["vare"]

    dft = LogCpTransformer(C=10, base=base).fit_transform(X)
    result = _to_dict(dft)
    expected = _expected_log(10, base)
    for var in DATA_VARS:
        assert result[var] == pytest.approx(expected[var], abs=1e-6)
    assert result["vare"] == DATA["vare"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform(make_df):
    X = make_df(DATA)

    tr = LogCpTransformer(C="auto", base="10")
    dft = tr.fit_transform(X)
    orig = tr.inverse_transform(dft)
    result = _to_dict(orig)
    for var in DATA_VARS:
        assert result[var] == pytest.approx(DATA[var], abs=0.1)

    tr = LogCpTransformer(C=10, base="e")
    dft = tr.fit_transform(X)
    orig = tr.inverse_transform(dft)
    result = _to_dict(orig)
    for var in DATA_VARS:
        assert result[var] == pytest.approx(DATA[var], abs=0.1)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_raises_error_if_na_in_df(make_df):
    X_na = make_df(DATA_NA)
    X = make_df(DATA_VARTYPES)

    # when dataset contains na, fit method
    transformer = LogCpTransformer()
    with pytest.raises(ValueError):
        transformer.fit(X_na)

    # when dataset contains na, transform method
    transformer = LogCpTransformer()
    transformer.fit(X)
    with pytest.raises(ValueError):
        transformer.transform(X_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_non_fitted_error(make_df):
    X = make_df(DATA_VARTYPES)
    transformer = LogCpTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(X)
