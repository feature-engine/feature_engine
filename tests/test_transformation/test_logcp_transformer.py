import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import LogCpTransformer


@pytest.mark.parametrize("base", ["e", "10"])
def test_base_parameter(base):
    tr = LogCpTransformer(base=base)
    assert tr.base == base


@pytest.mark.parametrize("base", [False, 1, 10])
def test_base_raises_error(base):
    msg = f"base can take only '10' or 'e' as values. Got {base} instead."
    with pytest.raises(ValueError) as record:
        LogCpTransformer(base=base)
    assert str(record.value) == msg


@pytest.mark.parametrize("c", [1, 0.1, {"var1": 1, "var2": 2}, "auto"])
def test_c_parameter(c):
    tr = LogCpTransformer(C=c)
    assert tr.C == c


@pytest.mark.parametrize("c", ["string", [1, 2]])
def test_c_raises_error(c):
    msg = f"C can take only 'auto', integers or floats. Got {c} instead."
    with pytest.raises(ValueError) as record:
        LogCpTransformer(C=c)
    assert str(record.value) == msg


@pytest.fixture(scope="module")
def df():
    df = pd.DataFrame(
        {
            "vara": [0, 1, 2, 3],
            "varb": [5, 5, 6, 7],
            "varc": [-2, -1, 0, 4],
            "vard": [-3, -2, -1, -5],
            "vare": ["a", "b", "c", "d"],
        }
    )
    return df


def test_C_when_auto(df):
    tr = LogCpTransformer(C="auto")
    tr.fit(df)
    c = {"vara": 1, "varb": 0, "varc": 3, "vard": 6}
    assert tr.C_ == c


def test_C_when_dict(df):
    c = {"vara": 1, "varb": 0, "varc": 3, "vard": 6}
    tr = LogCpTransformer(C=c)
    tr.fit(df)
    assert tr.C_ == c


def test_C_when_int(df):
    tr = LogCpTransformer(C=10)
    tr.fit(df)
    assert tr.C_ == 10


def test_raises_error_when_transformed_data_has_negative_values(df):
    tr = LogCpTransformer(C="auto")
    tr.fit(df)
    dft = df.copy()
    dft["vara"] = dft["vara"] - 2
    msg = (
        "Some variables contain zero or negative values after adding constant C, "
        "can't apply log."
    )
    with pytest.raises(ValueError) as record:
        tr.transform(dft)
    assert str(record.value) == msg


def test_log_base_e(df):
    dft = LogCpTransformer(C="auto").fit_transform(df)
    exp = np.log(
        df[["vara", "varb", "varc", "vard"]]
        + {"vara": 1, "varb": 0, "varc": 3, "vard": 6}
    )
    exp["vare"] = df["vare"]
    pd.testing.assert_frame_equal(dft, exp)

    dft = LogCpTransformer(C=10).fit_transform(df)
    exp = np.log(df[["vara", "varb", "varc", "vard"]] + 10)
    exp["vare"] = df["vare"]
    pd.testing.assert_frame_equal(dft, exp)


def test_log_base_10(df):
    dft = LogCpTransformer(C="auto", base="10").fit_transform(df)
    exp = np.log10(
        df[["vara", "varb", "varc", "vard"]]
        + {"vara": 1, "varb": 0, "varc": 3, "vard": 6}
    )
    exp["vare"] = df["vare"]
    pd.testing.assert_frame_equal(dft, exp)

    dft = LogCpTransformer(C=10, base="10").fit_transform(df)
    exp = np.log10(df[["vara", "varb", "varc", "vard"]] + 10)
    exp["vare"] = df["vare"]
    pd.testing.assert_frame_equal(dft, exp)


def test_inverse_transform(df):
    tr = LogCpTransformer(C="auto", base="10")
    dft = tr.fit_transform(df)
    orig = tr.inverse_transform(dft)
    pd.testing.assert_frame_equal(
        orig, df, check_dtype=False, check_exact=False, rtol=0.1
    )

    tr = LogCpTransformer(C=10, base="e")
    dft = tr.fit_transform(df)
    orig = tr.inverse_transform(dft)
    pd.testing.assert_frame_equal(
        orig, df, check_dtype=False, check_exact=False, rtol=0.1
    )


def test_raises_error_if_na_in_df(df_na, df_vartypes):
    # when dataset contains na, fit method
    transformer = LogCpTransformer()
    with pytest.raises(ValueError):
        transformer.fit(df_na)

    # when dataset contains na, transform method
    transformer = LogCpTransformer()
    transformer.fit(df_vartypes)
    with pytest.raises(ValueError):
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_non_fitted_error(df_vartypes):
    transformer = LogCpTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(df_vartypes)
