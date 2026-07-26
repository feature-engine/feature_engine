import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import LogTransformer


def test_transforming_int_vars():
    df = pd.DataFrame(
        {
            "var1": [1, 2, 3],
            "var2": [4, 5, 3],
        }
    )
    dft = np.log(df)
    transformer = LogTransformer(base="e", variables=None)
    X = transformer.fit_transform(df)
    pd.testing.assert_frame_equal(X, dft)


def test_log_base_e_plus_automatically_find_variables(df_vartypes):
    # test case 1: log base e, automatically select variables
    transformer = LogTransformer(base="e", variables=None)
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = [2.99573, 3.04452, 2.94444, 2.89037]
    transf_df["Marks"] = [-0.105361, -0.223144, -0.356675, -0.510826]

    # test init params
    assert transformer.base == "e"
    assert transformer.variables is None
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 5
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)

    # test inverse_transform
    Xit = transformer.inverse_transform(X)

    # convert numbers to original format.
    Xit["Age"] = Xit["Age"].round().astype("int64")
    Xit["Marks"] = Xit["Marks"].round(1)

    # test
    pd.testing.assert_frame_equal(Xit, df_vartypes)


def test_log_base_10_plus_user_passes_var_list(df_vartypes):
    # test case 2: log base 10, user passes variables
    transformer = LogTransformer(base="10", variables="Age")
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = [1.30103, 1.32222, 1.27875, 1.25527]

    # test init params
    assert transformer.base == "10"
    assert transformer.variables == "Age"
    # test fit attr
    assert transformer.variables_ == ["Age"]
    assert transformer.n_features_in_ == 5
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)

    # test inverse_transform
    Xit = transformer.inverse_transform(X)

    # convert numbers to original format.
    Xit["Age"] = Xit["Age"].round().astype("int64")

    # test
    pd.testing.assert_frame_equal(Xit, df_vartypes)


def test_error_if_base_value_not_allowed():
    with pytest.raises(ValueError):
        LogTransformer(base="other")


def test_fit_raises_error_if_na_in_df(df_na):
    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(df_na)


def test_transform_raises_error_if_na_in_df(df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_error_if_df_contains_negative_values(df_vartypes):
    # test error when data contains negative values
    df_neg = df_vartypes.copy()
    df_neg.loc[1, "Age"] = -1

    # test case 5: when variable contains negative value, fit
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(df_neg)

    # test case 6: when variable contains negative value, transform
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_neg)


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = LogTransformer()
        transformer.transform(df_vartypes)


def test_inverse_e_plus_user_passes_var_list(df_vartypes):
    # test case 7: inverse log, user passes variables
    transformer = LogTransformer(variables="Age")
    Xt = transformer.fit_transform(df_vartypes)
    X = transformer.inverse_transform(Xt)

    # convert floats to int
    X["Age"] = X["Age"].round().astype("int64")

    # test init params
    assert transformer.base == "e"
    assert transformer.variables == "Age"
    # test fit attr
    assert transformer.variables_ == ["Age"]
    assert transformer.n_features_in_ == 5
    # test transform output
    pd.testing.assert_frame_equal(X, df_vartypes)


def test_default_C_preserves_original_fail_fast_behavior():
    """LogTransformer()'s default C=0 must raise at fit() time, with the
    original exact message, matching pre-merge behavior. See #957."""
    df = pd.DataFrame({"x": [1, 2, 0, 4]})
    tr = LogTransformer()

    assert tr.C == 0

    with pytest.raises(ValueError) as record:
        tr.fit(df)

    assert str(record.value) == (
        "Some variables contain zero or negative values, can't apply log"
    )


@pytest.fixture(scope="module")
def df_c():
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


@pytest.mark.parametrize("c", [1, 0.1, {"var1": 1, "var2": 2}, "auto"])
def test_c_parameter(c):
    tr = LogTransformer(C=c)
    assert tr.C == c


@pytest.mark.parametrize("c", ["string", [1, 2]])
def test_c_raises_error(c):
    msg = f"C can take only 'auto', integers or floats. Got {c} instead."
    with pytest.raises(ValueError) as record:
        LogTransformer(C=c)
    assert str(record.value) == msg


def test_C_when_auto(df_c):
    tr = LogTransformer(C="auto")
    tr.fit(df_c)
    c = {"vara": 1, "varb": 0, "varc": 3, "vard": 6}
    assert tr.C_ == c


def test_C_when_dict(df_c):
    c = {"vara": 1, "varb": 0, "varc": 3, "vard": 6}
    tr = LogTransformer(C=c)
    tr.fit(df_c)
    assert tr.C_ == c


def test_C_when_int(df_c):
    tr = LogTransformer(C=10)
    tr.fit(df_c)
    assert tr.C_ == 10


def test_raises_error_when_transformed_data_has_negative_values_with_C(df_c):
    tr = LogTransformer(C="auto")
    tr.fit(df_c)
    dft = df_c.copy()
    dft["vara"] = dft["vara"] - 2
    msg = (
        "Some variables contain zero or negative values after adding constant C, "
        "can't apply log."
    )
    with pytest.raises(ValueError) as record:
        tr.transform(dft)
    assert str(record.value) == msg


def test_log_base_e_with_C(df_c):
    dft = LogTransformer(C="auto").fit_transform(df_c)
    exp = np.log(
        df_c[["vara", "varb", "varc", "vard"]]
        + {"vara": 1, "varb": 0, "varc": 3, "vard": 6}
    )
    exp["vare"] = df_c["vare"]
    pd.testing.assert_frame_equal(dft, exp)

    dft = LogTransformer(C=10).fit_transform(df_c)
    exp = np.log(df_c[["vara", "varb", "varc", "vard"]] + 10)
    exp["vare"] = df_c["vare"]
    pd.testing.assert_frame_equal(dft, exp)


def test_log_base_10_with_C(df_c):
    dft = LogTransformer(C="auto", base="10").fit_transform(df_c)
    exp = np.log10(
        df_c[["vara", "varb", "varc", "vard"]]
        + {"vara": 1, "varb": 0, "varc": 3, "vard": 6}
    )
    exp["vare"] = df_c["vare"]
    pd.testing.assert_frame_equal(dft, exp)

    dft = LogTransformer(C=10, base="10").fit_transform(df_c)
    exp = np.log10(df_c[["vara", "varb", "varc", "vard"]] + 10)
    exp["vare"] = df_c["vare"]
    pd.testing.assert_frame_equal(dft, exp)


def test_inverse_transform_with_C(df_c):
    tr = LogTransformer(C="auto", base="10")
    dft = tr.fit_transform(df_c)
    orig = tr.inverse_transform(dft)
    pd.testing.assert_frame_equal(
        orig, df_c, check_dtype=False, check_exact=False, rtol=0.1
    )

    tr = LogTransformer(C=10, base="e")
    dft = tr.fit_transform(df_c)
    orig = tr.inverse_transform(dft)
    pd.testing.assert_frame_equal(
        orig, df_c, check_dtype=False, check_exact=False, rtol=0.1
    )
