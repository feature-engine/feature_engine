import math

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from numpy import nan
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import OrdinalEncoder


def _to_backend(df: pd.DataFrame, make_df):
    """Rebuild a pandas fixture dataframe on the requested backend.

    Swaps float NaN for None in string columns - polars (unlike pandas)
    rejects a float NaN mixed into an otherwise-string column.
    """
    data = {}
    for col in df.columns:
        values = df[col].tolist()
        if any(isinstance(v, str) for v in values):
            values = [
                None if isinstance(v, float) and math.isnan(v) else v for v in values
            ]
        data[col] = values
    return make_df(data)


def _assert_values(X, expected: dict) -> None:
    """NaN-aware, backend-agnostic comparison of a dataframe's contents."""
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(expected.keys())
    for col, exp_values in expected.items():
        got_values = result[col]
        assert len(got_values) == len(exp_values)
        for got, exp in zip(got_values, exp_values):
            if isinstance(exp, float) and math.isnan(exp):
                assert got is None or (isinstance(got, float) and math.isnan(got))
            else:
                assert got == exp


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_ordered_encoding_1_variable(df_enc, make_df):
    # test case 1: 1 variable, ordered encoding
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = OrdinalEncoder(encoding_method="ordered", variables=["var_A"])
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # expected output
    expected = {
        "var_A": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
        "var_B": df_enc["var_B"].tolist(),
    }

    # test init params
    assert encoder.encoding_method == "ordered"
    assert encoder.variables == ["var_A"]
    # test fit attr
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": {"A": 1, "B": 0, "C": 2}}
    assert encoder.n_features_in_ == 2
    # test transform output
    _assert_values(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_arbitrary_encoding_automatically_find_variables(df_enc, make_df):
    # test case 2: automatically select variables, unordered encoding
    X = _to_backend(df_enc, make_df)

    encoder = OrdinalEncoder(encoding_method="arbitrary", variables=None)
    Xt = encoder.fit_transform(X)

    # expected output
    expected = {
        "var_A": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        "var_B": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        "target": df_enc["target"].tolist(),
    }

    # test init params
    assert encoder.encoding_method == "arbitrary"
    assert encoder.variables is None
    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": {"A": 0, "B": 1, "C": 2},
        "var_B": {"A": 0, "B": 1, "C": 2},
    }
    assert encoder.n_features_in_ == 3
    # test transform output
    _assert_values(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encoding_when_nan_in_fit_df(df_enc, make_df):
    data = {
        "var_A": df_enc["var_A"].tolist() + [None],
        "var_B": df_enc["var_B"].tolist() + [None],
        "target": df_enc["target"].tolist() + [0],
    }
    X = make_df(data)[["var_A", "var_B"]]
    y = data["target"]

    encoder = OrdinalEncoder(encoding_method="arbitrary", missing_values="ignore")
    encoder.fit(X)

    Xt = encoder.transform(make_df({"var_A": ["A", None], "var_B": ["A", None]}))
    _assert_values(Xt, {"var_A": [0, nan], "var_B": [0, nan]})

    encoder = OrdinalEncoder(encoding_method="ordered", missing_values="ignore")
    encoder.fit(X, y)

    Xt = encoder.transform(make_df({"var_A": ["A", None], "var_B": ["A", None]}))
    _assert_values(Xt, {"var_A": [1, nan], "var_B": [0, nan]})


@pytest.mark.parametrize("enc_method", ["other", False, 1])
def test_error_if_encoding_method_not_allowed(enc_method):
    with pytest.raises(ValueError):
        OrdinalEncoder(encoding_method=enc_method)


@pytest.mark.parametrize("enc_method", ["other", False, 1])
@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_encoding_method_not_recognized_in_fit(enc_method, df_enc, make_df):
    X = _to_backend(df_enc, make_df)
    enc = OrdinalEncoder()
    enc.encoding_method = enc_method
    with pytest.raises(ValueError):
        enc.fit(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_ordinal_encoding_and_no_y_passed(df_enc, make_df):
    # test case 3: raises error if target is  not passed
    X = _to_backend(df_enc, make_df)
    with pytest.raises(ValueError):
        encoder = OrdinalEncoder(encoding_method="ordered")
        encoder.fit(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_input_df_contains_categories_not_present_in_training_df(
    df_enc, df_enc_rare, make_df
):
    # test case 4: when dataset to be transformed contains categories not present
    # in training dataset
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()
    X_rare = _to_backend(df_enc_rare[["var_A", "var_B"]], make_df)
    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for warning when rare_labels equals 'ignore'
    with pytest.warns(UserWarning) as record:
        encoder = OrdinalEncoder(unseen="ignore")
        encoder.fit(X, y)
        encoder.transform(X_rare)

    # check that at least one warning was raised (Pandas 3 may emit additional
    # deprecation warnings)
    assert len(record) >= 1
    # check that the message matches
    assert any(r.message.args[0] == msg for r in record)

    # check for error when rare_labels equals 'raise'
    with pytest.raises(ValueError) as record2:
        encoder = OrdinalEncoder(unseen="raise")
        encoder.fit(X, y)
        encoder.transform(X_rare)

    # check that the error message matches
    assert str(record2.value) == msg


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_df_contains_na(df_enc_na, make_df):
    # test case 4: when dataset contains na, fit method
    X = _to_backend(df_enc_na, make_df)
    encoder = OrdinalEncoder(encoding_method="arbitrary")
    with pytest.raises(ValueError) as record:
        encoder.fit(X)

    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_raises_error_if_df_contains_na(df_enc, df_enc_na, make_df):
    # test case 4: when dataset contains na, transform method
    X = _to_backend(df_enc, make_df)
    X_na = _to_backend(df_enc_na, make_df)
    encoder = OrdinalEncoder(encoding_method="arbitrary")
    encoder.fit(X)
    with pytest.raises(ValueError) as record:
        encoder.transform(X_na)

    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_ordered_encoding_1_variable_ignore_format(df_enc_numeric, make_df):
    X = _to_backend(df_enc_numeric[["var_A", "var_B"]], make_df)
    y = df_enc_numeric["target"].tolist()

    encoder = OrdinalEncoder(
        encoding_method="ordered", variables=["var_A"], ignore_format=True
    )
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # expected output
    expected = {
        "var_A": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2],
        "var_B": df_enc_numeric["var_B"].tolist(),
    }

    # test init params
    assert encoder.encoding_method == "ordered"
    assert encoder.variables == ["var_A"]
    # test fit attr
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": {1: 1, 2: 0, 3: 2}}
    assert encoder.n_features_in_ == 2
    # test transform output
    _assert_values(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_arbitrary_encoding_automatically_find_variables_ignore_format(
    df_enc_numeric, make_df
):
    X = _to_backend(df_enc_numeric[["var_A", "var_B"]], make_df)

    encoder = OrdinalEncoder(
        encoding_method="arbitrary", variables=None, ignore_format=True
    )
    Xt = encoder.fit_transform(X)

    # expected output
    expected = {
        "var_A": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        "var_B": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    }

    # test init params
    assert encoder.encoding_method == "arbitrary"
    assert encoder.variables is None
    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": {1: 0, 2: 1, 3: 2},
        "var_B": {1: 0, 2: 1, 3: 2},
    }
    assert encoder.n_features_in_ == 2
    # test transform output
    _assert_values(Xt, expected)


def test_variables_cast_as_category(df_enc_category_dtypes):
    # pandas-only: polars has no equivalent "unused categorical categories"
    # concept to exercise here.
    df = df_enc_category_dtypes.copy()
    encoder = OrdinalEncoder(encoding_method="ordered", variables=["var_A"])
    encoder.fit(df[["var_A", "var_B"]], df["target"])
    X = encoder.transform(df[["var_A", "var_B"]])

    # expected output
    transf_df = df.copy()
    transf_df["var_A"] = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]

    # test transform output
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]], check_dtype=False)
    assert X["var_A"].dtypes.name == "int64"


@pytest.mark.parametrize(
    "unseen", ["empanada", False, 1, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_unseen_not_permitted_value(unseen):
    with pytest.raises(ValueError):
        OrdinalEncoder(unseen=unseen)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_when_no_unseen(make_df):
    df = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    enc = OrdinalEncoder(encoding_method="arbitrary")
    enc.fit(df)
    dft = enc.transform(df)
    expected = {"words": ["dog", "dog", "cat", "cat", "cat", "bird"]}
    _assert_values(enc.inverse_transform(dft), expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_when_ignore_unseen(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    enc = OrdinalEncoder(encoding_method="arbitrary", unseen="ignore")
    enc.fit(df1)
    dft = enc.transform(df2)
    _assert_values(
        enc.inverse_transform(dft),
        {"words": ["dog", "dog", "cat", "cat", "cat", nan]},
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_when_encode_unseen(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    enc = OrdinalEncoder(encoding_method="arbitrary", unseen="encode")
    enc.fit(df1)
    dft = enc.transform(df2)
    _assert_values(
        enc.inverse_transform(dft),
        {"words": ["dog", "dog", "cat", "cat", "cat", nan]},
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_raises_non_fitted_error(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    enc = OrdinalEncoder(encoding_method="arbitrary")

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1)

    df1_na = make_df({"words": ["dog", "dog", "cat", "cat", "cat", None]})

    with pytest.raises(ValueError):
        enc.fit(df1_na)

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encoding_new_categories(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    df_unseen = make_df({"var_A": ["D"], "var_B": ["D"]})
    encoder = OrdinalEncoder(encoding_method="arbitrary", unseen="encode")
    encoder.fit(X)
    df_transformed = encoder.transform(df_unseen)
    _assert_values(df_transformed, {"var_A": [-1], "var_B": [-1]})
