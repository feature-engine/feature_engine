import math
import re

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from sklearn.exceptions import NotFittedError

from feature_engine.encoding import DecisionTreeEncoder


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
            elif isinstance(exp, float):
                assert got == pytest.approx(exp)
            else:
                assert got == exp


# init parameters
@pytest.mark.parametrize("enc_method", ["count", False, 1])
def test_error_if_encoding_method_not_permitted_value(enc_method):
    msg = (
        "`encoding_method` takes only values 'ordered' and 'arbitrary'."
        f" Got {enc_method} instead."
    )
    with pytest.raises(ValueError, match=msg):
        DecisionTreeEncoder(encoding_method=enc_method)


@pytest.mark.parametrize(
    "unseen", ["string", False, ("raise", "ignore"), ["ignore"], np.nan]
)
def test_error_if_unseen_gets_not_permitted_value(unseen):
    msg = re.escape(
        "Parameter `unseen` takes only values ignore, raise, encode. "
        rf"Got {unseen} instead."
    )
    with pytest.raises(ValueError, match=msg):
        DecisionTreeEncoder(unseen=unseen)


def test_error_if_unseen_is_encode_and_fill_value_is_none():
    msg = (
        "When `unseen='encode'` you need to pass a number to `fill_value`. "
        f"Got {None} instead."
    )
    with pytest.raises(ValueError, match=msg):
        DecisionTreeEncoder(unseen="encode", fill_value=None)


@pytest.mark.parametrize("precision", ["string", 0.1, -1, np.nan])
def test_error_if_precision_gets_not_permitted_value(precision):
    msg = "Parameter `precision` takes integers or None. " f"Got {precision} instead."
    with pytest.raises(ValueError, match=msg):
        DecisionTreeEncoder(precision=precision)


@pytest.mark.parametrize(
    "encoding_method,ignore_format,precision,unseen,fill_value",
    [
        ("arbitrary", True, 1, "raise", None),
        ("ordered", False, 2, "ignore", 1),
        ("ordered", False, None, "encode", 0.1),
    ],
)
def test_init_param_assignment(
    encoding_method, ignore_format, precision, unseen, fill_value
):
    DecisionTreeEncoder(
        encoding_method=encoding_method,
        ignore_format=ignore_format,
        precision=precision,
        unseen=unseen,
        fill_value=fill_value,
    )


# fit attributes
@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encoding_dictionary(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(X, y)

    # Tree: var_A <= 1.5 -> 0.25 else 0.5
    # Tree: var_B <= 0.5 -> 0.2 else 0.4
    expected_encodings = {
        "var_A": {"A": 0.25, "B": 0.25, "C": 0.5},
        "var_B": {"A": 0.2, "B": 0.4, "C": 0.4},
    }
    assert encoder.encoder_dict_ == expected_encodings


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_ordered_encoding_dictionary(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = DecisionTreeEncoder(regression=False, encoding_method="ordered")
    encoder.fit(X, y)

    # ordered ranks: var_A -> B(mean 0.2) < A(mean 0.333) < C(mean 0.5)
    #                var_B -> A(mean 0.2) < B(mean 0.333) < C(mean 0.5)
    # so var_A's ordinal codes are B=0, A=1, C=2 (split at code <= 0.5,
    # i.e. B alone); var_B's are A=0, B=1, C=2 (split at code <= 0.5, i.e. A
    # alone). Same tree-split logic as the arbitrary-encoding case above,
    # applied to the reordered codes.
    expected_encodings = {
        "var_A": {"B": 0.2, "A": 0.4, "C": 0.4},
        "var_B": {"A": 0.2, "B": 0.4, "C": 0.4},
    }
    assert encoder.encoder_dict_ == expected_encodings


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_precision(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = DecisionTreeEncoder(regression=False, precision=1)
    encoder.fit(X, y)

    # Tree: var_A <= 1.5 -> 0.25 else 0.5
    # Tree: var_B <= 0.5 -> 0.2 else 0.4
    expected_encodings = {
        "var_A": {"A": 0.2, "B": 0.2, "C": 0.5},
        "var_B": {"A": 0.2, "B": 0.4, "C": 0.4},
    }
    assert encoder.encoder_dict_ == expected_encodings


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_classification(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    expected = {
        "var_A": [0.25] * 16 + [0.5] * 4,  # Tree: var_A <= 1.5 -> 0.25 else 0.5
        "var_B": [0.2] * 10 + [0.4] * 10,  # Tree: var_B <= 0.5 -> 0.2 else 0.4
    }
    _assert_values(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_regression(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    random = np.random.RandomState(42)
    y = random.normal(0, 0.1, len(df_enc))
    encoder = DecisionTreeEncoder(
        regression=True,
        random_state=random,
    )
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    expected = {
        "var_A": (
            [0.034348] * 6 + [-0.024679] * 10 + [-0.075473] * 4
        ),  # Tree: var_A <= 1.5 -> 0.25 else 0.5
        "var_B": [0.044806] * 10 + [-0.079066] * 10,
    }
    nw_Xt = nw.from_native(Xt, eager_only=True)
    rounded = {
        col: [round(v, 6) for v in nw_Xt.get_column(col).to_list()]
        for col in ["var_A", "var_B"]
    }
    assert rounded == expected


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_df_contains_na(df_enc_na, make_df):
    X = _to_backend(df_enc_na[["var_A", "var_B"]], make_df)
    y = df_enc_na["target"].tolist()

    encoder = DecisionTreeEncoder(regression=False)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=msg):
        encoder.fit(X, y)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_raises_error_if_df_contains_na(df_enc, df_enc_na, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    X_na = _to_backend(df_enc_na[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(X, y)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=msg):
        encoder.transform(X_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_classification_ignore_format(df_enc_numeric, make_df):
    X = _to_backend(df_enc_numeric[["var_A", "var_B"]], make_df)
    y = df_enc_numeric["target"].tolist()

    encoder = DecisionTreeEncoder(
        regression=False,
        ignore_format=True,
    )
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    expected = {
        "var_A": [0.25] * 16 + [0.5] * 4,  # Tree: var_A <= 1.5 -> 0.25 else 0.5
        "var_B": [0.2] * 10 + [0.4] * 10,  # Tree: var_B <= 0.5 -> 0.2 else 0.4
    }
    _assert_values(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_regression_ignore_format(df_enc_numeric, make_df):
    X = _to_backend(df_enc_numeric[["var_A", "var_B"]], make_df)
    random = np.random.RandomState(42)
    y = random.normal(0, 0.1, len(df_enc_numeric))
    encoder = DecisionTreeEncoder(
        regression=True,
        random_state=random,
        ignore_format=True,
    )
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    expected = {
        "var_A": (
            [0.034348] * 6 + [-0.024679] * 10 + [-0.075473] * 4
        ),  # Tree: var_A <= 1.5 -> 0.25 else 0.5
        "var_B": [0.044806] * 10 + [-0.079066] * 10,
    }
    nw_Xt = nw.from_native(Xt, eager_only=True)
    rounded = {
        col: [round(v, 6) for v in nw_Xt.get_column(col).to_list()]
        for col in ["var_A", "var_B"]
    }
    assert rounded == expected


def test_variables_cast_as_category(df_enc_category_dtypes):
    # pandas Categorical dtype has no direct polars equivalent - pandas-only.
    df = df_enc_category_dtypes.copy()
    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(df[["var_A", "var_B"]], df["target"])
    X = encoder.transform(df[["var_A", "var_B"]])

    transf_df = df.copy()
    transf_df["var_A"] = [0.25] * 16 + [0.5] * 4  # Tree: var_A <= 1.5 -> 0.25 else 0.5
    transf_df["var_B"] = [0.2] * 10 + [0.4] * 10  # Tree: var_B <= 0.5 -> 0.2 else 0.4
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]], check_dtype=False)
    assert X["var_A"].dtypes == float


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_when_regression_is_true_and_target_is_binary(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = DecisionTreeEncoder(regression=True)
    msg = (
        "Trying to fit a regression to a binary target is not "
        "allowed by this transformer. Check the target values "
        "or set regression to False."
    )
    with pytest.raises(ValueError, match=msg):
        encoder.fit(X, y)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_when_regression_is_false_and_target_is_continuous(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    random = np.random.RandomState(42)
    y = random.normal(0, 10, len(df_enc))
    encoder = DecisionTreeEncoder(regression=False)
    # the error message comes from sklearn api - won't test
    with pytest.raises(ValueError):
        encoder.fit(X, y)


@pytest.mark.parametrize(
    "grid",
    [None, {"max_depth": [1, 2, 3]}, {"max_depth": [1, 2], "estimators": [10, 12]}],
)
def test_assigns_param_grid(grid):
    encoder = DecisionTreeEncoder(param_grid=grid)
    if grid is None:
        assert encoder._assign_param_grid() == {"max_depth": [1, 2, 3, 4]}
    else:
        assert encoder._assign_param_grid() == grid


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_unseen_is_encode(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = DecisionTreeEncoder(unseen="encode", regression=False, fill_value=-1)
    encoder.fit(X, y)

    X_unseen_input = make_df(
        {
            "var_A": ["A", "ZZZ", "YYY"],
            "var_B": ["C", "YYY", "ZZZ"],
        }
    )
    expected = {
        "var_A": [0.25, -1, -1],
        "var_B": [0.4, -1, -1],
    }

    Xt = encoder.transform(X_unseen_input)
    _assert_values(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_unseen_is_ignore(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = DecisionTreeEncoder(unseen="ignore", regression=False)
    encoder.fit(X, y)

    X_unseen_input = make_df(
        {
            "var_A": ["A", "ZZZ", "YYY"],
            "var_B": ["C", "YYY", "ZZZ"],
        }
    )
    expected = {
        "var_A": [0.25, np.nan, np.nan],
        "var_B": [0.4, np.nan, np.nan],
    }

    Xt = encoder.transform(X_unseen_input)
    _assert_values(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_errors_if_new_cat_values_and_unseen_is_raise_param(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = DecisionTreeEncoder(unseen="raise", regression=False)
    encoder.fit(X, y)
    X_unseen = make_df(
        {
            "var_A": ["A", "ZZZ", "YYY"],
            "var_B": ["C", "YYY", "ZZZ"],
        }
    )
    var_ls = "var_A, var_B"
    msg = (
        "During the encoding, NaN values were introduced in the "
        rf"feature\(s\) {var_ls}."
    )
    # new categories will raise an error
    with pytest.raises(ValueError, match=msg):
        encoder.transform(X_unseen)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_when_no_unseen(make_df):
    X = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", "bird"]})
    y = [0, 0, 1, 1, 1, 1, 0]
    enc = DecisionTreeEncoder(regression=False)
    enc.fit(X, y)
    dft = enc.transform(X)
    Xi = enc.inverse_transform(dft)
    _assert_values(Xi, {"words": ["dog", "dog", "dog", "cat", "cat", "cat", "bird"]})


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_when_ignore_unseen(make_df):
    X = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", "bird"]})
    y = [0, 0, 1, 1, 1, 1, 0]
    enc = DecisionTreeEncoder(regression=False, unseen="ignore")
    enc.fit(X, y)

    df1 = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", "frog"]})
    dft = enc.transform(df1)
    Xi = enc.inverse_transform(dft)
    _assert_values(
        Xi, {"words": ["dog", "dog", "dog", "cat", "cat", "cat", np.nan]}
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_when_encode_unseen(make_df):
    X = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", "bird"]})
    y = [0, 0, 1, 1, 1, 1, 0]
    enc = DecisionTreeEncoder(regression=False, unseen="encode", fill_value=1000)
    enc.fit(X, y)

    df1 = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", "frog"]})
    dft = enc.transform(df1)
    Xi = enc.inverse_transform(dft)
    _assert_values(
        Xi, {"words": ["dog", "dog", "dog", "cat", "cat", "cat", np.nan]}
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_raises_non_fitted_error(make_df):
    X = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", "bird"]})
    y = [0, 0, 1, 1, 1, 1, 0]
    enc = DecisionTreeEncoder()

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(X)

    X_na = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", None]})

    with pytest.raises(ValueError):
        enc.fit(X_na, y)

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(X_na)
