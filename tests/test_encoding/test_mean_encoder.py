import math

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from numpy import nan
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import MeanEncoder


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
                assert got == pytest.approx(exp)


# test init params
@pytest.mark.parametrize("params", [("raise", True, "auto"), ("ignore", False, 1)])
def test_init_param_assignment(params):
    MeanEncoder(
        missing_values=params[0],
        ignore_format=params[1],
        unseen=params[0],
        smoothing=params[2],
    )


@pytest.mark.parametrize(
    "errors", ["empanada", False, 1, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_unseen_gets_not_permitted_value(errors):
    with pytest.raises(ValueError):
        MeanEncoder(unseen=errors)


@pytest.mark.parametrize("smoothing", ["hello", ["auto"], -1])
def test_raises_error_when_not_allowed_smoothing_param_in_init(smoothing):
    with pytest.raises(ValueError):
        MeanEncoder(smoothing=smoothing)


# fit and transform
@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_user_enters_1_variable(df_enc, make_df):
    # test case 1: 1 variable
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = MeanEncoder(variables=["var_A"])
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # expected output
    expected = {
        "var_A": [
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        "var_B": df_enc["var_B"].tolist(),
    }

    # test init params
    assert encoder.variables == ["var_A"]
    # test fit attr
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {
        "var_A": {"A": 0.3333333333333333, "B": 0.2, "C": 0.5}
    }
    assert encoder.n_features_in_ == 2
    # test transform output
    _assert_values(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_find_variables(df_enc, make_df):
    # test case 2: automatically select variables
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = MeanEncoder(variables=None)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # expected output
    expected = {
        "var_A": [
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        "var_B": [
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.5,
            0.5,
            0.5,
            0.5,
        ],
    }

    # test init params
    assert encoder.variables is None
    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": {"A": 0.3333333333333333, "B": 0.2, "C": 0.5},
        "var_B": {"A": 0.2, "B": 0.3333333333333333, "C": 0.5},
    }
    assert encoder.n_features_in_ == 2
    # test transform output
    _assert_values(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encoding_when_nan_in_fit_df(df_enc, make_df):
    data = {
        "var_A": df_enc["var_A"].tolist() + [None],
        "var_B": df_enc["var_B"].tolist() + [None],
        "target": df_enc["target"].tolist() + [0],
    }
    df = make_df(data)

    encoder = MeanEncoder(missing_values="ignore")
    encoder.fit(df[["var_A", "var_B"]], data["target"])

    Xt = encoder.transform(make_df({"var_A": ["A", None], "var_B": ["A", None]}))

    _assert_values(
        Xt,
        {
            "var_A": [0.3333333333333333, nan],
            "var_B": [0.2, nan],
        },
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_warning_if_transform_df_contains_categories_not_present_in_fit_df(
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
        encoder = MeanEncoder(unseen="ignore")
        encoder.fit(X, y)
        encoder.transform(X_rare)

    # check that at least one warning was raised (Pandas 3 may emit additional
    # deprecation warnings)
    assert len(record) >= 1
    # check that the message matches
    assert any(r.message.args[0] == msg for r in record)

    # check for error when rare_labels equals 'raise'
    with pytest.raises(ValueError) as record2:
        encoder = MeanEncoder(unseen="raise")
        encoder.fit(X, y)
        encoder.transform(X_rare)

    # check that the error message matches
    assert str(record2.value) == msg


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fit_raises_error_if_df_contains_na(df_enc_na, make_df):
    # test case 4: when dataset contains na, fit method
    X = _to_backend(df_enc_na[["var_A", "var_B"]], make_df)
    y = df_enc_na["target"].tolist()

    encoder = MeanEncoder()
    with pytest.raises(ValueError) as record:
        encoder.fit(X, y)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_transform_raises_error_if_df_contains_na(df_enc, df_enc_na, make_df):
    # test case 4: when dataset contains na, transform method
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()
    X_na = _to_backend(df_enc_na[["var_A", "var_B"]], make_df)

    encoder = MeanEncoder()
    encoder.fit(X, y)
    with pytest.raises(ValueError) as record:
        encoder.transform(X_na)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_user_enters_1_variable_ignore_format(df_enc_numeric, make_df):
    # test case 1: 1 variable
    X = _to_backend(df_enc_numeric[["var_A", "var_B"]], make_df)
    y = df_enc_numeric["target"].tolist()

    encoder = MeanEncoder(variables=["var_A"], ignore_format=True)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # expected output
    expected = {
        "var_A": [
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        "var_B": df_enc_numeric["var_B"].tolist(),
    }

    # test init params
    assert encoder.variables == ["var_A"]
    # test fit attr
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": {1: 0.3333333333333333, 2: 0.2, 3: 0.5}}
    assert encoder.n_features_in_ == 2
    # test transform output
    _assert_values(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_find_variables_ignore_format(df_enc_numeric, make_df):
    # test case 2: automatically select variables
    X = _to_backend(df_enc_numeric[["var_A", "var_B"]], make_df)
    y = df_enc_numeric["target"].tolist()

    encoder = MeanEncoder(variables=None, ignore_format=True)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # expected output
    expected = {
        "var_A": [
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.5,
            0.5,
            0.5,
            0.5,
        ],
        "var_B": [
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333,
            0.5,
            0.5,
            0.5,
            0.5,
        ],
    }

    # test init params
    assert encoder.variables is None
    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": {1: 0.3333333333333333, 2: 0.2, 3: 0.5},
        "var_B": {1: 0.2, 2: 0.3333333333333333, 3: 0.5},
    }
    assert encoder.n_features_in_ == 2
    # test transform output
    _assert_values(Xt, expected)


def test_variables_cast_as_category(df_enc_category_dtypes):
    # pandas-only: exercises pandas Categorical dtype, which polars has no
    # direct equivalent for.
    df = df_enc_category_dtypes.copy()
    encoder = MeanEncoder(variables=["var_A"])
    encoder.fit(df[["var_A", "var_B"]], df["target"])
    X = encoder.transform(df[["var_A", "var_B"]])

    # expected output
    transf_df = df.copy()
    transf_df["var_A"] = [
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.5,
        0.5,
        0.5,
        0.5,
    ]

    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]], check_dtype=False)
    assert X["var_A"].dtypes.name == "float64"


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_auto_smoothing(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = MeanEncoder(smoothing="auto")
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # expected output
    var_A_dict = {
        "A": 0.328335832083958,
        "B": 0.20707964601769913,
        "C": 0.4541284403669725,
    }
    var_B_dict = {
        "A": 0.20707964601769913,
        "B": 0.328335832083958,
        "C": 0.4541284403669725,
    }
    expected = {
        "var_A": [var_A_dict[v] for v in df_enc["var_A"]],
        "var_B": [var_B_dict[v] for v in df_enc["var_B"]],
    }

    # test init params
    assert encoder.variables is None
    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": var_A_dict,
        "var_B": var_B_dict,
    }
    assert encoder.n_features_in_ == 2
    # test transform output
    _assert_values(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_value_smoothing(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()

    encoder = MeanEncoder(smoothing=100)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # expected output
    var_A_dict = {
        "A": 0.3018867924528302,
        "B": 0.2909090909090909,
        "C": 0.30769230769230765,
    }
    var_B_dict = {
        "A": 0.2909090909090909,
        "B": 0.3018867924528302,
        "C": 0.30769230769230765,
    }
    expected = {
        "var_A": [var_A_dict[v] for v in df_enc["var_A"]],
        "var_B": [var_B_dict[v] for v in df_enc["var_B"]],
    }

    # test init params
    assert encoder.variables is None
    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": var_A_dict,
        "var_B": var_B_dict,
    }
    assert encoder.n_features_in_ == 2
    # test transform output
    _assert_values(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_encoding_new_categories(df_enc, make_df):
    X = _to_backend(df_enc[["var_A", "var_B"]], make_df)
    y = df_enc["target"].tolist()
    df_unseen = make_df({"var_A": ["D"], "var_B": ["D"]})

    encoder = MeanEncoder(unseen="encode")
    encoder.fit(X, y)
    df_transformed = encoder.transform(df_unseen)
    _assert_values(
        df_transformed,
        {"var_A": [df_enc["target"].mean()], "var_B": [df_enc["target"].mean()]},
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_when_no_unseen(make_df):
    df = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    y = [1, 0, 1, 0, 1, 0]
    enc = MeanEncoder()
    enc.fit(df, y)
    dft = enc.transform(df)
    expected = {"words": ["dog", "dog", "cat", "cat", "cat", "bird"]}
    _assert_values(enc.inverse_transform(dft), expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_when_ignore_unseen(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    y = [1, 0, 1, 0, 1, 0]
    enc = MeanEncoder(unseen="ignore")
    enc.fit(df1, y)
    dft = enc.transform(df2)
    _assert_values(
        enc.inverse_transform(dft),
        {"words": ["dog", "dog", "cat", "cat", "cat", nan]},
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_when_encode_unseen(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    y = [1, 0, 1, 0, 1, 0]
    enc = MeanEncoder(unseen="encode")
    enc.fit(df1, y)
    dft = enc.transform(df2)
    with pytest.raises(NotImplementedError) as record:
        enc.inverse_transform(dft)
    msg = (
        "inverse_transform is not implemented for this transformer when "
        "`unseen='encode'`."
    )
    assert str(record.value) == msg


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_raises_non_fitted_error(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    y = [1, 0, 1, 0, 1, 0]
    enc = MeanEncoder()

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1)

    df1_na = make_df({"words": ["dog", "dog", "cat", "cat", "cat", None]})

    with pytest.raises(ValueError):
        enc.fit(df1_na, y)

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1_na)
