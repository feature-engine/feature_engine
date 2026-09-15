import re

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import MeanEncoder
from tests.backend_helpers import make_series, frame_to_dict

ENC_DICT_VAR_A = {"A": 0.3333333333333333, "B": 0.2, "C": 0.5}
ENC_DICT_VAR_B = {"A": 0.2, "B": 0.3333333333333333, "C": 0.5}


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
def test_user_enters_1_variable(make_df, data_enc):
    # test case 1: 1 variable
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = MeanEncoder(variables=["var_A"])
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # test fit attr
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": ENC_DICT_VAR_A}
    assert encoder.n_features_in_ == 2
    # test transform output
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [ENC_DICT_VAR_A[v] for v in data_enc["var_A"]],
        "var_B": data_enc["var_B"],
    }


def test_automatically_find_variables(make_df, data_enc):
    # test case 2: automatically select variables
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = MeanEncoder(variables=None)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {"var_A": ENC_DICT_VAR_A, "var_B": ENC_DICT_VAR_B}
    assert encoder.n_features_in_ == 2
    # test transform output
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [ENC_DICT_VAR_A[v] for v in data_enc["var_A"]],
        "var_B": [ENC_DICT_VAR_B[v] for v in data_enc["var_B"]],
    }


@pytest.mark.parametrize("to_target", [list, np.array])
def test_target_as_list_or_array(make_df, data_enc, to_target):
    # a list or numpy array target takes a different code path than a Series
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = to_target(data_enc["target"])

    encoder = MeanEncoder()
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    assert encoder.encoder_dict_ == {"var_A": ENC_DICT_VAR_A, "var_B": ENC_DICT_VAR_B}
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [ENC_DICT_VAR_A[v] for v in data_enc["var_A"]],
        "var_B": [ENC_DICT_VAR_B[v] for v in data_enc["var_B"]],
    }


def test_encoding_when_nan_in_fit_df(make_df, data_enc):
    data = {
        "var_A": data_enc["var_A"] + [None],
        "var_B": data_enc["var_B"] + [None],
        "target": data_enc["target"] + [0],
    }
    X = make_df(data)[["var_A", "var_B"]]
    y = make_series(make_df, data["target"])

    encoder = MeanEncoder(missing_values="ignore")
    encoder.fit(X, y)

    Xt = encoder.transform(make_df({"var_A": ["A", None], "var_B": ["A", None]}))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [0.3333333333333333, None],
        "var_B": [0.2, None],
    }


def test_raises_if_transform_df_contains_categories_not_present_in_fit_df(
    make_df, data_enc, data_enc_rare
):
    # test case 4: when dataset to be transformed contains categories not present
    # in training dataset
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])
    X_rare = make_df(data_enc_rare)[["var_A", "var_B"]]

    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for warning when unseen equals 'ignore'
    encoder = MeanEncoder(unseen="ignore")
    encoder.fit(X, y)
    with pytest.warns(UserWarning, match=re.escape(msg)):
        encoder.transform(X_rare)

    # check for error when unseen equals 'raise'
    encoder = MeanEncoder(unseen="raise")
    encoder.fit(X, y)
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.transform(X_rare)


def test_fit_raises_error_if_df_contains_na(make_df, data_enc_na):
    # test case 4: when dataset contains na, fit method
    X = make_df(data_enc_na)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc_na["target"])

    encoder = MeanEncoder()
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.fit(X, y)


def test_transform_raises_error_if_df_contains_na(make_df, data_enc, data_enc_na):
    # test case 4: when dataset contains na, transform method
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])
    X_na = make_df(data_enc_na)[["var_A", "var_B"]]

    encoder = MeanEncoder()
    encoder.fit(X, y)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.transform(X_na)


def test_user_enters_1_variable_ignore_format(make_df, data_enc_numeric):
    # test case 1: 1 variable
    X = make_df(data_enc_numeric)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc_numeric["target"])

    encoder = MeanEncoder(variables=["var_A"], ignore_format=True)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    enc_dict_var_a = {1: 0.3333333333333333, 2: 0.2, 3: 0.5}

    # test fit attr
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": enc_dict_var_a}
    assert encoder.n_features_in_ == 2
    # test transform output
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [enc_dict_var_a[v] for v in data_enc_numeric["var_A"]],
        "var_B": data_enc_numeric["var_B"],
    }


def test_automatically_find_variables_ignore_format(make_df, data_enc_numeric):
    # test case 2: automatically select variables
    X = make_df(data_enc_numeric)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc_numeric["target"])

    encoder = MeanEncoder(variables=None, ignore_format=True)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    enc_dict_var_a = {1: 0.3333333333333333, 2: 0.2, 3: 0.5}
    enc_dict_var_b = {1: 0.2, 2: 0.3333333333333333, 3: 0.5}

    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {"var_A": enc_dict_var_a, "var_B": enc_dict_var_b}
    assert encoder.n_features_in_ == 2
    # test transform output
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [enc_dict_var_a[v] for v in data_enc_numeric["var_A"]],
        "var_B": [enc_dict_var_b[v] for v in data_enc_numeric["var_B"]],
    }


def test_variables_cast_as_category(df_enc_category_dtypes):
    # pandas-only.
    df = df_enc_category_dtypes.copy()
    encoder = MeanEncoder(variables=["var_A"])
    encoder.fit(df[["var_A", "var_B"]], df["target"])
    X = encoder.transform(df[["var_A", "var_B"]])

    # expected output
    transf_df = df.copy()
    transf_df["var_A"] = [0.3333333333333333] * 6 + [0.2] * 10 + [0.5] * 4

    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]], check_dtype=False)
    assert X["var_A"].dtypes.name == "float64"


def test_auto_smoothing(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

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

    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {"var_A": var_A_dict, "var_B": var_B_dict}
    assert encoder.n_features_in_ == 2
    # test transform output
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [var_A_dict[v] for v in data_enc["var_A"]],
        "var_B": [var_B_dict[v] for v in data_enc["var_B"]],
    }


def test_value_smoothing(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

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

    # test fit attr
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {"var_A": var_A_dict, "var_B": var_B_dict}
    assert encoder.n_features_in_ == 2
    # test transform output
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [var_A_dict[v] for v in data_enc["var_A"]],
        "var_B": [var_B_dict[v] for v in data_enc["var_B"]],
    }


def test_encoding_new_categories(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])
    df_unseen = make_df({"var_A": ["D"], "var_B": ["D"]})

    encoder = MeanEncoder(unseen="encode")
    encoder.fit(X, y)
    Xt = encoder.transform(df_unseen)

    target_mean = sum(data_enc["target"]) / len(data_enc["target"])
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"var_A": [target_mean], "var_B": [target_mean]}


def test_inverse_transform_when_no_unseen(make_df):
    words = ["dog", "dog", "cat", "cat", "cat", "bird"]
    df = make_df({"words": words})
    y = make_series(make_df, [1, 0, 1, 0, 1, 0])
    enc = MeanEncoder()
    enc.fit(df, y)
    dft = enc.transform(df)
    Xi = enc.inverse_transform(dft)
    assert isinstance(Xi, make_df)
    assert frame_to_dict(Xi) == {"words": words}


def test_inverse_transform_when_ignore_unseen(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    y = make_series(make_df, [1, 0, 1, 0, 1, 0])
    enc = MeanEncoder(unseen="ignore")
    enc.fit(df1, y)
    dft = enc.transform(df2)
    Xi = enc.inverse_transform(dft)
    assert isinstance(Xi, make_df)
    assert frame_to_dict(Xi) == {"words": ["dog", "dog", "cat", "cat", "cat", None]}


def test_inverse_transform_when_encode_unseen(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    y = make_series(make_df, [1, 0, 1, 0, 1, 0])
    enc = MeanEncoder(unseen="encode")
    enc.fit(df1, y)
    dft = enc.transform(df2)
    msg = (
        "inverse_transform is not implemented for this transformer when "
        "`unseen='encode'`."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        enc.inverse_transform(dft)


def test_inverse_transform_raises_non_fitted_error(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    y = make_series(make_df, [1, 0, 1, 0, 1, 0])
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
