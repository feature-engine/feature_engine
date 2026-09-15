import re

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import OrdinalEncoder
from tests.backend_helpers import make_series, frame_to_dict

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer or set the parameter "
    "`missing_values='ignore'` when initialising this transformer."
)


def test_ordered_encoding_1_variable(make_df, data_enc):
    # test case 1: 1 variable, ordered encoding
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = OrdinalEncoder(encoding_method="ordered", variables=["var_A"])
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # test init params
    assert encoder.encoding_method == "ordered"
    assert encoder.variables == ["var_A"]
    # test fit attr
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": {"A": 1, "B": 0, "C": 2}}
    assert encoder.n_features_in_ == 2
    # test transform output
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [1] * 6 + [0] * 10 + [2] * 4,
        "var_B": data_enc["var_B"],
    }


@pytest.mark.parametrize("to_target", [list, np.array])
def test_ordered_encoding_with_target_as_list_or_array(make_df, data_enc, to_target):
    # a list or numpy array target takes a different code path than a Series
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = to_target(data_enc["target"])

    encoder = OrdinalEncoder(encoding_method="ordered", variables=["var_A"])
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    assert encoder.encoder_dict_ == {"var_A": {"A": 1, "B": 0, "C": 2}}
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [1] * 6 + [0] * 10 + [2] * 4,
        "var_B": data_enc["var_B"],
    }


def test_arbitrary_encoding_automatically_find_variables(make_df, data_enc):
    # test case 2: automatically select variables, unordered encoding
    encoder = OrdinalEncoder(encoding_method="arbitrary", variables=None)
    Xt = encoder.fit_transform(make_df(data_enc))

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
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [0] * 6 + [1] * 10 + [2] * 4,
        "var_B": [0] * 10 + [1] * 6 + [2] * 4,
        "target": data_enc["target"],
    }


def test_encoding_when_nan_in_fit_df(make_df, data_enc):
    data = {
        "var_A": data_enc["var_A"] + [None],
        "var_B": data_enc["var_B"] + [None],
        "target": data_enc["target"] + [0],
    }
    X = make_df(data)[["var_A", "var_B"]]
    y = make_series(make_df, data["target"])
    X_new = make_df({"var_A": ["A", None], "var_B": ["A", None]})

    encoder = OrdinalEncoder(encoding_method="arbitrary", missing_values="ignore")
    encoder.fit(X)
    Xt = encoder.transform(X_new)
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"var_A": [0, None], "var_B": [0, None]}

    encoder = OrdinalEncoder(encoding_method="ordered", missing_values="ignore")
    encoder.fit(X, y)
    Xt = encoder.transform(X_new)
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"var_A": [1, None], "var_B": [0, None]}


@pytest.mark.parametrize("enc_method", ["other", False, 1])
def test_error_if_encoding_method_not_allowed(enc_method):
    with pytest.raises(ValueError):
        OrdinalEncoder(encoding_method=enc_method)


@pytest.mark.parametrize("enc_method", ["other", False, 1])
def test_error_if_encoding_method_not_recognized_in_fit(enc_method, make_df, data_enc):
    enc = OrdinalEncoder()
    enc.encoding_method = enc_method
    with pytest.raises(ValueError):
        enc.fit(make_df(data_enc))


def test_error_if_ordinal_encoding_and_no_y_passed(make_df, data_enc):
    # test case 3: raises error if target is  not passed
    encoder = OrdinalEncoder(encoding_method="ordered")
    with pytest.raises(ValueError):
        encoder.fit(make_df(data_enc))


def test_error_if_input_df_contains_categories_not_present_in_training_df(
    make_df, data_enc, data_enc_rare
):
    # test case 4: when dataset to be transformed contains categories not present
    # in training dataset
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])
    X_rare = make_df(data_enc_rare)[["var_A", "var_B"]]
    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for warning when unseen equals 'ignore'
    encoder = OrdinalEncoder(unseen="ignore")
    encoder.fit(X, y)
    with pytest.warns(UserWarning, match=re.escape(msg)):
        encoder.transform(X_rare)

    # check for error when unseen equals 'raise'
    encoder = OrdinalEncoder(unseen="raise")
    encoder.fit(X, y)
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.transform(X_rare)


def test_fit_raises_error_if_df_contains_na(make_df, data_enc_na):
    # test case 4: when dataset contains na, fit method
    encoder = OrdinalEncoder(encoding_method="arbitrary")
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        encoder.fit(make_df(data_enc_na))


def test_transform_raises_error_if_df_contains_na(make_df, data_enc, data_enc_na):
    # test case 4: when dataset contains na, transform method
    encoder = OrdinalEncoder(encoding_method="arbitrary")
    encoder.fit(make_df(data_enc))
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        encoder.transform(make_df(data_enc_na))


def test_ordered_encoding_1_variable_ignore_format(make_df, data_enc_numeric):
    X = make_df(data_enc_numeric)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc_numeric["target"])

    encoder = OrdinalEncoder(
        encoding_method="ordered", variables=["var_A"], ignore_format=True
    )
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # test init params
    assert encoder.encoding_method == "ordered"
    assert encoder.variables == ["var_A"]
    # test fit attr
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": {1: 1, 2: 0, 3: 2}}
    assert encoder.n_features_in_ == 2
    # test transform output
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [1] * 6 + [0] * 10 + [2] * 4,
        "var_B": data_enc_numeric["var_B"],
    }


def test_arbitrary_encoding_automatically_find_variables_ignore_format(
    make_df, data_enc_numeric
):
    X = make_df(data_enc_numeric)[["var_A", "var_B"]]

    encoder = OrdinalEncoder(
        encoding_method="arbitrary", variables=None, ignore_format=True
    )
    Xt = encoder.fit_transform(X)

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
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [0] * 6 + [1] * 10 + [2] * 4,
        "var_B": [0] * 10 + [1] * 6 + [2] * 4,
    }


def test_variables_cast_as_category(df_enc_category_dtypes):
    # pandas-only: polars has no equivalent "unused categorical categories"
    # concept to exercise here.
    df = df_enc_category_dtypes.copy()
    encoder = OrdinalEncoder(encoding_method="ordered", variables=["var_A"])
    encoder.fit(df[["var_A", "var_B"]], df["target"])
    X = encoder.transform(df[["var_A", "var_B"]])

    # expected output
    transf_df = df.copy()
    transf_df["var_A"] = [1] * 6 + [0] * 10 + [2] * 4

    # test transform output
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]], check_dtype=False)
    assert X["var_A"].dtypes.name == "int64"


@pytest.mark.parametrize(
    "unseen", ["empanada", False, 1, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_unseen_not_permitted_value(unseen):
    with pytest.raises(ValueError):
        OrdinalEncoder(unseen=unseen)


def test_inverse_transform_when_no_unseen(make_df):
    words = ["dog", "dog", "cat", "cat", "cat", "bird"]
    df = make_df({"words": words})
    enc = OrdinalEncoder(encoding_method="arbitrary")
    enc.fit(df)
    dft = enc.transform(df)
    Xi = enc.inverse_transform(dft)
    assert isinstance(Xi, make_df)
    assert frame_to_dict(Xi) == {"words": words}


def test_inverse_transform_when_ignore_unseen(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    enc = OrdinalEncoder(encoding_method="arbitrary", unseen="ignore")
    enc.fit(df1)
    dft = enc.transform(df2)
    Xi = enc.inverse_transform(dft)
    assert isinstance(Xi, make_df)
    assert frame_to_dict(Xi) == {"words": ["dog", "dog", "cat", "cat", "cat", None]}


def test_inverse_transform_when_encode_unseen(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    enc = OrdinalEncoder(encoding_method="arbitrary", unseen="encode")
    enc.fit(df1)
    dft = enc.transform(df2)
    Xi = enc.inverse_transform(dft)
    assert isinstance(Xi, make_df)
    assert frame_to_dict(Xi) == {"words": ["dog", "dog", "cat", "cat", "cat", None]}


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


def test_encoding_new_categories(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    df_unseen = make_df({"var_A": ["D"], "var_B": ["D"]})
    encoder = OrdinalEncoder(encoding_method="arbitrary", unseen="encode")
    encoder.fit(X)
    Xt = encoder.transform(df_unseen)
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"var_A": [-1], "var_B": [-1]}
