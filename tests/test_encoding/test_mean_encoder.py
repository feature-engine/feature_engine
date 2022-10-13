import numpy as np
import pandas as pd
import pytest

from feature_engine.encoding import MeanEncoder


def test_user_enters_1_variable(df_enc):
    # test case 1: 1 variable
    encoder = MeanEncoder(variables=["var_A"])
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    # expected output
    transf_df = df_enc.copy()
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

    # test init params
    assert encoder.variables == ["var_A"]
    # test fit attr
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {
        "var_A": {"A": 0.3333333333333333, "B": 0.2, "C": 0.5}
    }
    assert encoder.n_features_in_ == 2
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_automatically_find_variables(df_enc):
    # test case 2: automatically select variables
    encoder = MeanEncoder(variables=None)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    # expected output
    transf_df = df_enc.copy()
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
    transf_df["var_B"] = [
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
    ]

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
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_warning_if_transform_df_contains_categories_not_present_in_fit_df(
    df_enc, df_enc_rare
):
    # test case 4: when dataset to be transformed contains categories not present
    # in training dataset

    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for warning when rare_labels equals 'ignore'
    with pytest.warns(UserWarning) as record:
        encoder = MeanEncoder(unseen="ignore")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[0] == msg

    # check for error when rare_labels equals 'raise'
    with pytest.raises(ValueError) as record:
        encoder = MeanEncoder(unseen="raise")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])

    # check that the error message matches
    assert str(record.value) == msg


def test_fit_raises_error_if_df_contains_na(df_enc_na):
    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = MeanEncoder()
        encoder.fit(df_enc_na[["var_A", "var_B"]], df_enc_na["target"])


def test_transform_raises_error_if_df_contains_na(df_enc, df_enc_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = MeanEncoder()
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_na)


def test_user_enters_1_variable_ignore_format(df_enc_numeric):
    # test case 1: 1 variable
    encoder = MeanEncoder(variables=["var_A"], ignore_format=True)
    encoder.fit(df_enc_numeric[["var_A", "var_B"]], df_enc_numeric["target"])
    X = encoder.transform(df_enc_numeric[["var_A", "var_B"]])

    # expected output
    transf_df = df_enc_numeric.copy()
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

    # test init params
    assert encoder.variables == ["var_A"]
    # test fit attr
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": {1: 0.3333333333333333, 2: 0.2, 3: 0.5}}
    assert encoder.n_features_in_ == 2
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_automatically_find_variables_ignore_format(df_enc_numeric):
    # test case 2: automatically select variables
    encoder = MeanEncoder(variables=None, ignore_format=True)
    encoder.fit(df_enc_numeric[["var_A", "var_B"]], df_enc_numeric["target"])
    X = encoder.transform(df_enc_numeric[["var_A", "var_B"]])

    # expected output
    transf_df = df_enc_numeric.copy()
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
    transf_df["var_B"] = [
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
    ]

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
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_variables_cast_as_category(df_enc_category_dtypes):
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
    assert X["var_A"].dtypes == float


def test_auto_smoothing(df_enc):
    encoder = MeanEncoder(smoothing="auto")
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    # expected output
    transf_df = df_enc.copy()
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
    transf_df["var_A"] = transf_df["var_A"].map(var_A_dict)
    transf_df["var_B"] = transf_df["var_B"].map(var_B_dict)

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
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_value_smoothing(df_enc):
    encoder = MeanEncoder(smoothing=100)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    # expected output
    transf_df = df_enc.copy()
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
    transf_df["var_A"] = transf_df["var_A"].map(var_A_dict)
    transf_df["var_B"] = transf_df["var_B"].map(var_B_dict)

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
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_error_if_rare_labels_not_permitted_value():
    with pytest.raises(ValueError):
        MeanEncoder(unseen="empanada")


def test_encoding_new_categories(df_enc):
    df_unseen = pd.DataFrame({"var_A": ["D"], "var_B": ["D"]})
    encoder = MeanEncoder(unseen="encode")
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    df_transformed = encoder.transform(df_unseen)
    assert (df_transformed == df_enc["target"].mean()).all(axis=None)


def test_inverse_transform_when_no_unseen():
    df = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    y = [1, 0, 1, 0, 1, 0]
    enc = MeanEncoder()
    enc.fit(df, y)
    dft = enc.transform(df)
    pd.testing.assert_frame_equal(enc.inverse_transform(dft), df)


def test_inverse_transform_when_ignore_unseen():
    df1 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    df3 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", np.nan]})
    y = [1, 0, 1, 0, 1, 0]
    enc = MeanEncoder(unseen="ignore")
    enc.fit(df1, y)
    dft = enc.transform(df2)
    pd.testing.assert_frame_equal(enc.inverse_transform(dft), df3)


def test_inverse_transform_when_encode_unseen():
    df1 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    y = [1, 0, 1, 0, 1, 0]
    enc = MeanEncoder(unseen="encode")
    enc.fit(df1, y)
    dft = enc.transform(df2)
    with pytest.raises(NotImplementedError):
        enc.inverse_transform(dft)


@pytest.mark.parametrize("smoothing", ["hello", ["auto"], -1])
def test_raises_error_when_not_allowed_smoothing_param_in_init(smoothing):
    with pytest.raises(ValueError):
        MeanEncoder(smoothing=smoothing)
