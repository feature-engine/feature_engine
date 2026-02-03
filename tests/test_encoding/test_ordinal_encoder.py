import pandas as pd
import pytest
from numpy import nan
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import OrdinalEncoder


def test_ordered_encoding_1_variable(df_enc):
    # test case 1: 1 variable, ordered encoding
    encoder = OrdinalEncoder(encoding_method="ordered", variables=["var_A"])
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    # expected output
    transf_df = df_enc.copy()
    transf_df["var_A"] = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]

    # test init params
    assert encoder.encoding_method == "ordered"
    assert encoder.variables == ["var_A"]
    # test fit attr
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": {"A": 1, "B": 0, "C": 2}}
    assert encoder.n_features_in_ == 2
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_arbitrary_encoding_automatically_find_variables(df_enc):
    # test case 2: automatically select variables, unordered encoding
    encoder = OrdinalEncoder(encoding_method="arbitrary", variables=None)
    X = encoder.fit_transform(df_enc)

    # expected output
    transf_df = df_enc.copy()
    transf_df["var_A"] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    transf_df["var_B"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]

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
    pd.testing.assert_frame_equal(X, transf_df)


def test_encoding_when_nan_in_fit_df(df_enc):
    df = df_enc.copy()
    df.loc[len(df)] = [nan, nan, 0]

    encoder = OrdinalEncoder(encoding_method="arbitrary", missing_values="ignore")
    encoder.fit(df[["var_A", "var_B"]])

    X = encoder.transform(
        pd.DataFrame(
            {
                "var_A": ["A", nan],
                "var_B": ["A", nan],
            }
        )
    )

    # transform params
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame(
            {
                "var_A": [0, nan],
                "var_B": [0, nan],
            }
        ),
        check_dtype=False,
    )

    encoder = OrdinalEncoder(encoding_method="ordered", missing_values="ignore")
    encoder.fit(df[["var_A", "var_B"]], df["target"])

    X = encoder.transform(
        pd.DataFrame(
            {
                "var_A": ["A", nan],
                "var_B": ["A", nan],
            }
        )
    )

    # transform params
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame(
            {
                "var_A": [1, nan],
                "var_B": [0, nan],
            }
        ),
        check_dtype=False,
    )


@pytest.mark.parametrize("enc_method", ["other", False, 1])
def test_error_if_encoding_method_not_allowed(enc_method):
    with pytest.raises(ValueError):
        OrdinalEncoder(encoding_method=enc_method)


@pytest.mark.parametrize("enc_method", ["other", False, 1])
def test_error_if_encoding_method_not_recognized_in_fit(enc_method, df_enc):
    enc = OrdinalEncoder()
    enc.encoding_method = enc_method
    with pytest.raises(ValueError):
        enc.fit(df_enc)


def test_error_if_ordinal_encoding_and_no_y_passed(df_enc):
    # test case 3: raises error if target is  not passed
    with pytest.raises(ValueError):
        encoder = OrdinalEncoder(encoding_method="ordered")
        encoder.fit(df_enc)


def test_error_if_input_df_contains_categories_not_present_in_training_df(
    df_enc, df_enc_rare
):
    # test case 4: when dataset to be transformed contains categories not present
    # in training dataset
    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for warning when rare_labels equals 'ignore'
    with pytest.warns(UserWarning) as record:
        encoder = OrdinalEncoder(unseen="ignore")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[0] == msg

    # check for error when rare_labels equals 'raise'
    with pytest.raises(ValueError) as record:
        encoder = OrdinalEncoder(unseen="raise")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])

    # check that the error message matches
    assert str(record.value) == msg


def test_fit_raises_error_if_df_contains_na(df_enc_na):
    # test case 4: when dataset contains na, fit method
    encoder = OrdinalEncoder(encoding_method="arbitrary")
    with pytest.raises(ValueError) as record:
        encoder.fit(df_enc_na)

    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


def test_transform_raises_error_if_df_contains_na(df_enc, df_enc_na):
    # test case 4: when dataset contains na, transform method
    encoder = OrdinalEncoder(encoding_method="arbitrary")
    encoder.fit(df_enc)
    with pytest.raises(ValueError) as record:
        encoder.transform(df_enc_na)

    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


def test_ordered_encoding_1_variable_ignore_format(df_enc_numeric):

    encoder = OrdinalEncoder(
        encoding_method="ordered", variables=["var_A"], ignore_format=True
    )
    encoder.fit(df_enc_numeric[["var_A", "var_B"]], df_enc_numeric["target"])
    X = encoder.transform(df_enc_numeric[["var_A", "var_B"]])

    # expected output
    transf_df = df_enc_numeric.copy()
    transf_df["var_A"] = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]

    # test init params
    assert encoder.encoding_method == "ordered"
    assert encoder.variables == ["var_A"]
    # test fit attr
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {"var_A": {1: 1, 2: 0, 3: 2}}
    assert encoder.n_features_in_ == 2
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_arbitrary_encoding_automatically_find_variables_ignore_format(df_enc_numeric):

    encoder = OrdinalEncoder(
        encoding_method="arbitrary", variables=None, ignore_format=True
    )
    X = encoder.fit_transform(df_enc_numeric[["var_A", "var_B"]])

    # expected output
    transf_df = df_enc_numeric[["var_A", "var_B"]].copy()
    transf_df["var_A"] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    transf_df["var_B"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]

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
    pd.testing.assert_frame_equal(X, transf_df)


def test_variables_cast_as_category(df_enc_category_dtypes):
    df = df_enc_category_dtypes.copy()
    encoder = OrdinalEncoder(encoding_method="ordered", variables=["var_A"])
    encoder.fit(df[["var_A", "var_B"]], df["target"])
    X = encoder.transform(df[["var_A", "var_B"]])

    # expected output
    transf_df = df.copy()
    transf_df["var_A"] = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]

    # test transform output
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]], check_dtype=False)
    assert X["var_A"].dtypes == int


@pytest.mark.parametrize(
    "unseen", ["empanada", False, 1, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_unseen_not_permitted_value(unseen):
    with pytest.raises(ValueError):
        OrdinalEncoder(unseen=unseen)


def test_inverse_transform_when_no_unseen():
    df = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    enc = OrdinalEncoder(encoding_method="arbitrary")
    enc.fit(df)
    dft = enc.transform(df)
    pd.testing.assert_frame_equal(enc.inverse_transform(dft), df)


def test_inverse_transform_when_ignore_unseen():
    df1 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    df3 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", nan]})
    enc = OrdinalEncoder(encoding_method="arbitrary", unseen="ignore")
    enc.fit(df1)
    dft = enc.transform(df2)
    pd.testing.assert_frame_equal(enc.inverse_transform(dft), df3)


def test_inverse_transform_when_encode_unseen():
    df1 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    df2 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "frog"]})
    df3 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", nan]})
    enc = OrdinalEncoder(encoding_method="arbitrary", unseen="encode")
    enc.fit(df1)
    dft = enc.transform(df2)
    pd.testing.assert_frame_equal(enc.inverse_transform(dft), df3)


def test_inverse_transform_raises_non_fitted_error():
    df1 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    enc = OrdinalEncoder(encoding_method="arbitrary")

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1)

    df1.loc[len(df1) - 1] = nan

    with pytest.raises(ValueError):
        enc.fit(df1)

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1)


def test_encoding_new_categories(df_enc):
    df_unseen = pd.DataFrame({"var_A": ["D"], "var_B": ["D"]})
    encoder = OrdinalEncoder(encoding_method="arbitrary", unseen="encode")
    encoder.fit(df_enc[["var_A", "var_B"]])
    df_transformed = encoder.transform(df_unseen)
    assert (df_transformed == -1).all(axis=None)
