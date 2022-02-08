import pandas as pd
import pytest

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


def test_error_if_encoding_method_not_allowed():
    with pytest.raises(ValueError):
        OrdinalEncoder(encoding_method="other")


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
        encoder = OrdinalEncoder(errors="ignore")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[0] == msg

    # check for error when rare_labels equals 'raise'
    with pytest.raises(ValueError) as record:
        encoder = OrdinalEncoder(errors="raise")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])

    # check that the error message matches
    assert str(record.value) == msg


def test_fit_raises_error_if_df_contains_na(df_enc_na):
    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = OrdinalEncoder(encoding_method="arbitrary")
        encoder.fit(df_enc_na)


def test_transform_raises_error_if_df_contains_na(df_enc, df_enc_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = OrdinalEncoder(encoding_method="arbitrary")
        encoder.fit(df_enc)
        encoder.transform(df_enc_na)


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


def test_error_if_rare_labels_not_permitted_value():
    with pytest.raises(ValueError):
        OrdinalEncoder(errors="empanada")
