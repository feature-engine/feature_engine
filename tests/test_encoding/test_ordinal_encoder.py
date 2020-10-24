import pandas as pd
import pytest
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
    assert encoder.encoder_dict_ == {"var_A": {"A": 1, "B": 0, "C": 2}}
    assert encoder.input_shape_ == (20, 2)
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
    assert encoder.variables == ["var_A", "var_B"]
    # test fit attr
    assert encoder.encoder_dict_ == {
        "var_A": {"A": 0, "B": 1, "C": 2},
        "var_B": {"A": 0, "B": 1, "C": 2},
    }
    assert encoder.input_shape_ == (20, 3)
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
    with pytest.warns(UserWarning):
        encoder = OrdinalEncoder(encoding_method="arbitrary")
        encoder.fit(df_enc)
        encoder.transform(df_enc_rare)


def test_non_fitted_error(df_enc):
    with pytest.raises(NotFittedError):
        imputer = OrdinalEncoder()
        imputer.transform(df_enc)


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
