import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import MeanEncoder, OrdinalEncoder


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
    assert encoder.encoder_dict_ == {
        "var_A": {"A": 0.3333333333333333, "B": 0.2, "C": 0.5}
    }
    assert encoder.input_shape_ == (20, 2)
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
    assert encoder.variables == ["var_A", "var_B"]
    # test fit attr
    assert encoder.encoder_dict_ == {
        "var_A": {"A": 0.3333333333333333, "B": 0.2, "C": 0.5},
        "var_B": {"A": 0.2, "B": 0.3333333333333333, "C": 0.5},
    }
    assert encoder.input_shape_ == (20, 2)
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_error_if_y_not_passed_to_fit(df_enc):
    # test case 3: raises error if target is not passed
    with pytest.raises(TypeError):
        encoder = MeanEncoder()
        encoder.fit(df_enc)


def test_warning_if_transform_df_contains_categories_not_present_in_fit_df(
    df_enc, df_enc_rare
):
    # test case 4: when dataset to be transformed contains categories not present
    # in training dataset
    with pytest.warns(UserWarning):
        encoder = MeanEncoder()
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])


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


def test_raises_non_fitted_error(df_enc):
    with pytest.raises(NotFittedError):
        imputer = OrdinalEncoder()
        imputer.transform(df_enc)
