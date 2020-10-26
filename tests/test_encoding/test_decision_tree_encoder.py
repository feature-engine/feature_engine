import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import DecisionTreeEncoder


def test_encoding_method_param(df_enc):
    # defaults
    encoder = DecisionTreeEncoder()
    encoder.fit(df_enc, df_enc["target"])
    assert encoder.encoder_[0].encoding_method == "arbitrary"

    # ordered encoding
    encoder = DecisionTreeEncoder(encoding_method="ordered")
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    assert encoder.encoder_[0].encoding_method == "ordered"

    # incorrect input
    with pytest.raises(ValueError):
        encoder = DecisionTreeEncoder(encoding_method="other")
        encoder.fit(df_enc, df_enc["target"])


def test_classification(df_enc):
    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    transf_df = df_enc.copy()
    transf_df["var_A"] = [0.25] * 16 + [0.5] * 4  # Tree: var_A <= 1.5 -> 0.25 else 0.5
    transf_df["var_B"] = [0.2] * 10 + [0.4] * 10  # Tree: var_B <= 0.5 -> 0.2 else 0.4
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_regression(df_enc):
    random = np.random.RandomState(42)
    y = random.normal(0, 0.1, len(df_enc))
    encoder = DecisionTreeEncoder(regression=True, random_state=random)
    encoder.fit(df_enc[["var_A", "var_B"]], y)
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    transf_df = df_enc.copy()
    transf_df["var_A"] = (
        [0.034348] * 6 + [-0.024679] * 10 + [-0.075473] * 4
    )  # Tree: var_A <= 1.5 -> 0.25 else 0.5
    transf_df["var_B"] = [0.044806] * 10 + [-0.079066] * 10
    pd.testing.assert_frame_equal(X.round(6), transf_df[["var_A", "var_B"]])


def test_non_fitted_error(df_enc):
    with pytest.raises(NotFittedError):
        encoder = DecisionTreeEncoder()
        encoder.transform(df_enc)


def test_fit_raises_error_if_df_contains_na(df_enc_na):
    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = DecisionTreeEncoder()
        encoder.fit(df_enc_na)


def test_transform_raises_error_if_df_contains_na(df_enc, df_enc_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = DecisionTreeEncoder()
        encoder.fit(df_enc_na[["var_A", "var_B"]], df_enc_na["target"])
        encoder.transform(df_enc_na)
