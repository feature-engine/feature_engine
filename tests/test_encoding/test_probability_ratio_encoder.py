import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import PRatioEncoder


def test_ratio_with_one_variable(df_enc):
    # test case 1: 1 variable, ratio
    encoder = PRatioEncoder(encoding_method="ratio", variables=["var_A"])
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    # transformed dataframe
    transf_df = df_enc.copy()
    transf_df["var_A"] = [
        0.49999999999999994,
        0.49999999999999994,
        0.49999999999999994,
        0.49999999999999994,
        0.49999999999999994,
        0.49999999999999994,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    # init params
    assert encoder.encoding_method == "ratio"
    assert encoder.variables == ["var_A"]
    # fit params
    assert encoder.encoder_dict_ == {
        "var_A": {"A": 0.49999999999999994, "B": 0.25, "C": 1.0}
    }
    assert encoder.input_shape_ == (20, 2)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_logratio_and_automaticcally_select_variables(df_enc):
    # test case 2: automatically select variables, log_ratio
    encoder = PRatioEncoder(encoding_method="log_ratio", variables=None)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    # transformed dataframe
    transf_df = df_enc.copy()
    transf_df["var_A"] = [
        -0.6931471805599454,
        -0.6931471805599454,
        -0.6931471805599454,
        -0.6931471805599454,
        -0.6931471805599454,
        -0.6931471805599454,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    transf_df["var_B"] = [
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -1.3862943611198906,
        -0.6931471805599454,
        -0.6931471805599454,
        -0.6931471805599454,
        -0.6931471805599454,
        -0.6931471805599454,
        -0.6931471805599454,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    # init params
    assert encoder.encoding_method == "log_ratio"
    assert encoder.variables == ["var_A", "var_B"]
    # fit params
    assert encoder.encoder_dict_ == {
        "var_A": {"A": -0.6931471805599454, "B": -1.3862943611198906, "C": 0.0},
        "var_B": {"A": -1.3862943611198906, "B": -0.6931471805599454, "C": 0.0},
    }
    assert encoder.input_shape_ == (20, 2)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])

    # test error raise
    with pytest.raises(ValueError):
        PRatioEncoder(encoding_method="other")


def test_ratio_error_if_denominator_probability_zero():
    # test case 3: when the denominator probability is zero, ratio
    with pytest.raises(ValueError):
        df = {
            "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
            "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
            "target": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        }
        df = pd.DataFrame(df)
        encoder = PRatioEncoder(encoding_method="ratio")
        encoder.fit(df[["var_A", "var_B"]], df["target"])


def test_log_ratio_error_if_denominator_probability_zero():
    # test case 4: when the denominator probability is zero, log_ratio
    with pytest.raises(ValueError):
        df = {
            "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
            "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
            "target": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        }
        df = pd.DataFrame(df)
        encoder = PRatioEncoder(encoding_method="log_ratio")
        encoder.fit(df[["var_A", "var_B"]], df["target"])


def test_logratio_error_if_numerator_probability_zero():
    # test case 5: when the numerator probability is zero, only applies to log_ratio
    with pytest.raises(ValueError):
        df = {
            "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
            "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        }
        df = pd.DataFrame(df)
        encoder = PRatioEncoder(encoding_method="log_ratio")
        encoder.fit(df[["var_A", "var_B"]], df["target"])


def test_raises_non_fitted_error(df_enc):
    # test case 6: non fitted error
    with pytest.raises(NotFittedError):
        imputer = PRatioEncoder()
        imputer.transform(df_enc)


def test_error_if_df_contains_na_in_fit(df_enc_na):
    # test case 7: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = PRatioEncoder(encoding_method="ratio")
        encoder.fit(df_enc_na[["var_A", "var_B"]], df_enc_na["target"])


def test_error_if_df_contains_na_in_transform(df_enc, df_enc_na):
    # test case 8: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = PRatioEncoder(encoding_method="ratio")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_na)
