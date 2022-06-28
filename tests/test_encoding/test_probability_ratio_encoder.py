import pandas as pd
import pytest

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
    assert encoder.variables_ == ["var_A"]
    assert encoder.encoder_dict_ == {
        "var_A": {"A": 0.49999999999999994, "B": 0.25, "C": 1.0}
    }
    assert encoder.n_features_in_ == 2
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
    assert encoder.variables is None
    # fit params
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": {"A": -0.6931471805599454, "B": -1.3862943611198906, "C": 0.0},
        "var_B": {"A": -1.3862943611198906, "B": -0.6931471805599454, "C": 0.0},
    }
    assert encoder.n_features_in_ == 2
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


def test_logratio_on_numerical_variables(df_enc_numeric):
    # test ignore_format
    encoder = PRatioEncoder(
        encoding_method="log_ratio", variables=None, ignore_format=True
    )

    encoder.fit(df_enc_numeric[["var_A", "var_B"]], df_enc_numeric["target"])
    X = encoder.transform(df_enc_numeric[["var_A", "var_B"]])

    # transformed dataframe
    transf_df = df_enc_numeric.copy()
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
    assert encoder.variables is None
    # fit params
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": {1: -0.6931471805599454, 2: -1.3862943611198906, 3: 0.0},
        "var_B": {1: -1.3862943611198906, 2: -0.6931471805599454, 3: 0.0},
    }
    assert encoder.n_features_in_ == 2
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])

    # test error raise
    with pytest.raises(ValueError):
        PRatioEncoder(encoding_method="other")


def test_warn_if_transform_df_contains_categories_not_seen_in_fit(df_enc, df_enc_rare):
    # test case 3: when dataset to be transformed contains categories not present
    # in training dataset
    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for warning when rare_labels equals 'ignore'
    with pytest.warns(UserWarning) as record:
        encoder = PRatioEncoder(errors="ignore")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[0] == msg

    # check for error when rare_labels equals 'raise'
    with pytest.raises(ValueError) as record:
        encoder = PRatioEncoder(errors="raise")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])

    # check that the error message matches
    assert str(record.value) == msg


def test_error_if_rare_labels_not_permitted_value():
    with pytest.raises(ValueError):
        PRatioEncoder(errors="empanada")
