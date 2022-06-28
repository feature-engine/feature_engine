import pandas as pd
import pytest

from feature_engine.encoding import WoEEncoder


def test_automatically_select_variables(df_enc):

    # test case 1: automatically select variables, woe
    encoder = WoEEncoder(variables=None)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    # transformed dataframe
    transf_df = df_enc.copy()
    transf_df["var_A"] = [
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        0.8472978603872037,
        0.8472978603872037,
        0.8472978603872037,
        0.8472978603872037,
    ]
    transf_df["var_B"] = [
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.8472978603872037,
        0.8472978603872037,
        0.8472978603872037,
        0.8472978603872037,
    ]

    # init params
    assert encoder.variables is None
    # fit params
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": {
            "A": 0.15415067982725836,
            "B": -0.5389965007326869,
            "C": 0.8472978603872037,
        },
        "var_B": {
            "A": -0.5389965007326869,
            "B": 0.15415067982725836,
            "C": 0.8472978603872037,
        },
    }
    assert encoder.n_features_in_ == 2
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_warn_if_transform_df_contains_categories_not_seen_in_fit(df_enc, df_enc_rare):
    # test case 3: when dataset to be transformed contains categories not present
    # in training dataset
    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for error when rare_labels equals 'raise'
    with pytest.warns(UserWarning) as record:
        encoder = WoEEncoder(errors="ignore")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[0] == msg

    # check for error when rare_labels equals 'raise'
    with pytest.raises(ValueError) as record:
        encoder = WoEEncoder(errors="raise")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])

    # check that the error message matches
    assert str(record.value) == msg


def test_error_if_target_not_binary():
    # test case 4: the target is not binary
    encoder = WoEEncoder(variables=None)
    with pytest.raises(ValueError):
        df = {
            "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
            "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
            "target": [1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        }
        df = pd.DataFrame(df)
        encoder.fit(df[["var_A", "var_B"]], df["target"])


def test_error_if_denominator_probability_is_zero():
    # test case 5: when the denominator probability is zero
    encoder = WoEEncoder(variables=None)
    with pytest.raises(ValueError):
        df = {
            "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
            "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
            "target": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        }
        df = pd.DataFrame(df)
        encoder.fit(df[["var_A", "var_B"]], df["target"])

    # # # test case 6: when the numerator probability is zero, woe
    # # with pytest.raises(ValueError):
    # #     df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
    # #           'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
    # #           'target': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1,
    # 1, 0, 0]}
    # #     df = pd.DataFrame(df)
    # #     encoder.fit(df[['var_A', 'var_B']], df['target'])
    #
    # # # test case 7: when the denominator probability is zero, woe
    # # with pytest.raises(ValueError):
    # #     df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
    # #           'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
    # #           'target': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,
    # 0, 0]}
    # #     df = pd.DataFrame(df)
    # #     encoder.fit(df[['var_A', 'var_B']], df['target'])


def test_error_if_contains_na_in_fit(df_enc_na):
    # test case 9: when dataset contains na, fit method
    encoder = WoEEncoder(variables=None)
    with pytest.raises(ValueError):
        encoder.fit(df_enc_na[["var_A", "var_B"]], df_enc_na["target"])


def test_error_if_df_contains_na_in_transform(df_enc, df_enc_na):
    # test case 10: when dataset contains na, transform method}
    encoder = WoEEncoder(variables=None)
    with pytest.raises(ValueError):
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_na)


def test_on_numerical_variables(df_enc_numeric):

    # ignore_format=True
    encoder = WoEEncoder(variables=None, ignore_format=True)
    encoder.fit(df_enc_numeric[["var_A", "var_B"]], df_enc_numeric["target"])
    X = encoder.transform(df_enc_numeric[["var_A", "var_B"]])

    # transformed dataframe
    transf_df = df_enc_numeric.copy()
    transf_df["var_A"] = [
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        0.8472978603872037,
        0.8472978603872037,
        0.8472978603872037,
        0.8472978603872037,
    ]
    transf_df["var_B"] = [
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.8472978603872037,
        0.8472978603872037,
        0.8472978603872037,
        0.8472978603872037,
    ]

    # init params
    assert encoder.variables is None
    # fit params
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": {
            1: 0.15415067982725836,
            2: -0.5389965007326869,
            3: 0.8472978603872037,
        },
        "var_B": {
            1: -0.5389965007326869,
            2: 0.15415067982725836,
            3: 0.8472978603872037,
        },
    }
    assert encoder.n_features_in_ == 2
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_variables_cast_as_category(df_enc_category_dtypes):
    df = df_enc_category_dtypes.copy()
    encoder = WoEEncoder(variables=None)
    encoder.fit(df[["var_A", "var_B"]], df["target"])
    X = encoder.transform(df[["var_A", "var_B"]])

    # transformed dataframe
    transf_df = df.copy()
    transf_df["var_A"] = [
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        0.8472978603872037,
        0.8472978603872037,
        0.8472978603872037,
        0.8472978603872037,
    ]
    transf_df["var_B"] = [
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        -0.5389965007326869,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.15415067982725836,
        0.8472978603872037,
        0.8472978603872037,
        0.8472978603872037,
        0.8472978603872037,
    ]

    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]], check_dtype=False)
    assert X["var_A"].dtypes == float


def test_error_if_rare_labels_not_permitted_value():
    with pytest.raises(ValueError):
        WoEEncoder(errors="empanada")
