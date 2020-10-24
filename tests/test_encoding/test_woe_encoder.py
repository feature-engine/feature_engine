import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

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
    assert encoder.variables == ["var_A", "var_B"]
    # fit params
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
    assert encoder.input_shape_ == (20, 2)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_error_target_is_not_passed(df_enc):
    # test case 2: raises error if target is  not passed
    encoder = WoEEncoder(variables=None)
    with pytest.raises(TypeError):
        encoder.fit(df_enc)


def test_warn_if_transform_df_contains_categories_not_seen_in_fit(df_enc, df_enc_rare):
    # test case 3: when dataset to be transformed contains categories not present
    # in training dataset
    encoder = WoEEncoder(variables=None)
    with pytest.warns(UserWarning):
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])


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


def test_non_fitted_error(df_enc):
    # test case 8: non fitted error
    with pytest.raises(NotFittedError):
        imputer = WoEEncoder()
        imputer.transform(df_enc)


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
