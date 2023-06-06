import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import WoEEncoder

VAR_A = [
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

VAR_B = [
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


def test_automatically_select_variables(df_enc):
    encoder = WoEEncoder(variables=None)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

    # transformed dataframe
    transf_df = df_enc.copy()
    transf_df["var_A"] = VAR_A
    transf_df["var_B"] = VAR_B

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
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]])


def test_user_passes_variables(df_enc):
    encoder = WoEEncoder(variables=["var_A", "var_B"])
    encoder.fit(df_enc, df_enc["target"])
    X = encoder.transform(df_enc)

    # transformed dataframe
    transf_df = df_enc.copy()
    transf_df["var_A"] = VAR_A
    transf_df["var_B"] = VAR_B

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
    pd.testing.assert_frame_equal(X, transf_df)


_targets = [
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0],
    [1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1],
    [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1],
]


@pytest.mark.parametrize("target", _targets)
def test_when_target_class_not_0_1(df_enc, target):
    encoder = WoEEncoder(variables=["var_A", "var_B"])
    df_enc["target"] = target
    encoder.fit(df_enc, df_enc["target"])
    X = encoder.transform(df_enc)

    # transformed dataframe
    transf_df = df_enc.copy()
    transf_df["var_A"] = VAR_A
    transf_df["var_B"] = VAR_B

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
    pd.testing.assert_frame_equal(X, transf_df)


def test_warn_if_transform_df_contains_categories_not_seen_in_fit(df_enc, df_enc_rare):
    # test case 3: when dataset to be transformed contains categories not present
    # in training dataset
    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for error when rare_labels equals 'raise'
    with pytest.warns(UserWarning) as record:
        encoder = WoEEncoder(unseen="ignore")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[0] == msg

    # check for error when rare_labels equals 'raise'
    with pytest.raises(ValueError) as record:
        encoder = WoEEncoder(unseen="raise")
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


def test_error_if_denominator_probability_is_zero_1_var():
    df = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)
    encoder = WoEEncoder(variables=None)

    with pytest.raises(ValueError) as record:
        encoder.fit(df[["var_A", "var_B"]], df["target"])

    msg = (
        "During the WoE calculation, some of the categories in the "
        "following features contained 0 in the denominator or numerator, "
        "and hence the WoE can't be calculated: var_A."
    )
    assert str(record.value) == msg

    df = {
        "var_A": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "var_B": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "target": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)
    encoder = WoEEncoder(variables=None)

    with pytest.raises(ValueError) as record:
        encoder.fit(df[["var_A", "var_B"]], df["target"])

    msg = (
        "During the WoE calculation, some of the categories in the "
        "following features contained 0 in the denominator or numerator, "
        "and hence the WoE can't be calculated: var_B."
    )
    assert str(record.value) == msg


def test_error_if_denominator_probability_is_zero_2_vars():
    df = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "var_C": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "target": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)
    encoder = WoEEncoder(variables=None)

    with pytest.raises(ValueError) as record:
        encoder.fit(df, df["target"])

    msg = (
        "During the WoE calculation, some of the categories in the "
        "following features contained 0 in the denominator or numerator, "
        "and hence the WoE can't be calculated: var_A, var_C."
    )
    assert str(record.value) == msg


def test_error_if_numerator_probability_is_zero():
    df = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "var_C": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)
    encoder = WoEEncoder(variables=None)

    with pytest.raises(ValueError) as record:
        encoder.fit(df, df["target"])

    msg = (
        "During the WoE calculation, some of the categories in the "
        "following features contained 0 in the denominator or numerator, "
        "and hence the WoE can't be calculated: var_A, var_C."
    )
    assert str(record.value) == msg

    with pytest.raises(ValueError) as record:
        encoder.fit(df[["var_A", "var_B"]], df["target"])

    msg = (
        "During the WoE calculation, some of the categories in the "
        "following features contained 0 in the denominator or numerator, "
        "and hence the WoE can't be calculated: var_A."
    )
    assert str(record.value) == msg


def test_error_if_contains_na_in_fit(df_enc_na):
    # test case 9: when dataset contains na, fit method
    encoder = WoEEncoder(variables=None)
    with pytest.raises(ValueError) as record:
        encoder.fit(df_enc_na[["var_A", "var_B"]], df_enc_na["target"])

    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    assert str(record.value) == msg


def test_error_if_df_contains_na_in_transform(df_enc, df_enc_na):
    # test case 10: when dataset contains na, transform method}
    encoder = WoEEncoder(variables=None)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    with pytest.raises(ValueError) as record:
        encoder.transform(df_enc_na[["var_A", "var_B"]])
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    assert str(record.value) == msg


def test_on_numerical_variables(df_enc_numeric):
    # ignore_format=True
    encoder = WoEEncoder(variables=None, ignore_format=True)
    encoder.fit(df_enc_numeric[["var_A", "var_B"]], df_enc_numeric["target"])
    X = encoder.transform(df_enc_numeric[["var_A", "var_B"]])

    # transformed dataframe
    transf_df = df_enc_numeric.copy()
    transf_df["var_A"] = VAR_A
    transf_df["var_B"] = VAR_B

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
    transf_df["var_A"] = VAR_A
    transf_df["var_B"] = VAR_B

    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]], check_dtype=False)
    assert X["var_A"].dtypes == float


@pytest.mark.parametrize(
    "errors", ["empanada", False, 1, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_rare_labels_not_permitted_value(errors):
    with pytest.raises(ValueError):
        WoEEncoder(unseen=errors)


def test_inverse_transform_raises_non_fitted_error():
    df1 = pd.DataFrame({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    enc = WoEEncoder()

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1)

    df1.loc[len(df1) - 1] = np.nan

    with pytest.raises(ValueError):
        enc.fit(df1, pd.Series([0, 1, 0, 1, 1, 0]))

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1)
