import math

import numpy as np
import pandas as pd
import pytest

from feature_engine.selection import SelectByInformationValue


@pytest.mark.parametrize("_threshold", ["python", (True, False), [4.3, 3]])
def test_error_when_not_permitted_threshold(_threshold):
    with pytest.raises(ValueError):
        SelectByInformationValue(threshold=_threshold)


@pytest.mark.parametrize("_bins", ["python", (True, False), 4.3, -1])
def test_error_when_not_permitted_bins(_bins):
    with pytest.raises(ValueError):
        SelectByInformationValue(bins=_bins)


@pytest.mark.parametrize("_strategy", ["python", (True, False), 4.3, -1])
def test_error_when_not_permitted_strategy(_strategy):
    with pytest.raises(ValueError):
        SelectByInformationValue(strategy=_strategy)


def test_raises_error_when_target_not_binary(df_enc_numeric):
    transformer = SelectByInformationValue()
    with pytest.raises(ValueError):
        transformer.fit(df_enc_numeric[["var_A", "target"]], df_enc_numeric["var_B"])


def test_calculate_iv_method():
    # values taken from here:
    # https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    pos = np.array(
        [
            0.05379574,
            0.12288367,
            0.134352813,
            0.163025669,
            0.166302567,
            0.158929547,
            0.105406881,
            0.045057346,
            0.050245767,
        ]
    )

    neg = np.array(
        [
            0.059171598,
            0.100591716,
            0.115384615,
            0.150887574,
            0.159763314,
            0.162721893,
            0.121301775,
            0.068047337,
            0.062130178,
        ]
    )
    woe = np.log(pos / neg)
    sel = SelectByInformationValue()
    assert sel._calculate_iv(pos, neg, woe) == 0.0233856621001144


def test_iv_dictionary(df_enc):
    sel = SelectByInformationValue().fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    expected_dict = {"var_A": 0.29706307738283366, "var_B": 0.29706307738283366}
    assert sel.information_values_ == expected_dict


def test_transformer_with_default_params():
    df = pd.DataFrame(
        {
            "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
            "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
            "var_C": ["X"] * 7 + ["Y"] * 5 + ["Z"] * 8,
            "var_D": ["L"] * 3 + ["M"] * 9 + ["N"] * 8,
            "var_E": ["R"] * 7 + ["S"] * 4 + ["T"] * 9,
            "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        }
    )
    X = df.drop("target", axis=1).copy()
    y = df["target"].copy()

    sel = SelectByInformationValue()
    sel.fit(df.drop("target", axis=1), df["target"])
    X_tr = sel.fit_transform(X, y)

    exp_dict = {
        "var_A": 0.29706307738283366,
        "var_B": 0.29706307738283366,
        "var_C": 0.07817653204775647,
        "var_D": 0.494962117149986,
        "var_E": 0.024620803988822354,
    }

    features_to_drop = ["var_C", "var_E"]
    exp_df = X.drop(features_to_drop, axis=1)

    for key in exp_dict.keys():
        assert math.isclose(exp_dict[key], sel.information_values_[key])
    assert sel.features_to_drop_ == features_to_drop
    assert X_tr.equals(exp_df)


def test_transformer_with_numerical_and_categorical_variables(df_enc):
    df = pd.DataFrame(
        {
            "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
            "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
            "var_C": np.linspace(0, 20, num=20),
            "var_D": np.linspace(0, 20, num=20),
            "var_E": ["R"] * 7 + ["S"] * 4 + ["T"] * 9,
            "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        }
    )
    X = df.drop("target", axis=1).copy()
    y = df["target"].copy()

    sel = SelectByInformationValue(bins=3)
    sel.fit(df.drop("target", axis=1), df["target"])
    X_tr = sel.fit_transform(X, y)

    exp_dict = {
        "var_A": 0.29706307738283366,
        "var_B": 0.29706307738283366,
        "var_C": 0.010625883395914762,
        "var_D": 0.010625883395914762,
        "var_E": 0.024620803988822354,
    }

    features_to_drop = ["var_C", "var_D", "var_E"]
    exp_df = X.drop(features_to_drop, axis=1)

    for key in exp_dict.keys():
        assert math.isclose(exp_dict[key], sel.information_values_[key])
    assert sel.features_to_drop_ == features_to_drop
    assert X_tr.equals(exp_df)


def test_transformer_with_equal_frequency_discretization(df_enc):
    df = pd.DataFrame(
        {
            "var_C": np.linspace(0, 20, num=20),
            "var_D": np.linspace(0, 20, num=20),
            "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        }
    )
    X = df.drop("target", axis=1).copy()
    y = df["target"].copy()

    sel = SelectByInformationValue(bins=3, strategy="equal_frequency")
    sel.fit(df.drop("target", axis=1), df["target"])
    sel.fit(X, y)

    exp_dict = {"var_C": 0.010625883395914762, "var_D": 0.010625883395914762}

    for key in exp_dict.keys():
        assert math.isclose(exp_dict[key], sel.information_values_[key])
