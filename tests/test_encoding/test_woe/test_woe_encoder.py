import math
import re

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
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

DF_ENC = {
    "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
    "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
    "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
}

DF_ENC_NUMERIC = {
    "var_A": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
    "var_B": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
    "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
}

DF_ENC_RARE = {
    "var_A": ["B"] * 9 + ["A"] * 6 + ["C"] * 4 + ["D"] * 1,
    "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
    "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
}

# None (not np.nan) is what both pandas and polars accept as a missing
# value inside a string column literal.
DF_ENC_NA = {
    "var_A": [None] + ["B"] * 8 + ["A"] * 6 + ["C"] * 4 + ["D"] * 1,
    "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
    "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
}


def _none_to_nan(values):
    # Missing values print as None for polars, NaN for pandas float columns
    # - both mean "missing" here, so normalize both sides before comparing.
    return [np.nan if v is None else v for v in values]


def assert_df_equal(X, expected: dict, abs_tol: float = 1e-5) -> None:
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(expected.keys())
    for col, values in expected.items():
        assert _none_to_nan(result[col]) == pytest.approx(
            _none_to_nan(values), abs=abs_tol, nan_ok=True
        )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_automatically_select_variables(make_df):
    df_enc = make_df(DF_ENC)
    encoder = WoEEncoder(variables=None)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    X = encoder.transform(df_enc[["var_A", "var_B"]])

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
    assert_df_equal(X, {"var_A": VAR_A, "var_B": VAR_B})


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_user_passes_variables(make_df):
    df_enc = make_df(DF_ENC)
    encoder = WoEEncoder(variables=["var_A", "var_B"])
    encoder.fit(df_enc, df_enc["target"])
    X = encoder.transform(df_enc)

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
    assert_df_equal(
        X, {"var_A": VAR_A, "var_B": VAR_B, "target": DF_ENC["target"]}
    )


_targets = [
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0],
    [1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1],
    [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1],
]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("target", _targets)
def test_when_target_class_not_0_1(make_df, target):
    data = dict(DF_ENC)
    data["target"] = target
    df_enc = make_df(data)
    encoder = WoEEncoder(variables=["var_A", "var_B"])
    encoder.fit(df_enc, df_enc["target"])
    X = encoder.transform(df_enc)

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
    assert_df_equal(X, {"var_A": VAR_A, "var_B": VAR_B, "target": target})


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_warn_if_transform_df_contains_categories_not_seen_in_fit(make_df):
    df_enc = make_df(DF_ENC)
    df_enc_rare = make_df(DF_ENC_RARE)
    # test case 3: when dataset to be transformed contains categories not present
    # in training dataset
    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for error when rare_labels equals 'raise'
    with pytest.warns(UserWarning) as record:
        encoder = WoEEncoder(unseen="ignore")
        encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
        encoder.transform(df_enc_rare[["var_A", "var_B"]])

    # check that at least one warning was raised (Pandas 3 may emit additional
    # deprecation warnings)
    assert len(record) >= 1
    # check that the message matches
    assert any(r.message.args[0] == msg for r in record)

    # check for error when rare_labels equals 'raise'
    encoder = WoEEncoder(unseen="raise")
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.transform(df_enc_rare[["var_A", "var_B"]])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_target_not_binary(make_df):
    # test case 4: the target is not binary
    encoder = WoEEncoder(variables=None)
    with pytest.raises(ValueError):
        df = {
            "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
            "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
            "target": [1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        }
        df = make_df(df)
        encoder.fit(df[["var_A", "var_B"]], df["target"])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_denominator_probability_is_zero_1_var(make_df):
    df = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = make_df(df)
    encoder = WoEEncoder(variables=None)

    msg = (
        "During the WoE calculation, some of the categories in the "
        "following features contained 0 in the denominator or numerator, "
        "and hence the WoE can't be calculated: var_A."
    )
    with pytest.raises(ValueError, match=msg):
        encoder.fit(df[["var_A", "var_B"]], df["target"])

    df = {
        "var_A": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "var_B": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "target": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = make_df(df)
    encoder = WoEEncoder(variables=None)

    msg = (
        "During the WoE calculation, some of the categories in the "
        "following features contained 0 in the denominator or numerator, "
        "and hence the WoE can't be calculated: var_B."
    )
    with pytest.raises(ValueError, match=msg):
        encoder.fit(df[["var_A", "var_B"]], df["target"])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_denominator_probability_is_zero_2_vars(make_df):
    df = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "var_C": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "target": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = make_df(df)
    encoder = WoEEncoder(variables=None)

    msg = (
        "During the WoE calculation, some of the categories in the "
        "following features contained 0 in the denominator or numerator, "
        "and hence the WoE can't be calculated: var_A, var_C."
    )
    with pytest.raises(ValueError, match=msg):
        encoder.fit(df, df["target"])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_numerator_probability_is_zero(make_df):
    df = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "var_C": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = make_df(df)
    encoder = WoEEncoder(variables=None)

    msg = (
        "During the WoE calculation, some of the categories in the "
        "following features contained 0 in the denominator or numerator, "
        "and hence the WoE can't be calculated: var_A, var_C."
    )
    with pytest.raises(ValueError, match=msg):
        encoder.fit(df, df["target"])

    msg = (
        "During the WoE calculation, some of the categories in the "
        "following features contained 0 in the denominator or numerator, "
        "and hence the WoE can't be calculated: var_A."
    )
    with pytest.raises(ValueError, match=msg):
        encoder.fit(df[["var_A", "var_B"]], df["target"])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_fill_value(make_df):
    df = {
        "var_A": ["A"] * 9 + ["B"] * 6 + ["C"] * 3 + ["D"] * 2,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    }
    df = make_df(df)
    encoder = WoEEncoder(variables=None, fill_value=1)
    encoder.fit(df, df["target"])
    woe_exp_a = {
        "A": -0.6337237600891445,
        "B": -0.07410797215372196,
        "C": -0.8472978603872037,
        "D": 1.8718021769015913,
    }
    woe_exp_b = {
        "A": -0.7672551527136673,
        "B": 0.6190392084062234,
        "C": 0.6190392084062234,
    }
    woe_exp = {"var_A": woe_exp_a, "var_B": woe_exp_b}

    for var in ["var_A", "var_B"]:
        for k, i in woe_exp[var].items():
            assert math.isclose(encoder.encoder_dict_[var][k], woe_exp[var][k])

    encoder = WoEEncoder(variables=None, fill_value=10)
    encoder.fit(df, df["target"])
    woe_exp_a = {
        "A": -0.6337237600891445,
        "B": -0.07410797215372196,
        "C": -3.1498829533812494,
        "D": 4.174387269895637,
    }
    woe_exp = {"var_A": woe_exp_a, "var_B": woe_exp_b}
    for var in ["var_A", "var_B"]:
        for k, i in woe_exp[var].items():
            assert math.isclose(encoder.encoder_dict_[var][k], woe_exp[var][k])


@pytest.mark.parametrize("fill_value", ["hola", [10]])
def test_error_if_fill_value_not_allowed(fill_value):
    with pytest.raises(ValueError):
        WoEEncoder(fill_value=fill_value)


@pytest.mark.parametrize("fill_value", [0, 1, 10, 0.5, 0.002, None])
def test_assigns_fill_value_at_init(fill_value):
    encoder = WoEEncoder(fill_value=fill_value)
    assert encoder.fill_value == fill_value


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_contains_na_in_fit(make_df):
    # test case 9: when dataset contains na, fit method
    df_enc_na = make_df(DF_ENC_NA)
    encoder = WoEEncoder(variables=None)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=msg):
        encoder.fit(df_enc_na[["var_A", "var_B"]], df_enc_na["target"])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_df_contains_na_in_transform(make_df):
    # test case 10: when dataset contains na, transform method}
    df_enc = make_df(DF_ENC)
    df_enc_na = make_df(DF_ENC_NA)
    encoder = WoEEncoder(variables=None)
    encoder.fit(df_enc[["var_A", "var_B"]], df_enc["target"])
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=msg):
        encoder.transform(df_enc_na[["var_A", "var_B"]])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_on_numerical_variables(make_df):
    # ignore_format=True
    df_enc_numeric = make_df(DF_ENC_NUMERIC)
    encoder = WoEEncoder(variables=None, ignore_format=True)
    encoder.fit(df_enc_numeric[["var_A", "var_B"]], df_enc_numeric["target"])
    X = encoder.transform(df_enc_numeric[["var_A", "var_B"]])

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
    assert_df_equal(X, {"var_A": VAR_A, "var_B": VAR_B})


def test_variables_cast_as_category():
    # pandas Categorical dtype has no direct polars equivalent.
    df = pd.DataFrame(DF_ENC)
    df[["var_A", "var_B"]] = df[["var_A", "var_B"]].astype("category")
    encoder = WoEEncoder(variables=None)
    encoder.fit(df[["var_A", "var_B"]], df["target"])
    X = encoder.transform(df[["var_A", "var_B"]])

    transf_df = df.copy()
    transf_df["var_A"] = VAR_A
    transf_df["var_B"] = VAR_B

    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]], check_dtype=False)
    assert X["var_A"].dtypes.name == "float64"


@pytest.mark.parametrize(
    "errors", ["empanada", False, 1, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_rare_labels_not_permitted_value(errors):
    with pytest.raises(ValueError):
        WoEEncoder(unseen=errors)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_inverse_transform_raises_non_fitted_error(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    enc = WoEEncoder()

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1)

    df1_na = make_df({"words": ["dog", "dog", "cat", "cat", "cat", None]})

    with pytest.raises(ValueError):
        enc.fit(df1_na, make_df({"target": [0, 1, 0, 1, 1, 0]})["target"])

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        enc.inverse_transform(df1_na)
