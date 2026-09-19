import math
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import WoEEncoder
from tests.backend_helpers import make_series, frame_to_dict

WOE_A = {
    "A": 0.15415067982725836,
    "B": -0.5389965007326869,
    "C": 0.8472978603872037,
}
WOE_B = {
    "A": -0.5389965007326869,
    "B": 0.15415067982725836,
    "C": 0.8472978603872037,
}
VAR_A = [WOE_A["A"]] * 6 + [WOE_A["B"]] * 10 + [WOE_A["C"]] * 4
VAR_B = [WOE_B["A"]] * 10 + [WOE_B["B"]] * 6 + [WOE_B["C"]] * 4

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)


# init parameters
@pytest.mark.parametrize(
    "unseen", ["empanada", "encode", False, 1, None, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_unseen_not_permitted_value(unseen):
    msg = f"Parameter `unseen` takes only values ignore, raise. Got {unseen} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        WoEEncoder(unseen=unseen)


@pytest.mark.parametrize(
    "ignore_format, unseen",
    [(False, "ignore"), (True, "raise"), (False, "raise"), (True, "ignore")],
)
def test_init_param_assignment(ignore_format, unseen):
    encoder = WoEEncoder(ignore_format=ignore_format, unseen=unseen)
    assert encoder.ignore_format is ignore_format
    assert encoder.unseen == unseen


# fit and transform
def test_automatically_select_variables(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = WoEEncoder(variables=None)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    assert encoder.encoder_dict_ == {"var_A": WOE_A, "var_B": WOE_B}
    assert encoder.variables_with_zero_counts_ == []
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": pytest.approx(VAR_A),
        "var_B": pytest.approx(VAR_B),
    }


@pytest.mark.parametrize("to_target", [list, np.array])
def test_target_as_list_or_array(make_df, data_enc, to_target):
    # a list or numpy array target takes a different code path than a Series
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = to_target(data_enc["target"])

    encoder = WoEEncoder(variables=None)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    assert encoder.encoder_dict_ == {"var_A": WOE_A, "var_B": WOE_B}
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": pytest.approx(VAR_A),
        "var_B": pytest.approx(VAR_B),
    }


def test_user_passes_variables(make_df, data_enc):
    X = make_df(data_enc)
    y = make_series(make_df, data_enc["target"])

    encoder = WoEEncoder(variables=["var_A", "var_B"])
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    assert encoder.encoder_dict_ == {"var_A": WOE_A, "var_B": WOE_B}
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": pytest.approx(VAR_A),
        "var_B": pytest.approx(VAR_B),
        "target": data_enc["target"],
    }


_targets = [
    [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0],
    [1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1],
    [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1],
]


@pytest.mark.parametrize("target", _targets)
def test_when_target_class_not_0_1(make_df, data_enc, target):
    data = dict(data_enc)
    data["target"] = target
    X = make_df(data)
    y = make_series(make_df, target)

    encoder = WoEEncoder(variables=["var_A", "var_B"])
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    assert encoder.encoder_dict_ == {"var_A": WOE_A, "var_B": WOE_B}
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": pytest.approx(VAR_A),
        "var_B": pytest.approx(VAR_B),
        "target": target,
    }


def test_warn_if_transform_df_contains_categories_not_seen_in_fit(
    make_df, data_enc, data_enc_rare
):
    # test case 3: when dataset to be transformed contains categories not present
    # in training dataset
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])
    X_rare = make_df(data_enc_rare)[["var_A", "var_B"]]
    msg = "During the encoding, NaN values were introduced in the feature(s) var_A."

    # check for warning when unseen equals 'ignore'
    encoder = WoEEncoder(unseen="ignore")
    encoder.fit(X, y)
    with pytest.warns(UserWarning, match=re.escape(msg)):
        encoder.transform(X_rare)

    # check for error when unseen equals 'raise'
    encoder = WoEEncoder(unseen="raise")
    encoder.fit(X, y)
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.transform(X_rare)


def test_error_if_target_not_binary(make_df):
    # test case 4: the target is not binary
    data = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    X = make_df(data)[["var_A", "var_B"]]
    y = make_series(make_df, data["target"])

    encoder = WoEEncoder(variables=None)
    msg = (
        "This encoder is designed for binary classification. The target "
        "used has more than 2 unique values."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.fit(X, y)


def test_zero_counts_are_replaced_by_half(make_df):
    # in var_A, C has no negative cases and D no positive cases
    data = {
        "var_A": ["A"] * 9 + ["B"] * 6 + ["C"] * 3 + ["D"] * 2,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    }
    X = make_df(data)[["var_A", "var_B"]]
    y = make_series(make_df, data["target"])

    encoder = WoEEncoder().fit(X, y)
    Xt = encoder.transform(X)

    # 7 positive and 13 negative cases
    woe_a = {
        "A": math.log((2 / 7) / (7 / 13)),
        "B": math.log((2 / 7) / (4 / 13)),
        "C": math.log((3 / 7) / (0.5 / 13)),
        "D": math.log((0.5 / 7) / (2 / 13)),
    }
    woe_b = {
        "A": math.log((2 / 7) / (8 / 13)),
        "B": math.log((3 / 7) / (3 / 13)),
        "C": math.log((2 / 7) / (2 / 13)),
    }
    assert encoder.encoder_dict_ == {
        "var_A": pytest.approx(woe_a),
        "var_B": pytest.approx(woe_b),
    }
    assert encoder.variables_with_zero_counts_ == ["var_A"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": pytest.approx([woe_a[v] for v in data["var_A"]]),
        "var_B": pytest.approx([woe_b[v] for v in data["var_B"]]),
    }


def test_variables_with_zero_counts(make_df):
    # category A of var_A and var_C has no negative cases
    data = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "var_C": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "target": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    X = make_df(data)[["var_A", "var_B", "var_C"]]
    y = make_series(make_df, data["target"])

    encoder = WoEEncoder().fit(X, y)

    assert encoder.variables_with_zero_counts_ == ["var_A", "var_C"]


def test_error_if_contains_na_in_fit(make_df, data_enc_na):
    # test case 9: when dataset contains na, fit method
    X = make_df(data_enc_na)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc_na["target"])

    encoder = WoEEncoder(variables=None)
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        encoder.fit(X, y)


def test_error_if_df_contains_na_in_transform(make_df, data_enc, data_enc_na):
    # test case 10: when dataset contains na, transform method
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])
    X_na = make_df(data_enc_na)[["var_A", "var_B"]]

    encoder = WoEEncoder(variables=None)
    encoder.fit(X, y)
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        encoder.transform(X_na)


def test_on_numerical_variables(make_df, data_enc_numeric):
    # ignore_format=True
    X = make_df(data_enc_numeric)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc_numeric["target"])

    encoder = WoEEncoder(variables=None, ignore_format=True)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    # fit params
    assert encoder.variables_ == ["var_A", "var_B"]
    assert encoder.encoder_dict_ == {
        "var_A": {1: WOE_A["A"], 2: WOE_A["B"], 3: WOE_A["C"]},
        "var_B": {1: WOE_B["A"], 2: WOE_B["B"], 3: WOE_B["C"]},
    }
    assert encoder.n_features_in_ == 2
    # transform params
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": pytest.approx(VAR_A),
        "var_B": pytest.approx(VAR_B),
    }


def test_integer_column_names(data_enc):
    # integer column names are pandas-only
    X = pd.DataFrame({0: data_enc["var_A"], 1: data_enc["var_B"]})
    y = pd.Series(data_enc["target"])

    encoder = WoEEncoder().fit(X, y)

    assert encoder.encoder_dict_ == {0: WOE_A, 1: WOE_B}


def test_variables_cast_as_category(df_enc_category_dtypes):
    # pandas Categorical dtype has no direct polars equivalent.
    df = df_enc_category_dtypes.copy()
    encoder = WoEEncoder(variables=None)
    encoder.fit(df[["var_A", "var_B"]], df["target"])
    X = encoder.transform(df[["var_A", "var_B"]])

    transf_df = df.copy()
    transf_df["var_A"] = VAR_A
    transf_df["var_B"] = VAR_B

    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]], check_dtype=False)
    assert X["var_A"].dtypes.name == "float64"


def test_inverse_transform_raises_non_fitted_error(make_df):
    df1 = make_df({"words": ["dog", "dog", "cat", "cat", "cat", "bird"]})
    y = make_series(make_df, [0, 1, 0, 1, 1, 0])
    enc = WoEEncoder()
    msg = (
        "This WoEEncoder instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        enc.inverse_transform(df1)

    df1_na = make_df({"words": ["dog", "dog", "cat", "cat", "cat", None]})

    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        enc.fit(df1_na, y)

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        enc.inverse_transform(df1_na)
