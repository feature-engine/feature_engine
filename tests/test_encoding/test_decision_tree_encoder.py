import re

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import DecisionTreeEncoder
from tests.backend_helpers import make_series, frame_to_dict

# Tree: var_A <= 1.5 -> 0.25 else 0.5
# Tree: var_B <= 0.5 -> 0.2 else 0.4
ENCODED = {
    "var_A": [0.25] * 16 + [0.5] * 4,
    "var_B": [0.2] * 10 + [0.4] * 10,
}
ENCODED_REGRESSION = {
    "var_A": [0.034348] * 6 + [-0.024679] * 10 + [-0.075473] * 4,
    "var_B": [0.044806] * 10 + [-0.079066] * 10,
}


def _rounded(X, decimals=6):
    return {
        col: [round(v, decimals) for v in values]
        for col, values in frame_to_dict(X).items()
    }


# init parameters
@pytest.mark.parametrize(
    "enc_method",
    ["count", "Ordered", "", False, 1, None, ["ordered"], ("arbitrary",)],
)
def test_error_if_encoding_method_not_permitted_value(enc_method):
    msg = (
        "`encoding_method` takes only values 'ordered' and 'arbitrary'."
        f" Got {enc_method} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DecisionTreeEncoder(encoding_method=enc_method)


@pytest.mark.parametrize(
    "unseen", ["string", False, ("raise", "ignore"), ["ignore"], np.nan]
)
def test_error_if_unseen_gets_not_permitted_value(unseen):
    msg = (
        "Parameter `unseen` takes only values ignore, raise, encode. "
        f"Got {unseen} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DecisionTreeEncoder(unseen=unseen)


def test_error_if_unseen_is_encode_and_fill_value_is_none():
    msg = (
        "When `unseen='encode'` you need to pass a number to `fill_value`. "
        f"Got {None} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DecisionTreeEncoder(unseen="encode", fill_value=None)


@pytest.mark.parametrize("precision", ["string", 0.1, -1, np.nan])
def test_error_if_precision_gets_not_permitted_value(precision):
    msg = "Parameter `precision` takes integers or None. " f"Got {precision} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DecisionTreeEncoder(precision=precision)


@pytest.mark.parametrize(
    "params",
    [
        {
            "encoding_method": "arbitrary",
            "cv": 3,
            "scoring": "neg_mean_squared_error",
            "regression": True,
            "param_grid": None,
            "random_state": None,
            "ignore_format": True,
            "precision": 1,
            "unseen": "raise",
            "fill_value": None,
            "n_jobs": None,
        },
        {
            "encoding_method": "ordered",
            "cv": 5,
            "scoring": "roc_auc",
            "regression": False,
            "param_grid": {"max_depth": [1, 2]},
            "random_state": 0,
            "ignore_format": False,
            "precision": None,
            "unseen": "encode",
            "fill_value": 0.1,
            "n_jobs": -1,
        },
        {
            "encoding_method": "ordered",
            "cv": 2,
            "scoring": "accuracy",
            "regression": False,
            "param_grid": {"max_depth": [3]},
            "random_state": 42,
            "ignore_format": False,
            "precision": 2,
            "unseen": "ignore",
            "fill_value": 1,
            "n_jobs": 2,
        },
    ],
)
def test_init_param_assignment(params):
    encoder = DecisionTreeEncoder(**params)
    for param, value in params.items():
        assert getattr(encoder, param) == value


# fit attributes
def test_encoding_dictionary(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(X, y)

    # Tree: var_A <= 1.5 -> 0.25 else 0.5
    # Tree: var_B <= 0.5 -> 0.2 else 0.4
    expected_encodings = {
        "var_A": {"A": 0.25, "B": 0.25, "C": 0.5},
        "var_B": {"A": 0.2, "B": 0.4, "C": 0.4},
    }
    assert encoder.encoder_dict_ == expected_encodings


def test_ordered_encoding_dictionary(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = DecisionTreeEncoder(regression=False, encoding_method="ordered")
    encoder.fit(X, y)

    # codes by target mean: var_A B=0, A=1, C=2 and var_B A=0, B=1, C=2
    # both trees split code 0 from the rest
    expected_encodings = {
        "var_A": {"B": 0.2, "A": 0.4, "C": 0.4},
        "var_B": {"A": 0.2, "B": 0.4, "C": 0.4},
    }
    assert encoder.encoder_dict_ == expected_encodings


def test_precision(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = DecisionTreeEncoder(regression=False, precision=1)
    encoder.fit(X, y)

    # Tree: var_A <= 1.5 -> 0.25 else 0.5
    # Tree: var_B <= 0.5 -> 0.2 else 0.4
    expected_encodings = {
        "var_A": {"A": 0.2, "B": 0.2, "C": 0.5},
        "var_B": {"A": 0.2, "B": 0.4, "C": 0.4},
    }
    assert encoder.encoder_dict_ == expected_encodings


def test_classification(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == ENCODED


@pytest.mark.parametrize("to_target", [list, np.array])
def test_target_as_list_or_array(make_df, data_enc, to_target):
    # a list or numpy array target takes a different code path than a Series
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = to_target(data_enc["target"])

    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == ENCODED


def test_regression(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    random = np.random.RandomState(42)
    y = make_series(make_df, random.normal(0, 0.1, len(data_enc["target"])))
    encoder = DecisionTreeEncoder(
        regression=True,
        random_state=random,
    )
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    assert isinstance(Xt, make_df)
    assert _rounded(Xt) == ENCODED_REGRESSION


def test_fit_raises_error_if_df_contains_na(make_df, data_enc_na):
    X = make_df(data_enc_na)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc_na["target"])

    encoder = DecisionTreeEncoder(regression=False)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.fit(X, y)


def test_transform_raises_error_if_df_contains_na(make_df, data_enc, data_enc_na):
    X = make_df(data_enc)[["var_A", "var_B"]]
    X_na = make_df(data_enc_na)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(X, y)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.transform(X_na)


def test_classification_ignore_format(make_df, data_enc_numeric):
    X = make_df(data_enc_numeric)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc_numeric["target"])

    encoder = DecisionTreeEncoder(
        regression=False,
        ignore_format=True,
    )
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == ENCODED


def test_regression_ignore_format(make_df, data_enc_numeric):
    X = make_df(data_enc_numeric)[["var_A", "var_B"]]
    random = np.random.RandomState(42)
    y = make_series(make_df, random.normal(0, 0.1, len(data_enc_numeric["target"])))
    encoder = DecisionTreeEncoder(
        regression=True,
        random_state=random,
        ignore_format=True,
    )
    encoder.fit(X, y)
    Xt = encoder.transform(X)

    assert isinstance(Xt, make_df)
    assert _rounded(Xt) == ENCODED_REGRESSION


def test_variables_cast_as_category(df_enc_category_dtypes):
    # pandas Categorical dtype has no direct polars equivalent - pandas-only.
    df = df_enc_category_dtypes.copy()
    encoder = DecisionTreeEncoder(regression=False)
    encoder.fit(df[["var_A", "var_B"]], df["target"])
    X = encoder.transform(df[["var_A", "var_B"]])

    transf_df = df.copy()
    transf_df["var_A"] = [0.25] * 16 + [0.5] * 4  # Tree: var_A <= 1.5 -> 0.25 else 0.5
    transf_df["var_B"] = [0.2] * 10 + [0.4] * 10  # Tree: var_B <= 0.5 -> 0.2 else 0.4
    pd.testing.assert_frame_equal(X, transf_df[["var_A", "var_B"]], check_dtype=False)
    assert X["var_A"].dtypes == float


def test_integer_column_names(data_enc):
    # integer column names are pandas-only
    X = pd.DataFrame({0: data_enc["var_A"], 1: data_enc["var_B"]})
    y = pd.Series(data_enc["target"])

    encoder = DecisionTreeEncoder(regression=False).fit(X, y)
    expected = DecisionTreeEncoder(regression=False).fit(X.rename(columns=str), y)

    assert encoder.encoder_dict_ == {
        0: expected.encoder_dict_["0"],
        1: expected.encoder_dict_["1"],
    }


def test_error_when_regression_is_true_and_target_is_binary(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = DecisionTreeEncoder(regression=True)
    msg = (
        "Trying to fit a regression to a binary target is not "
        "allowed by this transformer. Check the target values "
        "or set regression to False."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.fit(X, y)


def test_error_when_regression_is_false_and_target_is_continuous(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    random = np.random.RandomState(42)
    y = make_series(make_df, random.normal(0, 10, len(data_enc["target"])))
    encoder = DecisionTreeEncoder(regression=False)
    msg = (
        "Unknown label type: continuous. Maybe you are trying to fit a classifier, "
        "which expects discrete classes on a regression target with continuous values."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.fit(X, y)


@pytest.mark.parametrize(
    "grid",
    [None, {"max_depth": [1, 2, 3]}, {"max_depth": [1, 2], "estimators": [10, 12]}],
)
def test_assigns_param_grid(grid):
    encoder = DecisionTreeEncoder(param_grid=grid)
    if grid is None:
        assert encoder._assign_param_grid() == {"max_depth": [1, 2, 3, 4]}
    else:
        assert encoder._assign_param_grid() == grid


def test_unseen_is_encode(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = DecisionTreeEncoder(unseen="encode", regression=False, fill_value=-1)
    encoder.fit(X, y)

    X_unseen_input = make_df(
        {
            "var_A": ["A", "ZZZ", "YYY"],
            "var_B": ["C", "YYY", "ZZZ"],
        }
    )
    Xt = encoder.transform(X_unseen_input)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"var_A": [0.25, -1, -1], "var_B": [0.4, -1, -1]}


def test_unseen_is_ignore(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = DecisionTreeEncoder(unseen="ignore", regression=False)
    encoder.fit(X, y)

    X_unseen_input = make_df(
        {
            "var_A": ["A", "ZZZ", "YYY"],
            "var_B": ["C", "YYY", "ZZZ"],
        }
    )
    Xt = encoder.transform(X_unseen_input)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": [0.25, None, None],
        "var_B": [0.4, None, None],
    }


def test_fit_errors_if_new_cat_values_and_unseen_is_raise_param(make_df, data_enc):
    X = make_df(data_enc)[["var_A", "var_B"]]
    y = make_series(make_df, data_enc["target"])

    encoder = DecisionTreeEncoder(unseen="raise", regression=False)
    encoder.fit(X, y)
    X_unseen = make_df(
        {
            "var_A": ["A", "ZZZ", "YYY"],
            "var_B": ["C", "YYY", "ZZZ"],
        }
    )
    msg = (
        "During the encoding, NaN values were introduced in the "
        "feature(s) var_A, var_B."
    )
    # new categories will raise an error
    with pytest.raises(ValueError, match=re.escape(msg)):
        encoder.transform(X_unseen)


def test_inverse_transform_when_no_unseen(make_df):
    words = ["dog", "dog", "dog", "cat", "cat", "cat", "bird"]
    X = make_df({"words": words})
    y = make_series(make_df, [0, 0, 1, 1, 1, 1, 0])
    enc = DecisionTreeEncoder(regression=False)
    enc.fit(X, y)
    dft = enc.transform(X)
    Xi = enc.inverse_transform(dft)
    assert isinstance(Xi, make_df)
    assert frame_to_dict(Xi) == {"words": words}


def test_inverse_transform_when_ignore_unseen(make_df):
    X = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", "bird"]})
    y = make_series(make_df, [0, 0, 1, 1, 1, 1, 0])
    enc = DecisionTreeEncoder(regression=False, unseen="ignore")
    enc.fit(X, y)

    df1 = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", "frog"]})
    dft = enc.transform(df1)
    Xi = enc.inverse_transform(dft)
    assert isinstance(Xi, make_df)
    assert frame_to_dict(Xi) == {
        "words": ["dog", "dog", "dog", "cat", "cat", "cat", None]
    }


def test_inverse_transform_when_encode_unseen(make_df):
    X = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", "bird"]})
    y = make_series(make_df, [0, 0, 1, 1, 1, 1, 0])
    enc = DecisionTreeEncoder(regression=False, unseen="encode", fill_value=1000)
    enc.fit(X, y)

    df1 = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", "frog"]})
    dft = enc.transform(df1)
    Xi = enc.inverse_transform(dft)
    assert isinstance(Xi, make_df)
    assert frame_to_dict(Xi) == {
        "words": ["dog", "dog", "dog", "cat", "cat", "cat", None]
    }


def test_inverse_transform_raises_non_fitted_error(make_df):
    X = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", "bird"]})
    y = make_series(make_df, [0, 0, 1, 1, 1, 1, 0])
    enc = DecisionTreeEncoder(regression=False)
    msg = (
        "This DecisionTreeEncoder instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    msg_na = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        enc.inverse_transform(X)

    X_na = make_df({"words": ["dog", "dog", "dog", "cat", "cat", "cat", None]})

    with pytest.raises(ValueError, match=re.escape(msg_na)):
        enc.fit(X_na, y)

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        enc.inverse_transform(X_na)
