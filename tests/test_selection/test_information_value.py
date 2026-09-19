import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.selection import SelectByInformationValue
from tests.backend_helpers import frame_to_dict, make_series

TARGET = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]

DATA = {
    "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
    "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
    "var_C": ["X"] * 7 + ["Y"] * 5 + ["Z"] * 8,
    "var_D": ["L"] * 3 + ["M"] * 9 + ["N"] * 8,
    "var_E": ["R"] * 7 + ["S"] * 4 + ["T"] * 9,
}

IV = {
    "var_A": 0.29706307738283366,
    "var_B": 0.29706307738283366,
    "var_C": 0.07817653204775647,
    "var_D": 0.494962117149986,
    "var_E": 0.024620803988822354,
}

# with 3 intervals, var_num has the same IV with both strategies
DATA_NUM = {
    "var_A": DATA["var_A"],
    "var_num": np.linspace(0, 20, num=20).tolist(),
    "var_E": DATA["var_E"],
}
IV_NUM = {
    "var_A": 0.29706307738283366,
    "var_num": 0.010625883395914762,
    "var_E": 0.024620803988822354,
}


# init parameters
@pytest.mark.parametrize("bins", ["python", (True, False), 4.3, -1, 0, None])
def test_error_if_bins_not_permitted(bins):
    msg = f"bins must be an integer. Got {bins} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        SelectByInformationValue(bins=bins)


@pytest.mark.parametrize(
    "strategy", ["python", (True, False), 4.3, -1, None, ["equal_width"]]
)
def test_error_if_strategy_not_permitted(strategy):
    msg = (
        "strategy takes only values 'equal_width' or 'equal_frequency'. "
        f"Got {strategy} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SelectByInformationValue(strategy=strategy)


@pytest.mark.parametrize("threshold", ["python", (True, False), [4.3, 3], None])
def test_error_if_threshold_not_permitted(threshold):
    msg = f"threshold must be an integer or a float. Got {threshold} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        SelectByInformationValue(threshold=threshold)


@pytest.mark.parametrize("confirm_variables", [None, "hola", [True], 1, 0.5])
def test_error_if_confirm_variables_not_bool(confirm_variables):
    msg = (
        "confirm_variables takes only values True and False. "
        f"Got {confirm_variables} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SelectByInformationValue(confirm_variables=confirm_variables)


@pytest.mark.parametrize(
    "bins, strategy, threshold, confirm_variables",
    [
        (5, "equal_width", 0.2, False),
        (10, "equal_frequency", 1, True),
        (3, "equal_width", 0.05, True),
    ],
)
def test_init_param_assignment(bins, strategy, threshold, confirm_variables):
    sel = SelectByInformationValue(
        bins=bins,
        strategy=strategy,
        threshold=threshold,
        confirm_variables=confirm_variables,
    )
    assert sel.bins == bins
    assert sel.strategy == strategy
    assert sel.threshold == threshold
    assert sel.confirm_variables is confirm_variables


# fit and transform
def test_default_parameters(make_df):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    sel = SelectByInformationValue().fit(X, y)
    Xt = sel.transform(X)

    assert sel.variables_ == ["var_A", "var_B", "var_C", "var_D", "var_E"]
    assert sel.information_values_ == pytest.approx(IV)
    assert sel.features_to_drop_ == ["var_C", "var_E"]
    assert sel.feature_names_in_ == ["var_A", "var_B", "var_C", "var_D", "var_E"]
    assert sel.n_features_in_ == 5
    assert list(sel.get_support()) == [True, True, False, True, False]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": DATA["var_A"],
        "var_B": DATA["var_B"],
        "var_D": DATA["var_D"],
    }


def test_information_value_formula(make_df):
    # var_A: A has 2 positive and 4 negative cases, B 2 and 8, and C 2 and 2
    X = make_df({"var_A": DATA["var_A"], "var_B": DATA["var_B"]})
    y = make_series(make_df, TARGET)

    sel = SelectByInformationValue().fit(X, y)

    # 6 positive and 14 negative cases
    iv_a = (
        (2 / 6 - 4 / 14) * math.log((2 / 6) / (4 / 14))
        + (2 / 6 - 8 / 14) * math.log((2 / 6) / (8 / 14))
        + (2 / 6 - 2 / 14) * math.log((2 / 6) / (2 / 14))
    )
    assert sel.information_values_["var_A"] == pytest.approx(iv_a)


@pytest.mark.parametrize("threshold, features_to_drop", [(0, []), (0.5, list(IV))])
def test_threshold(make_df, threshold, features_to_drop):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    sel = SelectByInformationValue(threshold=threshold).fit(X, y)

    assert sel.information_values_ == pytest.approx(IV)
    assert sel.features_to_drop_ == features_to_drop


def test_user_passes_variables(make_df):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    sel = SelectByInformationValue(variables=["var_A", "var_C", "var_E"]).fit(X, y)
    Xt = sel.transform(X)

    assert sel.variables_ == ["var_A", "var_C", "var_E"]
    assert sel.information_values_ == pytest.approx(
        {"var_A": IV["var_A"], "var_C": IV["var_C"], "var_E": IV["var_E"]}
    )
    assert sel.features_to_drop_ == ["var_C", "var_E"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": DATA["var_A"],
        "var_B": DATA["var_B"],
        "var_D": DATA["var_D"],
    }


def test_confirm_variables(make_df):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    sel = SelectByInformationValue(
        variables=["var_A", "var_C", "var_Z"], confirm_variables=True
    ).fit(X, y)

    assert sel.variables_ == ["var_A", "var_C"]
    assert sel.features_to_drop_ == ["var_C"]


def test_error_if_variable_not_in_dataframe(make_df):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)
    sel = SelectByInformationValue(variables=["var_A", "var_Z"])
    msg = "Some of the variables are not in the dataframe."
    with pytest.raises(KeyError, match=re.escape(msg)):
        sel.fit(X, y)


@pytest.mark.parametrize("strategy", ["equal_width", "equal_frequency"])
def test_numerical_and_categorical_variables(make_df, strategy):
    X = make_df(DATA_NUM)
    y = make_series(make_df, TARGET)

    sel = SelectByInformationValue(bins=3, strategy=strategy).fit(X, y)
    Xt = sel.transform(X)

    assert sel.variables_ == ["var_A", "var_num", "var_E"]
    assert sel.information_values_ == pytest.approx(IV_NUM)
    assert sel.features_to_drop_ == ["var_num", "var_E"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"var_A": DATA["var_A"]}


def test_fit_does_not_modify_input(make_df):
    X = make_df(DATA_NUM)
    y = make_series(make_df, TARGET)
    SelectByInformationValue(bins=3).fit(X, y)
    assert frame_to_dict(X) == DATA_NUM


def test_zero_counts_are_replaced_by_half(make_df):
    # C has no negative cases and D no positive cases
    data = {
        "var_A": ["A"] * 9 + ["B"] * 6 + ["C"] * 3 + ["D"] * 2,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
    }
    target = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    X = make_df(data)
    y = make_series(make_df, target)

    sel = SelectByInformationValue().fit(X, y)

    # 7 positive and 13 negative cases
    iv_a = (
        (2 / 7 - 7 / 13) * math.log((2 / 7) / (7 / 13))
        + (2 / 7 - 4 / 13) * math.log((2 / 7) / (4 / 13))
        + (3 / 7 - 0.5 / 13) * math.log((3 / 7) / (0.5 / 13))
        + (0.5 / 7 - 2 / 13) * math.log((0.5 / 7) / (2 / 13))
    )
    assert sel.information_values_["var_A"] == pytest.approx(iv_a)


def test_empty_intervals_are_ignored(make_df):
    # with 5 equal-width intervals, 0 to 18 fall in the first and 1000 in the last
    X = make_df({"var_A": DATA["var_A"], "var_num": list(range(19)) + [1000]})
    y = make_series(make_df, TARGET)

    sel = SelectByInformationValue().fit(X, y)

    # the first interval has 6 positive and 13 negative cases, the last 0 and 1
    iv_num = (6 / 6 - 13 / 14) * math.log((6 / 6) / (13 / 14)) + (
        0.5 / 6 - 1 / 14
    ) * math.log((0.5 / 6) / (1 / 14))
    assert sel.information_values_["var_num"] == pytest.approx(iv_num)


@pytest.mark.parametrize(
    "target",
    [
        [2 if t == 1 else 1 for t in TARGET],
        [1 if t == 1 else -1 for t in TARGET],
        ["yes" if t == 1 else "no" for t in TARGET],
        [t == 1 for t in TARGET],
    ],
)
def test_target_not_0_1(make_df, target):
    X = make_df(DATA)
    y = make_series(make_df, target)

    sel = SelectByInformationValue().fit(X, y)

    assert sel.information_values_ == pytest.approx(IV)
    assert sel.features_to_drop_ == ["var_C", "var_E"]


@pytest.mark.parametrize("to_target", [list, np.array])
def test_target_as_list_or_array(make_df, to_target):
    X = make_df(DATA)
    y = to_target(TARGET)

    sel = SelectByInformationValue().fit(X, y)

    assert sel.information_values_ == pytest.approx(IV)
    assert sel.features_to_drop_ == ["var_C", "var_E"]


def test_error_if_target_not_binary(make_df):
    X = make_df(DATA)
    y = make_series(make_df, [0, 1, 2] * 6 + [0, 1])
    msg = (
        "This encoder is designed for binary classification. The target "
        "used has more than 2 unique values."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SelectByInformationValue().fit(X, y)


def test_error_if_missing_values(make_df):
    X = make_df({**DATA_NUM, "var_num": [None] + DATA_NUM["var_num"][1:]})
    y = make_series(make_df, TARGET)
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SelectByInformationValue().fit(X, y)


def test_error_if_infinite_values(make_df):
    X = make_df({**DATA_NUM, "var_num": [np.inf] + DATA_NUM["var_num"][1:]})
    y = make_series(make_df, TARGET)
    msg = (
        "Some of the variables to transform contain inf values. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SelectByInformationValue().fit(X, y)


def test_datetime_variables_are_not_evaluated(make_df):
    dates = [datetime(2020, 2, 24, 0, minute) for minute in range(20)]
    X = make_df({**DATA, "dob": dates})
    y = make_series(make_df, TARGET)

    sel = SelectByInformationValue().fit(X, y)
    Xt = sel.transform(X)

    assert sel.variables_ == ["var_A", "var_B", "var_C", "var_D", "var_E"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_A": DATA["var_A"],
        "var_B": DATA["var_B"],
        "var_D": DATA["var_D"],
        "dob": dates,
    }


def test_error_if_transform_before_fit(make_df):
    msg = (
        "This SelectByInformationValue instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        SelectByInformationValue().transform(make_df(DATA))


def test_integer_column_names():
    # pandas allows integer column names, polars doesn't
    X = pd.DataFrame(DATA_NUM)
    X.columns = [0, 1, 2]
    y = pd.Series(TARGET)

    sel = SelectByInformationValue(bins=3).fit(X, y)
    Xt = sel.transform(X)

    assert sel.variables_ == [0, 1, 2]
    assert sel.information_values_ == pytest.approx(
        {0: IV_NUM["var_A"], 1: IV_NUM["var_num"], 2: IV_NUM["var_E"]}
    )
    assert sel.features_to_drop_ == [1, 2]
    pd.testing.assert_frame_equal(Xt, X[[0]])


def test_unused_categories_are_ignored():
    # pandas category dtype can hold categories that don't appear in the data
    X = pd.DataFrame(DATA)
    X["var_A"] = pd.Categorical(X["var_A"], categories=["A", "B", "C", "D"])
    y = pd.Series(TARGET)

    sel = SelectByInformationValue().fit(X, y)
    Xt = sel.transform(X)

    assert sel.information_values_ == pytest.approx(IV)
    pd.testing.assert_frame_equal(Xt, X[["var_A", "var_B", "var_D"]])


def test_target_not_0_1_with_pandas_index():
    # the target is paired with X by position, keeping the index of X
    X = pd.DataFrame(DATA, index=range(100, 120))
    y = pd.Series([2 if t == 1 else 1 for t in TARGET], index=X.index)

    sel = SelectByInformationValue().fit(X, y)
    Xt = sel.transform(X)

    assert sel.information_values_ == pytest.approx(IV)
    pd.testing.assert_frame_equal(Xt, X[["var_A", "var_B", "var_D"]])
