import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine._prediction.base_predictor import BaseTargetMeanEstimator
from tests.backend_helpers import make_series

INF = float("inf")

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)
MSG_INF = (
    "Some of the variables to transform contain inf values. Check and "
    "remove those before using this transformer."
)

DATA = {
    "num": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
    "cat": ["a", "a", "b", "b", "b", "c", "c", "c", "c", "a"],
}
TARGET = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ENC_DICT_CAT = {"a": 13 / 3, "b": 4.0, "c": 7.5}
# num has an outlier, so 2 equal-width bins split it at 50.5 and 2 equal-frequency
# bins at the median, 5.5.
BINNER_DICT_NUM = {
    "equal_width": [-INF, 50.5, INF],
    "equal_frequency": [-INF, 5.5, INF],
}
ENC_DICT_NUM = {
    "equal_width": {"(-inf, 50.5]": 5.0, "(50.5, inf]": 10.0},
    "equal_frequency": {"(-inf, 5.5]": 3.0, "(5.5, inf]": 8.0},
}
PREDICTIONS = {
    "equal_width": [14 / 3, 14 / 3, 4.5, 4.5, 4.5, 6.25, 6.25, 6.25, 6.25, 43 / 6],
    "equal_frequency": [
        11 / 3,
        11 / 3,
        3.5,
        3.5,
        3.5,
        7.75,
        7.75,
        7.75,
        7.75,
        37 / 6,
    ],
}


# init parameters
@pytest.mark.parametrize("bins", [0, -1, 2.5, "3", None, [5]])
def test_error_if_bins_not_positive_integer(bins):
    msg = f"bins must be a positive integer. Got {bins} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        BaseTargetMeanEstimator(bins=bins)


@pytest.mark.parametrize(
    "strategy", ["arbitrary", "Equal_width", "", 1, None, ["equal_width"]]
)
def test_error_if_strategy_not_permitted(strategy):
    msg = (
        "strategy takes only values 'equal_width' or 'equal_frequency'. "
        f"Got {strategy} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        BaseTargetMeanEstimator(strategy=strategy)


@pytest.mark.parametrize(
    "bins, strategy", [(1, "equal_width"), (5, "equal_frequency"), (10, "equal_width")]
)
def test_init_param_assignment(bins, strategy):
    estimator = BaseTargetMeanEstimator(bins=bins, strategy=strategy)
    assert estimator.bins == bins
    assert estimator.strategy == strategy


# fit and transform
@pytest.mark.parametrize("strategy", ["equal_width", "equal_frequency"])
def test_fit_attributes(make_df, strategy):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    estimator = BaseTargetMeanEstimator(bins=2, strategy=strategy)
    estimator.fit(X, y)

    assert estimator.variables_numerical_ == ["num"]
    assert estimator.variables_categorical_ == ["cat"]
    assert estimator.binner_dict_ == {"num": BINNER_DICT_NUM[strategy]}
    assert estimator.encoder_dict_["num"] == pytest.approx(ENC_DICT_NUM[strategy])
    assert estimator.encoder_dict_["cat"] == pytest.approx(ENC_DICT_CAT)
    assert estimator.feature_names_in_ == ["num", "cat"]
    assert estimator.n_features_in_ == 2


@pytest.mark.parametrize("strategy", ["equal_width", "equal_frequency"])
def test_predict(make_df, strategy):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    estimator = BaseTargetMeanEstimator(bins=2, strategy=strategy).fit(X, y)
    y_pred = estimator._predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.tolist() == pytest.approx(PREDICTIONS[strategy])


@pytest.mark.parametrize("to_target", [list, np.array])
def test_target_as_list_or_array(make_df, to_target):
    X = make_df(DATA)
    y = to_target(TARGET)

    estimator = BaseTargetMeanEstimator(bins=2).fit(X, y)

    assert estimator.encoder_dict_["num"] == pytest.approx(ENC_DICT_NUM["equal_width"])
    assert estimator.encoder_dict_["cat"] == pytest.approx(ENC_DICT_CAT)
    assert estimator._predict(X).tolist() == pytest.approx(PREDICTIONS["equal_width"])


def test_only_numerical_variables(make_df):
    X = make_df({"num": DATA["num"]})
    y = make_series(make_df, TARGET)

    estimator = BaseTargetMeanEstimator(bins=2).fit(X, y)

    assert estimator.variables_categorical_ == []
    assert estimator.encoder_dict_["num"] == pytest.approx(ENC_DICT_NUM["equal_width"])
    assert estimator._predict(X).tolist() == pytest.approx([5.0] * 9 + [10.0])


def test_only_categorical_variables(make_df):
    X = make_df({"cat": DATA["cat"]})
    y = make_series(make_df, TARGET)

    estimator = BaseTargetMeanEstimator(bins=2).fit(X, y)

    assert estimator.variables_numerical_ == []
    assert estimator.binner_dict_ == {}
    assert estimator.encoder_dict_["cat"] == pytest.approx(ENC_DICT_CAT)
    assert estimator._predict(X).tolist() == pytest.approx(
        [13 / 3, 13 / 3, 4.0, 4.0, 4.0, 7.5, 7.5, 7.5, 7.5, 13 / 3]
    )


@pytest.mark.parametrize(
    "variables, numerical, categorical",
    [
        ("num", ["num"], []),
        (["cat"], [], ["cat"]),
        (["cat", "num"], ["num"], ["cat"]),
        (None, ["num"], ["cat"]),
    ],
)
def test_variable_selection(make_df, variables, numerical, categorical):
    # datetime variables are never used for prediction
    data = {**DATA, "date": [datetime(2020, 1, day) for day in range(1, 11)]}
    X = make_df(data)
    y = make_series(make_df, TARGET)

    estimator = BaseTargetMeanEstimator(bins=2, variables=variables).fit(X, y)

    assert estimator.variables_numerical_ == numerical
    assert estimator.variables_categorical_ == categorical
    assert estimator.feature_names_in_ == ["num", "cat", "date"]
    assert estimator.n_features_in_ == 3


def test_predict_with_reordered_columns(make_df):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    estimator = BaseTargetMeanEstimator(bins=2).fit(X, y)
    y_pred = estimator._predict(X[["cat", "num"]])

    assert y_pred.tolist() == pytest.approx(PREDICTIONS["equal_width"])


def test_values_outside_training_range_take_the_outer_bins(make_df):
    X = make_df({"num": DATA["num"]})
    y = make_series(make_df, TARGET)

    estimator = BaseTargetMeanEstimator(bins=2).fit(X, y)
    y_pred = estimator._predict(make_df({"num": [-1000, 1000]}))

    assert y_pred.tolist() == pytest.approx([5.0, 10.0])


def test_constant_numerical_variable(make_df):
    X = make_df({"num": [1.0, 1.0, 1.0, 1.0]})
    y = make_series(make_df, [1, 2, 3, 4])

    estimator = BaseTargetMeanEstimator(bins=2).fit(X, y)

    assert estimator.encoder_dict_ == {"num": {"(-inf, 1.0]": 2.5}}
    assert estimator._predict(X).tolist() == pytest.approx([2.5] * 4)


def test_error_if_predict_df_has_unseen_category(make_df):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)
    estimator = BaseTargetMeanEstimator(bins=2).fit(X, y)

    msg = "During the encoding, NaN values were introduced in the feature(s) cat."
    with pytest.raises(ValueError, match=re.escape(msg)):
        estimator._predict(make_df({"num": [1, 2], "cat": ["a", "z"]}))


def test_error_if_predict_df_has_values_in_bins_empty_in_train(make_df):
    # the bins between 2 and 8 have no observations in the train set
    X = make_df({"num": [0, 1, 2, 9, 10]})
    y = make_series(make_df, [1, 2, 3, 4, 5])
    estimator = BaseTargetMeanEstimator(bins=5).fit(X, y)

    msg = "During the encoding, NaN values were introduced in the feature(s) num."
    with pytest.raises(ValueError, match=re.escape(msg)):
        estimator._predict(make_df({"num": [0, 5]}))


@pytest.mark.parametrize(
    "variable, value", [("num", None), ("num", float("nan")), ("cat", None)]
)
def test_error_if_df_contains_na(make_df, variable, value):
    data_na = {**DATA, variable: [value] + DATA[variable][1:]}
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        BaseTargetMeanEstimator(bins=2).fit(make_df(data_na), y)

    estimator = BaseTargetMeanEstimator(bins=2).fit(X, y)
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        estimator._predict(make_df(data_na))


def test_error_if_df_contains_inf(make_df):
    data_inf = {**DATA, "num": [INF] + DATA["num"][1:]}
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    with pytest.raises(ValueError, match=re.escape(MSG_INF)):
        BaseTargetMeanEstimator(bins=2).fit(make_df(data_inf), y)

    estimator = BaseTargetMeanEstimator(bins=2).fit(X, y)
    with pytest.raises(ValueError, match=re.escape(MSG_INF)):
        estimator._predict(make_df(data_inf))


def test_error_if_predict_df_has_different_number_of_columns(make_df):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)
    estimator = BaseTargetMeanEstimator(bins=2).fit(X, y)

    msg = (
        "The number of columns in this dataset is different from the one used to "
        "fit this transformer (when using the fit() method)."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        estimator._predict(X[["num"]])


def test_error_if_not_fitted(make_df):
    msg = (
        "This BaseTargetMeanEstimator instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        BaseTargetMeanEstimator()._predict(make_df(DATA))


def test_integer_column_names():
    X = pd.DataFrame({0: DATA["num"], 1: DATA["cat"]})
    y = pd.Series(TARGET)

    estimator = BaseTargetMeanEstimator(bins=2).fit(X, y)

    assert estimator.variables_numerical_ == [0]
    assert estimator.variables_categorical_ == [1]
    assert estimator.binner_dict_ == {0: BINNER_DICT_NUM["equal_width"]}
    assert estimator.encoder_dict_[0] == pytest.approx(ENC_DICT_NUM["equal_width"])
    assert estimator.encoder_dict_[1] == pytest.approx(ENC_DICT_CAT)
    assert estimator._predict(X).tolist() == pytest.approx(PREDICTIONS["equal_width"])


def test_pandas_index_is_ignored():
    index = [10, 3, 7, 0, 1, 8, 2, 9, 4, 6]
    X = pd.DataFrame(DATA, index=index)
    y = pd.Series(TARGET, index=index)

    estimator = BaseTargetMeanEstimator(bins=2).fit(X, y)

    assert estimator.encoder_dict_["cat"] == pytest.approx(ENC_DICT_CAT)
    assert estimator._predict(X).tolist() == pytest.approx(PREDICTIONS["equal_width"])


def test_category_dtype():
    # the unused category "d" gets no target mean
    cat = pd.Categorical(DATA["cat"], categories=["a", "b", "c", "d"])
    X = pd.DataFrame({"cat": cat})
    y = pd.Series(TARGET)

    estimator = BaseTargetMeanEstimator().fit(X, y)

    assert estimator.encoder_dict_["cat"] == pytest.approx(ENC_DICT_CAT)
    assert estimator._predict(X).tolist() == pytest.approx(
        [13 / 3, 13 / 3, 4.0, 4.0, 4.0, 7.5, 7.5, 7.5, 7.5, 13 / 3]
    )
