import re

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine._prediction.target_mean_classifier import TargetMeanClassifier
from tests.backend_helpers import make_series

DATA = {
    "cat_var_A": ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5,
    "cat_var_B": ["A"] * 6 + ["B"] * 2 + ["C"] * 2 + ["B"] * 2 + ["C"] * 2 + ["D"] * 6,
    "num_var_A": [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5,
    "num_var_B": [1] * 6 + [2, 2, 3, 3, 2, 2, 3, 3] + [4] * 6,
}
TARGET = [0] * 10 + [1] * 10

# probability of the second class, per variable combination
PROBA = {
    "cat_var_A": [0.0] * 10 + [1.0] * 10,
    "cat_var_B": [0.0] * 6 + [0.5] * 8 + [1.0] * 6,
    "cat_var_A, cat_var_B": [0.0] * 6 + [0.25] * 4 + [0.75] * 4 + [1.0] * 6,
    "num_var_A": [0.0] * 10 + [1.0] * 10,
    "num_var_B": [0.2] * 8 + [0.8] * 2 + [0.2] * 2 + [0.8] * 8,
    "num_var_A, num_var_B": [0.1] * 8 + [0.4] * 2 + [0.6] * 2 + [0.9] * 8,
    "all": [0.05] * 6
    + [0.175] * 2
    + [0.325] * 2
    + [0.675] * 2
    + [0.825] * 2
    + [0.95] * 6,
}
PREDICTIONS = {
    "cat_var_A": [0] * 10 + [1] * 10,
    # a probability of 0.5 returns the first class
    "cat_var_B": [0] * 14 + [1] * 6,
    "cat_var_A, cat_var_B": [0] * 10 + [1] * 10,
    "num_var_A": [0] * 10 + [1] * 10,
    "num_var_B": [0] * 8 + [1] * 2 + [0] * 2 + [1] * 8,
    "num_var_A, num_var_B": [0] * 10 + [1] * 10,
    "all": [0] * 10 + [1] * 10,
}


# init parameters
# the errors of bins and strategy are tested in test_base_predictor.py
@pytest.mark.parametrize(
    "bins, strategy", [(1, "equal_width"), (5, "equal_frequency"), (10, "equal_width")]
)
def test_init_param_assignment(bins, strategy):
    estimator = TargetMeanClassifier(bins=bins, strategy=strategy)
    assert estimator.bins == bins
    assert estimator.strategy == strategy


# fit and predict
@pytest.mark.parametrize(
    "variables, key",
    [
        ("cat_var_A", "cat_var_A"),
        ("cat_var_B", "cat_var_B"),
        (["cat_var_A", "cat_var_B"], "cat_var_A, cat_var_B"),
        ("num_var_A", "num_var_A"),
        ("num_var_B", "num_var_B"),
        (["num_var_A", "num_var_B"], "num_var_A, num_var_B"),
        (None, "all"),
    ],
)
def test_predict_and_predict_proba(make_df, variables, key):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    estimator = TargetMeanClassifier(variables=variables, bins=2).fit(X, y)
    y_pred = estimator.predict(X)
    proba = estimator.predict_proba(X)

    assert estimator.classes_.tolist() == [0, 1]
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.tolist() == PREDICTIONS[key]
    assert proba.shape == (20, 2)
    assert proba[:, 1].tolist() == pytest.approx(PROBA[key])
    assert proba[:, 0].tolist() == pytest.approx([1 - p for p in PROBA[key]])


def test_predict_log_proba(make_df):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    estimator = TargetMeanClassifier(bins=2).fit(X, y)
    log_proba = estimator.predict_log_proba(X)

    assert log_proba[:, 1].tolist() == pytest.approx(np.log(PROBA["all"]).tolist())
    assert log_proba[:, 0].tolist() == pytest.approx(
        np.log([1 - p for p in PROBA["all"]]).tolist()
    )


@pytest.mark.parametrize(
    "labels",
    [[0, 1], [1, 2], [-1, 1], [0.0, 1.0], [False, True], ["no", "yes"]],
)
def test_classes_and_predictions_take_the_target_labels(make_df, labels):
    X = make_df(DATA)
    y = make_series(make_df, [labels[value] for value in TARGET])

    estimator = TargetMeanClassifier(variables="cat_var_B").fit(X, y)

    assert estimator.classes_.tolist() == labels
    assert estimator.predict(X).tolist() == [
        labels[value] for value in PREDICTIONS["cat_var_B"]
    ]
    assert estimator.predict_proba(X)[:, 1].tolist() == pytest.approx(
        PROBA["cat_var_B"]
    )


def test_classes_are_sorted(make_df):
    # the probabilities are those of the class with the largest label
    X = make_df(DATA)
    y = make_series(make_df, [2 if value == 0 else 1 for value in TARGET])

    estimator = TargetMeanClassifier(variables="cat_var_B").fit(X, y)

    assert estimator.classes_.tolist() == [1, 2]
    assert estimator.predict_proba(X)[:, 1].tolist() == pytest.approx(
        [1 - p for p in PROBA["cat_var_B"]]
    )
    assert estimator.predict(X).tolist() == [2] * 6 + [1] * 14


def test_score(make_df):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    estimator = TargetMeanClassifier(variables="num_var_B", bins=2).fit(X, y)

    assert estimator.score(X, y) == pytest.approx(0.8)


@pytest.mark.parametrize("to_target", [list, np.array])
@pytest.mark.parametrize("labels", [[0, 1], ["no", "yes"]])
def test_target_as_list_or_array(make_df, to_target, labels):
    X = make_df(DATA)
    y = to_target([labels[value] for value in TARGET])

    estimator = TargetMeanClassifier(bins=2).fit(X, y)

    assert estimator.classes_.tolist() == labels
    assert estimator.predict(X).tolist() == [
        labels[value] for value in PREDICTIONS["all"]
    ]
    assert estimator.predict_proba(X)[:, 1].tolist() == pytest.approx(PROBA["all"])


def test_error_if_target_has_more_than_two_classes(make_df):
    X = make_df(DATA)
    y = make_series(make_df, [0] * 10 + [1] * 9 + [2])

    msg = (
        "This classifier is designed for binary classification only. "
        "The target has more than 2 unique values."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        TargetMeanClassifier().fit(X, y)


def test_error_if_target_is_continuous(make_df):
    X = make_df(DATA)
    y = make_series(make_df, [0.5] * 10 + [1.5] * 10)

    msg = (
        "Unknown label type: continuous. Maybe you are trying to fit a classifier, "
        "which expects discrete classes on a regression target with continuous "
        "values."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        TargetMeanClassifier().fit(X, y)


@pytest.mark.parametrize("method", ["predict", "predict_proba", "predict_log_proba"])
def test_error_if_not_fitted(make_df, method):
    msg = (
        "This TargetMeanClassifier instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        getattr(TargetMeanClassifier(), method)(make_df(DATA))


def test_integer_column_names():
    X = pd.DataFrame({0: DATA["cat_var_B"], 1: DATA["num_var_B"]})
    y = pd.Series(["no" if value == 0 else "yes" for value in TARGET])

    estimator = TargetMeanClassifier(bins=2).fit(X, y)

    assert estimator.variables_categorical_ == [0]
    assert estimator.variables_numerical_ == [1]
    assert estimator.classes_.tolist() == ["no", "yes"]
    assert estimator.predict_proba(X)[:, 1].tolist() == pytest.approx(
        [0.1] * 6 + [0.35] * 2 + [0.65] * 2 + [0.35] * 2 + [0.65] * 2 + [0.9] * 6
    )


def test_pandas_index_is_ignored():
    index = list(range(100, 120))
    X = pd.DataFrame(DATA, index=index)
    y = pd.Series(TARGET, index=index)

    estimator = TargetMeanClassifier(bins=2).fit(X, y)

    assert estimator.predict(X).tolist() == PREDICTIONS["all"]
    assert estimator.predict_proba(X)[:, 1].tolist() == pytest.approx(PROBA["all"])
