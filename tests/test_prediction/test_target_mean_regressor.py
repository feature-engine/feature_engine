import re

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine._prediction.target_mean_regressor import TargetMeanRegressor
from tests.backend_helpers import make_series

DATA = {
    "cat_var_A": ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5,
    "cat_var_B": ["A"] * 6 + ["B"] * 2 + ["C"] * 2 + ["B"] * 2 + ["C"] * 2 + ["D"] * 6,
    "num_var_A": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
    "num_var_B": [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4],
}
TARGET = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
# with 2 bins, the numerical variables are split at 2.5.
PREDICTIONS_ALL_VARIABLES = (
    [41 / 120] * 5
    + [71 / 120, 0.925, 0.925, 1.325, 1.325, 1.675, 1.675, 2.075, 2.075, 289 / 120]
    + [319 / 120] * 5
)
R2_ALL_VARIABLES = 5533 / 6000
MSG_BINARY = (
    "Trying to fit a regression to a binary target is not "
    "allowed by this transformer. "
)


# init parameters
@pytest.mark.parametrize("bins", [0, -1, 2.5, "3", None, [5]])
def test_error_if_bins_not_positive_integer(bins):
    msg = f"bins must be a positive integer. Got {bins} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        TargetMeanRegressor(bins=bins)


@pytest.mark.parametrize(
    "strategy", ["arbitrary", "Equal_width", "", 1, None, ["equal_width"]]
)
def test_error_if_strategy_not_permitted(strategy):
    msg = (
        "strategy takes only values 'equal_width' or 'equal_frequency'. "
        f"Got {strategy} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        TargetMeanRegressor(strategy=strategy)


@pytest.mark.parametrize(
    "bins, strategy", [(1, "equal_width"), (5, "equal_frequency"), (10, "equal_width")]
)
def test_init_param_assignment(bins, strategy):
    estimator = TargetMeanRegressor(bins=bins, strategy=strategy)
    assert estimator.bins == bins
    assert estimator.strategy == strategy


# fit and predict
@pytest.mark.parametrize(
    "variables, expected",
    [
        ("cat_var_A", [0.0] * 5 + [1.0] * 5 + [2.0] * 5 + [3.0] * 5),
        ("cat_var_B", [1 / 6] * 6 + [1.5] * 8 + [17 / 6] * 6),
        (
            ["cat_var_A", "cat_var_B"],
            [1 / 12] * 5
            + [7 / 12, 1.25, 1.25, 1.25, 1.25, 1.75, 1.75, 1.75, 1.75, 29 / 12]
            + [35 / 12] * 5,
        ),
    ],
)
def test_predict_categorical_variables(make_df, variables, expected):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    y_pred = TargetMeanRegressor(variables=variables).fit(X, y).predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.tolist() == pytest.approx(expected)


@pytest.mark.parametrize(
    "variables, expected",
    [
        ("num_var_A", [0.5] * 10 + [2.5] * 10),
        ("num_var_B", [0.7] * 8 + [2.3] * 2 + [0.7] * 2 + [2.3] * 8),
        (
            ["num_var_A", "num_var_B"],
            [0.6] * 8 + [1.4, 1.4, 1.6, 1.6] + [2.4] * 8,
        ),
    ],
)
def test_predict_numerical_variables(make_df, variables, expected):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    y_pred = TargetMeanRegressor(variables=variables, bins=2).fit(X, y).predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.tolist() == pytest.approx(expected)


def test_predict_and_score_all_variables(make_df):
    X = make_df(DATA)
    y = make_series(make_df, TARGET)

    estimator = TargetMeanRegressor(bins=2).fit(X, y)

    assert estimator.predict(X).tolist() == pytest.approx(PREDICTIONS_ALL_VARIABLES)
    assert estimator.score(X, y) == pytest.approx(R2_ALL_VARIABLES)


@pytest.mark.parametrize("to_target", [list, np.array])
def test_target_as_list_or_array(make_df, to_target):
    X = make_df(DATA)
    y = to_target(TARGET)

    estimator = TargetMeanRegressor(bins=2).fit(X, y)

    assert estimator.predict(X).tolist() == pytest.approx(PREDICTIONS_ALL_VARIABLES)
    assert estimator.score(X, y) == pytest.approx(R2_ALL_VARIABLES)


@pytest.mark.parametrize("target", [[0, 1] * 10, [1.0, 2.0] * 10, ["a", "b"] * 10])
def test_error_if_target_is_binary(make_df, target):
    X = make_df(DATA)
    y = make_series(make_df, target)

    with pytest.raises(ValueError, match=re.escape(MSG_BINARY)):
        TargetMeanRegressor().fit(X, y)


def test_error_if_not_fitted(make_df):
    msg = (
        "This TargetMeanRegressor instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        TargetMeanRegressor().predict(make_df(DATA))


def test_integer_column_names():
    X = pd.DataFrame(DATA)
    X.columns = [0, 1, 2, 3]
    y = pd.Series(TARGET)

    estimator = TargetMeanRegressor(bins=2).fit(X, y)

    assert estimator.predict(X).tolist() == pytest.approx(PREDICTIONS_ALL_VARIABLES)
    assert estimator.score(X, y) == pytest.approx(R2_ALL_VARIABLES)
