import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import RecursiveFeatureElimination
from tests.backend_helpers import frame_to_dict, make_series


@pytest.fixture(scope="module")
def data_diabetes():
    X, y = load_diabetes(return_X_y=True)
    data = {f"var_{i}": X[:, i].tolist() for i in range(10)}
    data["target"] = y.tolist()
    return data


def _split_target(make_df, data):
    X = make_df({k: v for k, v in data.items() if k != "target"})
    return X, make_series(make_df, data["target"])


def _rounded(drifts):
    return {k: round(v, 4) for k, v in drifts.items()}


# init parameters
@pytest.mark.parametrize("threshold", [None, [0.1], "a_string", {"a": 1}])
def test_error_if_threshold_not_number(threshold):
    msg = f"threshold must be an integer or a float. Got {threshold} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        RecursiveFeatureElimination(RandomForestClassifier(), threshold=threshold)


@pytest.mark.parametrize(
    "estimator, scoring, cv, groups, threshold, confirm_variables",
    [
        (RandomForestClassifier(), "roc_auc", 3, None, 0.01, False),
        (Lasso(), "neg_mean_squared_error", KFold(), [1, 2], 1, True),
        (DecisionTreeRegressor(), "r2", StratifiedKFold(), None, -0.5, False),
    ],
)
def test_init_param_assignment(
    estimator, scoring, cv, groups, threshold, confirm_variables
):
    sel = RecursiveFeatureElimination(
        estimator,
        scoring=scoring,
        cv=cv,
        groups=groups,
        threshold=threshold,
        confirm_variables=confirm_variables,
    )
    assert sel.estimator is estimator
    assert sel.scoring == scoring
    assert sel.cv is cv
    assert sel.groups == groups
    assert sel.threshold == threshold
    assert sel.confirm_variables is confirm_variables


# fit and transform
_classification_expectations = [
    (
        RandomForestClassifier(n_estimators=5, random_state=1),
        3,
        0.001,
        "roc_auc",
        ["var_0", "var_4"],
        {
            "var_5": -0.0,
            "var_3": 0.0009,
            "var_2": -0.0001,
            "var_1": -0.002,
            "var_11": 0.001,
            "var_10": 0.0009,
            "var_8": 0.0001,
            "var_0": 0.002,
            "var_9": 0.0,
            "var_6": -0.0,
            "var_7": -0.0015,
            "var_4": 0.4149,
        },
    ),
    (
        LogisticRegression(random_state=10),
        2,
        0.0001,
        "accuracy",
        ["var_0", "var_7", "var_8"],
        {
            "var_2": 0.0,
            "var_9": 0.0,
            "var_10": 0.0,
            "var_3": 0.0,
            "var_5": -0.001,
            "var_1": 0.0,
            "var_11": 0.0,
            "var_4": -0.002,
            "var_6": 0.0,
            "var_0": 0.002,
            "var_8": 0.002,
            "var_7": 0.004,
        },
    ),
]


@pytest.mark.parametrize(
    "estimator, cv, threshold, scoring, selected, drifts",
    _classification_expectations,
)
def test_classification(
    make_df, data_classification, estimator, cv, threshold, scoring, selected, drifts
):
    X, y = _split_target(make_df, data_classification)
    sel = RecursiveFeatureElimination(
        estimator=estimator, cv=cv, threshold=threshold, scoring=scoring
    )
    Xt = sel.fit_transform(X, y)

    assert sel.features_to_drop_ == [f for f in X.columns if f not in selected]
    # features are evaluated from the least to the most important
    assert list(sel.performance_drifts_.keys()) == list(drifts.keys())
    assert _rounded(sel.performance_drifts_) == pytest.approx(drifts, abs=0.001)
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {f: data_classification[f] for f in selected}


_regression_expectations = [
    (
        Lasso(alpha=0.001, random_state=10),
        3,
        0.1,
        "r2",
        ["var_2", "var_8"],
        {
            "var_0": -0.0032,
            "var_9": -0.0003,
            "var_6": -0.0008,
            "var_7": 0.0001,
            "var_1": 0.012,
            "var_3": 0.0199,
            "var_5": 0.0023,
            "var_2": 0.1378,
            "var_4": 0.0069,
            "var_8": 0.115,
        },
    ),
    (
        DecisionTreeRegressor(random_state=10),
        2,
        100,
        "neg_mean_squared_error",
        ["var_0", "var_2", "var_3", "var_5", "var_6", "var_7", "var_8", "var_9"],
        {
            "var_1": 64.6086,
            "var_4": -200.8348,
            "var_0": 481.9525,
            "var_6": 286.6561,
            "var_9": 700.138,
            "var_3": 345.2262,
            "var_7": 246.7828,
            "var_5": 438.2579,
            "var_8": 301.6968,
            "var_2": 1418.81,
        },
    ),
]


@pytest.mark.parametrize(
    "estimator, cv, threshold, scoring, selected, drifts",
    _regression_expectations,
)
def test_regression(
    make_df, data_diabetes, estimator, cv, threshold, scoring, selected, drifts
):
    X, y = _split_target(make_df, data_diabetes)
    sel = RecursiveFeatureElimination(
        estimator=estimator, cv=cv, threshold=threshold, scoring=scoring
    )
    Xt = sel.fit_transform(X, y)

    assert sel.features_to_drop_ == [f for f in X.columns if f not in selected]
    assert list(sel.performance_drifts_.keys()) == list(drifts.keys())
    assert _rounded(sel.performance_drifts_) == pytest.approx(drifts)
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {f: data_diabetes[f] for f in selected}


def test_performance_drifts_and_std(make_df, data_diabetes):
    X, y = _split_target(make_df, data_diabetes)
    sel = RecursiveFeatureElimination(LinearRegression(), scoring="r2", cv=3)
    sel.fit(X, y)

    assert sel.features_to_drop_ == ["var_0", "var_6", "var_7", "var_9"]
    assert _rounded(sel.performance_drifts_) == {
        "var_0": -0.0033,
        "var_9": -0.0003,
        "var_6": -0.0007,
        "var_7": 0.0001,
        "var_1": 0.012,
        "var_3": 0.0286,
        "var_5": 0.0126,
        "var_2": 0.0663,
        "var_8": 0.1094,
        "var_4": 0.0243,
    }
    assert _rounded(sel.performance_drifts_std_) == {
        "var_0": 0.0136,
        "var_9": 0.0168,
        "var_6": 0.0169,
        "var_7": 0.018,
        "var_1": 0.0252,
        "var_3": 0.0084,
        "var_5": 0.0087,
        "var_2": 0.0425,
        "var_8": 0.0468,
        "var_4": 0.0162,
    }


def test_feature_importances_are_sorted_ascending(make_df, data_diabetes):
    X, y = _split_target(make_df, data_diabetes)
    sel = RecursiveFeatureElimination(LinearRegression(), scoring="r2", cv=3)
    sel.fit(X, y)

    importance_type = pd.Series if make_df is pd.DataFrame else dict
    assert isinstance(sel.feature_importances_, importance_type)
    assert list(sel.feature_importances_.keys()) == [
        "var_0",
        "var_9",
        "var_6",
        "var_7",
        "var_1",
        "var_3",
        "var_5",
        "var_2",
        "var_8",
        "var_4",
    ]
    assert [round(v, 2) for v in dict(sel.feature_importances_).values()] == [
        41.42,
        64.77,
        113.97,
        182.17,
        238.62,
        322.09,
        436.67,
        522.33,
        741.47,
        750.02,
    ]
    # the standard deviation keeps the order of the variables
    assert list(sel.feature_importances_std_.keys()) == list(X.columns)
    assert [round(v, 2) for v in dict(sel.feature_importances_std_).values()] == [
        18.22,
        68.35,
        86.03,
        57.11,
        329.38,
        299.76,
        72.81,
        47.93,
        117.83,
        42.75,
    ]


def test_features_with_tied_importance_keep_their_order(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = RecursiveFeatureElimination(
        Lasso(alpha=0.01, random_state=1), scoring="r2", threshold=-100
    )
    sel.fit(X, y)

    assert list(sel.feature_importances_.keys()) == [
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_6",
        "var_9",
        "var_10",
        "var_11",
        "var_8",
        "var_7",
    ]
    assert list(sel.performance_drifts_.keys()) == list(
        sel.feature_importances_.keys()
    )
    assert sel.features_to_drop_ == []


def test_stops_when_only_one_feature_remains(make_df):
    # x is identical to the target and z is constant
    x = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    X = make_df({"x": x, "z": [1] * 10})
    sel = RecursiveFeatureElimination(LinearRegression(), scoring="r2", cv=3)
    Xt = sel.fit_transform(X, make_series(make_df, x))

    assert sel.features_to_drop_ == ["z"]
    assert sel.performance_drifts_ == {"z": 0.0, "x": 0}
    assert list(sel.performance_drifts_std_.keys()) == ["z"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"x": x}


def test_selection_with_permutation_importance(make_df, data_classification):
    # KNN has no coef_ or feature_importances_
    X, y = _split_target(make_df, data_classification)
    sel = RecursiveFeatureElimination(KNeighborsClassifier(), threshold=0.001)
    Xt = sel.fit_transform(X, y)

    assert sel.features_to_drop_ == [
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_6",
        "var_8",
        "var_9",
        "var_11",
    ]
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_7", "var_10"]


@pytest.mark.parametrize("cv_type", ["splitter", "generator"])
def test_cv_splitter_and_generator(make_df, data_diabetes, cv_type):
    X, y = _split_target(make_df, data_diabetes)
    cv = KFold(n_splits=3)
    if cv_type == "generator":
        cv = cv.split(X, y)
    sel = RecursiveFeatureElimination(LinearRegression(), scoring="r2", cv=cv)
    sel.fit(X, y)

    assert sel.features_to_drop_ == ["var_0", "var_6", "var_7", "var_9"]


def test_groups(make_df):
    rng = np.random.default_rng(1)
    data = {f"var_{i}": rng.normal(size=100).tolist() for i in range(5)}
    data["target"] = rng.integers(0, 100, size=100).tolist()
    groups = np.repeat(np.arange(10), 10)
    X, y = _split_target(make_df, data)

    cv = GroupKFold(n_splits=3)
    sel_splits = RecursiveFeatureElimination(
        LinearRegression(),
        scoring="neg_mean_absolute_error",
        cv=cv.split(X, y, groups=groups),
    )
    sel_splits.fit(X, y)
    sel = RecursiveFeatureElimination(
        LinearRegression(), scoring="neg_mean_absolute_error", cv=cv, groups=groups
    )
    sel.fit(X, y)

    assert sel.features_to_drop_ == sel_splits.features_to_drop_
    assert sel.performance_drifts_ == pytest.approx(sel_splits.performance_drifts_)


def test_only_numerical_variables_are_evaluated(make_df, data_classification):
    data = {f"var_{i}": data_classification[f"var_{i}"] for i in [0, 4, 7]}
    data["cat"] = ["a", "b"] * 500
    data["target"] = data_classification["target"]
    X, y = _split_target(make_df, data)
    sel = RecursiveFeatureElimination(LogisticRegression(), threshold=0.0005)
    Xt = sel.fit_transform(X, y)

    assert sel.variables_ == ["var_0", "var_4", "var_7"]
    assert sel.features_to_drop_ == ["var_4"]
    assert frame_to_dict(Xt) == {
        "var_0": data["var_0"],
        "cat": data["cat"],
        "var_7": data["var_7"],
    }


@pytest.mark.parametrize("target_type", [list, np.array])
def test_list_and_array_target(make_df, data_diabetes, target_type):
    X, _ = _split_target(make_df, data_diabetes)
    y = target_type(data_diabetes["target"])
    sel = RecursiveFeatureElimination(LinearRegression(), scoring="r2", cv=3)
    sel.fit(X, y)

    assert sel.features_to_drop_ == ["var_0", "var_6", "var_7", "var_9"]


def test_integer_column_names(load_diabetes_dataset):
    X, y = load_diabetes_dataset
    sel = RecursiveFeatureElimination(
        Lasso(alpha=0.001, random_state=10), cv=3, threshold=0.1, scoring="r2"
    )
    Xt = sel.fit_transform(X, y)

    assert sel.features_to_drop_ == [0, 1, 3, 4, 5, 6, 7, 9]
    assert list(sel.feature_importances_.index) == [0, 9, 6, 7, 1, 3, 5, 2, 4, 8]
    assert list(sel.performance_drifts_.keys()) == [0, 9, 6, 7, 1, 3, 5, 2, 4, 8]
    pd.testing.assert_frame_equal(Xt, X[[2, 8]])
