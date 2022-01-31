import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import RecursiveFeatureElimination

# tests for classification
_model_and_expectations = [
    (
        RandomForestClassifier(n_estimators=5, random_state=1),
        3,
        0.001,
        "roc_auc",
        [
            "var_1",
            "var_2",
            "var_3",
            "var_5",
            "var_6",
            "var_7",
            "var_8",
            "var_9",
            "var_10",
            "var_11",
        ],
        {
            "var_5": -0.0,
            "var_3": 0.0009,
            "var_2": -0.0001,
            "var_1": -0.002,
            "var_11": 0.001,
            "var_10": 0.0009,
            "var_8": 0.0001,
            "var_0": 0.0019,
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
        [
            "var_1",
            "var_2",
            "var_3",
            "var_4",
            "var_5",
            "var_6",
            "var_9",
            "var_10",
            "var_11",
        ],
        {
            "var_2": 0.0,
            "var_9": 0.0,
            "var_10": 0.0,
            "var_3": 0.0,
            "var_5": -0.001,
            "var_1": 0.0,
            "var_11": 0.0,
            "var_4": -0.001,
            "var_6": -0.001,
            "var_0": 0.002,
            "var_8": 0.002,
            "var_7": 0.004,
        },
    ),
]


@pytest.mark.parametrize(
    "estimator, cv, threshold, scoring, dropped_features, performances",
    _model_and_expectations,
)
def test_classification(
    estimator, cv, threshold, scoring, dropped_features, performances, df_test
):
    X, y = df_test

    sel = RecursiveFeatureElimination(
        estimator=estimator, cv=cv, threshold=threshold, scoring=scoring
    )

    sel.fit(X, y)

    Xtransformed = X.copy()
    Xtransformed = Xtransformed.drop(labels=dropped_features, axis=1)

    # test fit attrs
    assert sel.features_to_drop_ == dropped_features

    assert len(sel.performance_drifts_.keys()) == len(X.columns)
    assert all([var in sel.performance_drifts_.keys() for var in X.columns])
    rounded_perfs = {
        key: round(sel.performance_drifts_[key], 4) for key in sel.performance_drifts_
    }
    assert rounded_perfs == performances

    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


# tests for regression
_model_and_expectations = [
    (
        Lasso(alpha=0.001, random_state=10),
        3,
        0.1,
        "r2",
        [0, 1, 3, 4, 5, 6, 7, 9],
        {
            0: -0.0032,
            9: -0.0003,
            6: -0.0008,
            7: 0.0001,
            1: 0.012,
            3: 0.0199,
            5: 0.0023,
            2: 0.1378,
            4: 0.0069,
            8: 0.115,
        },
    ),
    (
        DecisionTreeRegressor(random_state=10),
        2,
        100,
        "neg_mean_squared_error",
        [1, 4],
        {
            1: 64.1018,
            4: -199.9864,
            0: 481.6109,
            6: 282.4231,
            9: 699.7964,
            3: 327.1403,
            7: 246.4412,
            5: 436.9751,
            8: 350.4163,
            2: 1340.0226,
        },
    ),
]


@pytest.mark.parametrize(
    "estimator, cv, threshold, scoring, dropped_features, performances",
    _model_and_expectations,
)
def test_regression(
    estimator,
    cv,
    threshold,
    scoring,
    dropped_features,
    performances,
    load_diabetes_dataset,
):
    #  test for regression using cv=3, and the r2 as metric.
    X, y = load_diabetes_dataset

    sel = RecursiveFeatureElimination(
        estimator=estimator, cv=cv, threshold=threshold, scoring=scoring
    )

    sel.fit(X, y)

    Xtransformed = X.copy()
    Xtransformed = Xtransformed.drop(labels=dropped_features, axis=1)

    # test fit attrs
    assert sel.features_to_drop_ == dropped_features

    assert len(sel.performance_drifts_.keys()) == len(X.columns)
    assert all([var in sel.performance_drifts_.keys() for var in X.columns])
    rounded_perfs = {
        key: round(sel.performance_drifts_[key], 4) for key in sel.performance_drifts_
    }
    assert rounded_perfs == performances

    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)
