import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import RecursiveFeatureAddition

# tests for classification
_model_and_expectations = [
    (
        RandomForestClassifier(n_estimators=5, random_state=1),
        3,
        0.001,
        "roc_auc",
        [
            "var_0",
            "var_1",
            "var_2",
            "var_3",
            "var_5",
            "var_6",
            "var_8",
            "var_9",
            "var_10",
            "var_11",
        ],
        {
            "var_4": 0,
            "var_7": 0.0241,
            "var_6": -0.001,
            "var_9": -0.001,
            "var_0": -0.0,
            "var_8": -0.0011,
            "var_10": -0.0011,
            "var_11": -0.001,
            "var_1": -0.0,
            "var_2": -0.0001,
            "var_3": -0.0011,
            "var_5": -0.0001,
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
            "var_7": 0,
            "var_8": 0.001,
            "var_0": 0.002,
            "var_6": -0.001,
            "var_4": 0.0,
            "var_11": -0.001,
            "var_1": -0.001,
            "var_5": -0.003,
            "var_3": -0.002,
            "var_10": 0.0,
            "var_9": 0.0,
            "var_2": 0.0,
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

    sel = RecursiveFeatureAddition(
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
            8: 0,
            4: 0.0059,
            2: 0.1367,
            5: -0.0026,
            3: 0.0177,
            1: -0.0045,
            7: -0.0035,
            6: 0.0088,
            9: 0.002,
            0: -0.0114,
        },
    ),
    (
        DecisionTreeRegressor(random_state=10),
        2,
        100,
        "neg_mean_squared_error",
        [0, 3, 4, 5, 6, 7, 8, 9],
        {
            0: -1693.8544,
            1: 106.9272,
            2: 0,
            3: -222.2945,
            4: -781.6593,
            5: -943.6299,
            6: -1701.2939,
            7: 99.315,
            8: -660.1107,
            9: -716.6378,
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

    sel = RecursiveFeatureAddition(
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
