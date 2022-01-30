import numpy as np
import pandas as pd
# import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import RecursiveFeatureElimination

# TODO
# test performance_drifts_
# test features_to_drop
# the above with a mix of classification and regression and different scoring metrics


def test_classification_threshold_parameters(df_test):
    X, y = df_test
    sel = RecursiveFeatureElimination(
        RandomForestClassifier(random_state=1), threshold=0.001
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = X[["var_0", "var_6"]].copy()

    # expected ordred features by importance
    ordered_features = [
        "var_3",
        "var_2",
        "var_11",
        "var_5",
        "var_10",
        "var_1",
        "var_8",
        "var_0",
        "var_9",
        "var_6",
        "var_4",
        "var_7",
    ]

    # test init params
    assert sel.variables is None
    assert sel.threshold == 0.001
    assert sel.cv == 3
    assert sel.scoring == "roc_auc"
    # test fit attrs
    assert sel.variables_ == [
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_6",
        "var_7",
        "var_8",
        "var_9",
        "var_10",
        "var_11",
    ]
    assert np.round(sel.initial_model_performance_, 3) == 0.997
    assert sel.features_to_drop_ == [
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_7",
        "var_8",
        "var_9",
        "var_10",
        "var_11",
    ]
    assert list(sel.performance_drifts_.keys()) == ordered_features
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_3_and_r2(load_diabetes_dataset):
    #  test for regression using cv=3, and the r2 as metric.
    X, y = load_diabetes_dataset
    sel = RecursiveFeatureElimination(estimator=LinearRegression(), scoring="r2", cv=3)
    sel.fit(X, y)

    # expected output
    Xtransformed = X[[1, 2, 3, 4, 5, 8]].copy()

    # expected ordred features by importance
    ordered_features = [0, 9, 6, 7, 1, 3, 5, 2, 8, 4]

    # test init params
    assert sel.cv == 3
    assert sel.variables is None
    assert sel.scoring == "r2"
    assert sel.threshold == 0.01
    # fit params
    assert sel.variables_ == list(X.columns)
    assert np.round(sel.initial_model_performance_, 3) == 0.489
    assert sel.features_to_drop_ == [0, 6, 7, 9]
    assert list(sel.performance_drifts_.keys()) == ordered_features
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_2_and_mse(load_diabetes_dataset):
    #  test for regression using cv=2, and the neg_mean_squared_error as metric.
    # add suitable threshold for regression mse

    X, y = load_diabetes_dataset
    sel = RecursiveFeatureElimination(
        estimator=DecisionTreeRegressor(random_state=0),
        scoring="neg_mean_squared_error",
        cv=2,
        threshold=10,
    )
    # fit transformer
    sel.fit(X, y)

    # expected output
    Xtransformed = X[[0, 2, 3, 5, 6, 7, 8, 9]].copy()

    # expected ordred features by importance
    ordered_features = [1, 0, 4, 6, 9, 3, 7, 5, 8, 2]

    # test init params
    assert sel.cv == 2
    assert sel.variables is None
    assert sel.scoring == "neg_mean_squared_error"
    assert sel.threshold == 10
    # fit params
    assert sel.variables_ == list(X.columns)
    assert np.round(sel.initial_model_performance_, 0) == -5836.0
    assert sel.features_to_drop_ == [1, 4]
    assert list(sel.performance_drifts_.keys()) == ordered_features
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)
