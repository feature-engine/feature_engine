import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import RecursiveFeatureAddition


def test_classification_threshold_parameters(df_test):
    X, y = df_test

    sel = RecursiveFeatureAddition(
        RandomForestClassifier(random_state=1), threshold=0.001
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = X[["var_7", "var_10"]].copy()

    # expected ordered features by importance, from most important
    # to least important
    ordered_features = [
        "var_7",
        "var_4",
        "var_6",
        "var_9",
        "var_0",
        "var_8",
        "var_1",
        "var_10",
        "var_5",
        "var_11",
        "var_2",
        "var_3",
    ]

    # test init params
    assert sel.variables == [
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
    assert sel.threshold == 0.001
    assert sel.cv == 3
    assert sel.scoring == "roc_auc"
    # test fit attrs
    assert np.round(sel.initial_model_performance_, 3) == 0.997
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
    assert list(sel.performance_drifts_.keys()) == ordered_features
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_3_and_r2(load_diabetes_dataset):
    #  test for regression using cv=3, and the r2 as metric.
    X, y = load_diabetes_dataset

    sel = RecursiveFeatureAddition(estimator=LinearRegression(), scoring="r2", cv=3)
    sel.fit(X, y)

    # expected output
    Xtransformed = X[[2, 3, 4, 8]].copy()

    # expected ordred features by importance, from most important
    # to least important
    ordered_features = [4, 8, 2, 5, 3, 1, 7, 6, 9, 0]

    # test init params
    assert sel.cv == 3
    assert sel.variables == list(X.columns)
    assert sel.scoring == "r2"
    assert sel.threshold == 0.01
    # fit params
    assert np.round(sel.initial_model_performance_, 3) == 0.489
    assert sel.features_to_drop_ == [0, 1, 5, 6, 7, 9]
    assert list(sel.performance_drifts_.keys()) == ordered_features
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_2_and_mse(load_diabetes_dataset):
    #  test for regression using cv=2, and the neg_mean_squared_error as metric.
    # add suitable threshold for regression mse
    X, y = load_diabetes_dataset

    sel = RecursiveFeatureAddition(
        estimator=DecisionTreeRegressor(random_state=0),
        scoring="neg_mean_squared_error",
        cv=2,
        threshold=10,
    )
    # fit transformer
    sel.fit(X, y)

    # expected output
    Xtransformed = X[[1, 2, 7]].copy()

    # expected ordred features by importance, from most important
    # to least important
    ordered_features = [2, 8, 5, 7, 3, 9, 6, 4, 0, 1]

    # test init params
    assert sel.cv == 2
    assert sel.variables == list(X.columns)
    assert sel.scoring == "neg_mean_squared_error"
    assert sel.threshold == 10
    # fit params
    assert np.round(sel.initial_model_performance_, 0) == -5836.0
    assert sel.features_to_drop_ == [0, 3, 4, 5, 6, 8, 9]
    assert list(sel.performance_drifts_.keys()) == ordered_features
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_non_fitted_error(df_test):
    # when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        sel = RecursiveFeatureAddition()
        sel.transform(df_test)


def test_raises_cv_error():
    with pytest.raises(ValueError):
        RecursiveFeatureAddition(cv=0)


def test_raises_threshold_error():
    with pytest.raises(ValueError):
        RecursiveFeatureAddition(threshold=None)


def test_automatic_variable_selection(df_test):
    X, y = df_test

    # add 2 additional categorical variables, these should not be evaluated by
    # the selector
    X["cat_1"] = "cat1"
    X["cat_2"] = "cat2"

    sel = RecursiveFeatureAddition(
        RandomForestClassifier(random_state=1), threshold=0.001
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = X[["var_7", "var_10", "cat_1", "cat_2"]].copy()

    # expected ordered features by importance, from most important
    # to least important
    ordered_features = [
        "var_7",
        "var_4",
        "var_6",
        "var_9",
        "var_0",
        "var_8",
        "var_1",
        "var_10",
        "var_5",
        "var_11",
        "var_2",
        "var_3",
    ]

    # test init params
    assert sel.variables == [
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
    assert sel.threshold == 0.001
    assert sel.cv == 3
    assert sel.scoring == "roc_auc"
    # test fit attrs
    assert np.round(sel.initial_model_performance_, 3) == 0.997
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
    assert list(sel.performance_drifts_.keys()) == ordered_features
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)
