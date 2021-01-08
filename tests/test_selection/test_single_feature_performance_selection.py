import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import SelectBySingleFeaturePerformance


def test_default_parameters(df_test):
    X, y = df_test
    sel = SelectBySingleFeaturePerformance(
        RandomForestClassifier(random_state=1), threshold=0.5
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = X.copy()
    Xtransformed.drop(columns=["var_3", "var_10"], inplace=True)

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
    assert sel.threshold == 0.5
    assert sel.cv == 3
    assert sel.scoring == "roc_auc"
    # test fit attrs
    assert sel.features_to_drop_ == ["var_3", "var_10"]
    assert sel.feature_performance_ == {
        "var_0": 0.5957642619540211,
        "var_1": 0.5365534287221033,
        "var_2": 0.5001855546283257,
        "var_3": 0.4752954458526748,
        "var_4": 0.9780875304971691,
        "var_5": 0.5065441419357082,
        "var_6": 0.9758243290622809,
        "var_7": 0.994571685008432,
        "var_8": 0.5164434795458892,
        "var_9": 0.9543427678969847,
        "var_10": 0.47404183834906727,
        "var_11": 0.5227164067525513,
    }
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_3_and_r2(load_diabetes_dataset):
    #  test for regression using cv=3, and the r2 as metric.
    X, y = load_diabetes_dataset
    sel = SelectBySingleFeaturePerformance(
        estimator=LinearRegression(), scoring="r2", cv=3, threshold=0.01
    )
    sel.fit(X, y)

    # expected output
    Xtransformed = pd.DataFrame(X[[0, 2, 3, 4, 5, 6, 7, 8, 9]].copy())

    performance_dict = {
        0: 0.029,
        1: -0.004,
        2: 0.337,
        3: 0.192,
        4: 0.037,
        5: 0.018,
        6: 0.152,
        7: 0.177,
        8: 0.315,
        9: 0.139,
    }

    # test init params
    assert sel.cv == 3
    assert sel.variables == list(X.columns)
    assert sel.scoring == "r2"
    assert sel.threshold == 0.01
    # fit params
    assert sel.features_to_drop_ == [1]
    assert all(
        np.round(sel.feature_performance_[f], 3) == performance_dict[f]
        for f in sel.feature_performance_.keys()
    )
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_2_and_mse(load_diabetes_dataset):
    #  test for regression using cv=2, and the neg_mean_squared_error as metric.
    # add suitable threshold for regression mse

    X, y = load_diabetes_dataset
    sel = SelectBySingleFeaturePerformance(
        estimator=DecisionTreeRegressor(random_state=0),
        scoring="neg_mean_squared_error",
        cv=2,
        threshold=-6000,
    )
    # fit transformer
    sel.fit(X, y)

    # expected output
    Xtransformed = X.copy()
    Xtransformed = Xtransformed[[1, 7]].copy()

    # test init params
    assert sel.cv == 2
    assert sel.variables == list(X.columns)
    assert sel.scoring == "neg_mean_squared_error"
    assert sel.threshold == -6000
    # fit params
    assert sel.features_to_drop_ == [0, 2, 3, 4, 5, 6, 8, 9]
    assert sel.feature_performance_ == {
        0: -7657.154138192973,
        1: -5966.662211695372,
        2: -6613.779604700854,
        3: -6502.621725718592,
        4: -9415.586278197177,
        5: -11760.999622926094,
        6: -6592.584431571728,
        7: -5270.563893676307,
        8: -7641.414795123177,
        9: -6287.557824391035,
    }
    # test transform output
    print(sel.transform(X))
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_raises_warning_if_no_feature_selected(load_diabetes_dataset):
    X, y = load_diabetes_dataset
    sel = SelectBySingleFeaturePerformance(
        estimator=DecisionTreeRegressor(random_state=0),
        scoring="neg_mean_squared_error",
        cv=2,
        threshold=10,
    )
    with pytest.warns(UserWarning):
        warnings.warn(sel.fit(X, y), UserWarning)


def test_non_fitted_error(df_test):
    # when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        sel = SelectBySingleFeaturePerformance()
        sel.transform(df_test)


def test_raises_cv_error():
    with pytest.raises(ValueError):
        SelectBySingleFeaturePerformance(cv=0)


def test_raises_threshold_error():
    with pytest.raises(ValueError):
        SelectBySingleFeaturePerformance(threshold="hola")


def test_raises_error_when_roc_threshold_not_allowed():
    with pytest.raises(ValueError):
        SelectBySingleFeaturePerformance(scoring="roc_auc", threshold=0.4)


def test_raises_error_when_r2_threshold_not_allowed():
    with pytest.raises(ValueError):
        SelectBySingleFeaturePerformance(scoring="r2", threshold=4)


def test_automatic_variable_selection(df_test):
    X, y = df_test
    # add 2 additional categorical variables, these should not be evaluated by
    # the selector
    X["cat_1"] = "cat1"
    X["cat_2"] = "cat2"

    sel = SelectBySingleFeaturePerformance(
        RandomForestClassifier(random_state=1), threshold=0.5
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = X.copy()
    Xtransformed.drop(columns=["var_3", "var_10"], inplace=True)

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
    assert sel.threshold == 0.5
    assert sel.cv == 3
    assert sel.scoring == "roc_auc"
    # test fit attrs
    assert sel.features_to_drop_ == ["var_3", "var_10"]
    assert sel.feature_performance_ == {
        "var_0": 0.5957642619540211,
        "var_1": 0.5365534287221033,
        "var_2": 0.5001855546283257,
        "var_3": 0.4752954458526748,
        "var_4": 0.9780875304971691,
        "var_5": 0.5065441419357082,
        "var_6": 0.9758243290622809,
        "var_7": 0.994571685008432,
        "var_8": 0.5164434795458892,
        "var_9": 0.9543427678969847,
        "var_10": 0.47404183834906727,
        "var_11": 0.5227164067525513,
    }
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)
