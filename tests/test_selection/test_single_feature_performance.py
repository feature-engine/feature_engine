import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GroupKFold

from feature_engine.selection import SelectBySingleFeaturePerformance


def test_sel_with_default_parameters(df_test):
    X, y = df_test
    sel = SelectBySingleFeaturePerformance(
        RandomForestClassifier(random_state=1), threshold=0.5
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = X.copy()
    Xtransformed.drop(columns=["var_3", "var_10"], inplace=True)

    # test init params
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
    assert sel.scoring == "neg_mean_squared_error"
    assert sel.threshold == -6000
    # fit params
    assert sel.features_to_drop_ == [0, 2, 3, 4, 5, 6, 8, 9]
    rounded_perfs = {
        key: round(sel.feature_performance_[key], 2) for key in sel.feature_performance_
    }
    assert rounded_perfs == {
        0: -7657.15,
        1: -5966.66,
        2: -6613.78,
        3: -6488.93,
        4: -9415.59,
        5: -11761.0,
        6: -6592.58,
        7: -5270.56,
        8: -7547.2,
        9: -6287.56,
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
        sel.fit(X, y)


def test_raises_threshold_error():
    with pytest.raises(ValueError):
        SelectBySingleFeaturePerformance(
            RandomForestClassifier(random_state=1),
            threshold="hola",
        )


def test_raises_error_when_roc_threshold_not_allowed():
    with pytest.raises(ValueError):
        SelectBySingleFeaturePerformance(
            RandomForestClassifier(random_state=1), scoring="roc_auc", threshold=0.4
        )


def test_raises_error_when_r2_threshold_not_allowed():
    with pytest.raises(ValueError):
        SelectBySingleFeaturePerformance(
            RandomForestClassifier(random_state=1), scoring="r2", threshold=4
        )


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


def test_raises_error_if_evaluating_single_variable_and_threshold_is_None(df_test):
    X, y = df_test

    sel = SelectBySingleFeaturePerformance(
        RandomForestClassifier(random_state=1), variables=["var_1"], threshold=None
    )

    with pytest.raises(ValueError):
        sel.fit(X, y)


def test_test_selector_with_one_variable(df_test):
    X, y = df_test

    sel = SelectBySingleFeaturePerformance(
        RandomForestClassifier(random_state=1),
        variables=["var_0"],
        threshold=0.5,
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = X.copy()

    assert sel.features_to_drop_ == []
    assert sel.feature_performance_ == {
        "var_0": 0.5957642619540211,
    }
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)

    sel = SelectBySingleFeaturePerformance(
        RandomForestClassifier(random_state=1),
        variables=["var_3"],
        threshold=0.5,
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = X.copy()
    Xtransformed.drop(columns=["var_3"], inplace=True)

    assert sel.features_to_drop_ == ["var_3"]
    assert sel.feature_performance_ == {
        "var_3": 0.4752954458526748,
    }
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_single_feature_importance_with_groups(df_test_with_groups):
    X, y, groups = df_test_with_groups
    cv = GroupKFold(n_splits=3)
    cv_indices = cv.split(X=X, y=y, groups=groups)

    estimator = RandomForestRegressor(n_estimators=3, random_state=3)
    scoring = "neg_mean_absolute_error"

    sel_expected = SelectBySingleFeaturePerformance(
        estimator=estimator,
        scoring=scoring,
        cv=cv_indices,
    )

    X_tr_expected = sel_expected.fit_transform(X, y)

    sel = SelectBySingleFeaturePerformance(
        estimator=estimator,
        scoring=scoring,
        cv=cv,
        groups=groups,
    )

    X_tr = sel.fit_transform(X, y)

    pd.testing.assert_frame_equal(X_tr_expected, X_tr)
