import numpy as np
import pandas as pd
# import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import RecursiveFeatureAddition

# TODO
# test performance_drifts_
# test features_to_drop
# the above with a mix of classification and regression and different scoring metrics


def test_classification_threshold_parameters(df_test):
    X, y = df_test

    sel = RecursiveFeatureAddition(
        RandomForestClassifier(random_state=1), threshold=0.001
    )

    sel.fit(X, y)

    # expected result
    Xtransformed = X[["var_7", "var_10"]].copy()

    # # expected ordered features by importance, from most important
    # # to least important
    # ordered_features = [
    #     "var_7",
    #     "var_4",
    #     "var_6",
    #     "var_9",
    #     "var_0",
    #     "var_8",
    #     "var_1",
    #     "var_10",
    #     "var_5",
    #     "var_11",
    #     "var_2",
    #     "var_3",
    # ]

    # test fit attrs
    assert np.round(sel.initial_model_performance_, 3) == 0.997
    # assert sel.feature_importances_ ==
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
    assert len(sel.performance_drifts_.keys()) == len(X.columns)
    assert all([var in sel.performance_drifts_.keys() for var in X.columns])
    assert sel.n_features_in_ == len(X.columns)

    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_3_and_r2(load_diabetes_dataset):
    #  test for regression using cv=3, and the r2 as metric.
    X, y = load_diabetes_dataset

    kfold = KFold(n_splits=3, shuffle=True, random_state=10)
    sel = RecursiveFeatureAddition(
        estimator=LinearRegression(), scoring="r2", cv=kfold, threshold=0.001
    )
    sel.fit(X, y)

    # expected output
    Xtransformed = X[[1, 2, 3, 6, 8]].copy()

    # expected ordered features by importance, from most important
    # to least important
    ordered_features = [4, 8, 2, 5, 3, 1, 7, 6, 9, 0]

    # test init params
    # assert sel.cv == 3
    assert sel.variables is None
    assert sel.scoring == "r2"
    assert sel.threshold == 0.001
    # fit params
    assert sel.variables_ == list(X.columns)
    assert np.round(sel.initial_model_performance_, 2) == 0.49
    print(sel.performance_drifts_)
    assert sel.features_to_drop_ == [0, 4, 5, 7, 9]
    assert len(sel.performance_drifts_.keys()) == len(ordered_features)
    assert all([var in sel.performance_drifts_.keys() for var in ordered_features])

    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_2_and_mse(load_diabetes_dataset):
    #  test for regression using cv=2, and the neg_mean_squared_error as metric.
    # add suitable threshold for regression mse
    X, y = load_diabetes_dataset

    kfold = KFold(n_splits=2, shuffle=True, random_state=10)
    sel = RecursiveFeatureAddition(
        estimator=DecisionTreeRegressor(random_state=0),
        scoring="neg_mean_squared_error",
        cv=kfold,
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
    assert sel.variables is None
    assert sel.scoring == "neg_mean_squared_error"
    assert sel.threshold == 10
    # fit params
    assert sel.variables_ == list(X.columns)
    assert np.round(sel.initial_model_performance_, 0) == -5836.0
    assert sel.features_to_drop_ == [0, 3, 4, 5, 6, 8, 9]
    assert list(sel.performance_drifts_.keys()) == ordered_features
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)
