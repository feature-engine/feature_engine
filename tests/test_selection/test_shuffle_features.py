import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import SelectByShuffling


def test_default_parameters(df_test):
    X, y = df_test
    sel = SelectByShuffling(
        RandomForestClassifier(random_state=1), threshold=0.01, random_state=1
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = pd.DataFrame(X["var_7"].copy())

    # test init params
    assert sel.variables is None
    assert sel.threshold == 0.01
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
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_6",
        "var_8",
        "var_9",
        "var_10",
        "var_11",
    ]
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_3_and_r2(load_diabetes_dataset):
    #  test for regression using cv=3, and the r2 as metric.
    X, y = load_diabetes_dataset
    sel = SelectByShuffling(
        estimator=LinearRegression(), scoring="r2", cv=3, threshold=0.01, random_state=1
    )
    sel.fit(X, y)

    # expected output
    Xtransformed = X[[1, 2, 3, 4, 5, 8]].copy()

    # test init params
    assert sel.cv == 3
    assert sel.variables is None
    assert sel.scoring == "r2"
    assert sel.threshold == 0.01
    # fit params
    assert sel.variables_ == list(X.columns)
    assert np.round(sel.initial_model_performance_, 3) == 0.489
    assert sel.features_to_drop_ == [0, 6, 7, 9]
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_2_and_mse(load_diabetes_dataset):
    #  test for regression using cv=2, and the neg_mean_squared_error as metric.
    # add suitable threshold for regression mse
    X, y = load_diabetes_dataset

    sel = SelectByShuffling(
        estimator=DecisionTreeRegressor(random_state=0),
        scoring="neg_mean_squared_error",
        cv=2,
        threshold=5,
        random_state=1,
    )
    # fit transformer
    sel.fit(X, y)

    # expected output
    Xtransformed = X[[2, 8]].copy()

    # test init params
    assert sel.cv == 2
    assert sel.variables is None
    assert sel.scoring == "neg_mean_squared_error"
    assert sel.threshold == 5
    # fit params
    assert sel.variables_ == list(X.columns)
    assert np.round(sel.initial_model_performance_, 0) == -5836.0
    assert sel.features_to_drop_ == [0, 1, 3, 4, 5, 6, 7, 9]
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_non_fitted_error(df_test):
    # when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        sel = SelectByShuffling(RandomForestClassifier(random_state=1))
        sel.transform(df_test)


def test_raises_threshold_error():
    with pytest.raises(ValueError):
        SelectByShuffling(RandomForestClassifier(random_state=1), threshold="hello")


def test_automatic_variable_selection(df_test):
    X, y = df_test
    # add 2 additional categorical variables, these should not be evaluated by
    # the selector
    X["cat_1"] = "cat1"
    X["cat_2"] = "cat2"

    sel = SelectByShuffling(
        RandomForestClassifier(random_state=1), threshold=0.01, random_state=1
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = X[["var_7", "cat_1", "cat_2"]].copy()

    # test init params
    assert sel.variables is None
    assert sel.threshold == 0.01
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
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_6",
        "var_8",
        "var_9",
        "var_10",
        "var_11",
    ]
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_KFold_generators(df_test):

    X, y = df_test

    # Kfold
    sel = SelectByShuffling(
        RandomForestClassifier(random_state=1),
        threshold=0.01,
        random_state=1,
        cv=KFold(n_splits=3),
    )
    sel.fit(X, y)
    Xtransformed = sel.transform(X)

    # test fit attrs
    assert sel.initial_model_performance_ > 0.995
    assert isinstance(sel.features_to_drop_, list)
    assert all([x for x in sel.features_to_drop_ if x in X.columns])
    assert len(sel.features_to_drop_) < X.shape[1]
    assert not Xtransformed.empty
    assert all([x for x in Xtransformed.columns if x not in sel.features_to_drop_])
    assert isinstance(sel.performance_drifts_, dict)
    assert all([x for x in X.columns if x in sel.performance_drifts_.keys()])
    assert all(
        [
            isinstance(sel.performance_drifts_[var], (int, float))
            for var in sel.performance_drifts_.keys()
        ]
    )

    # Stratfied
    sel = SelectByShuffling(
        RandomForestClassifier(random_state=1),
        threshold=0.01,
        random_state=1,
        cv=StratifiedKFold(n_splits=3),
    )
    sel.fit(X, y)
    Xtransformed = sel.transform(X)

    # test fit attrs
    assert sel.initial_model_performance_ > 0.995
    assert isinstance(sel.features_to_drop_, list)
    assert all([x for x in sel.features_to_drop_ if x in X.columns])
    assert len(sel.features_to_drop_) < X.shape[1]
    assert not Xtransformed.empty
    assert all([x for x in Xtransformed.columns if x not in sel.features_to_drop_])
    assert isinstance(sel.performance_drifts_, dict)
    assert all([x for x in X.columns if x in sel.performance_drifts_.keys()])
    assert all(
        [
            isinstance(sel.performance_drifts_[var], (int, float))
            for var in sel.performance_drifts_.keys()
        ]
    )

    # None
    sel = SelectByShuffling(
        RandomForestClassifier(random_state=1),
        threshold=0.01,
        random_state=1,
        cv=None,
    )
    sel.fit(X, y)
    Xtransformed = sel.transform(X)

    # test fit attrs
    assert sel.initial_model_performance_ > 0.995
    assert isinstance(sel.features_to_drop_, list)
    assert all([x for x in sel.features_to_drop_ if x in X.columns])
    assert len(sel.features_to_drop_) < X.shape[1]
    assert not Xtransformed.empty
    assert all([x for x in Xtransformed.columns if x not in sel.features_to_drop_])
    assert isinstance(sel.performance_drifts_, dict)
    assert all([x for x in X.columns if x in sel.performance_drifts_.keys()])
    assert all(
        [
            isinstance(sel.performance_drifts_[var], (int, float))
            for var in sel.performance_drifts_.keys()
        ]
    )
