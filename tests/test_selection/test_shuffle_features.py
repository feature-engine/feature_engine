import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import SelectByShuffling


def test_sel_with_default_parameters(df_test):
    X, y = df_test
    sel = SelectByShuffling(
        RandomForestClassifier(random_state=1), threshold=0.01, random_state=1
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = pd.DataFrame(X["var_7"].copy())

    # test init params
    assert sel.threshold == 0.01
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
        "var_10",
        "var_11",
    ]
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_3_and_r2(load_diabetes_dataset):
    #  test for regression using cv=3, and the r2 as metric.
    X, y = load_diabetes_dataset

    sel = SelectByShuffling(
        estimator=LinearRegression(), scoring="r2", cv=3, threshold=0.05, random_state=1
    )
    sel.fit(X, y)

    # expected output
    Xtransformed = X[[1, 2, 3, 4, 5, 8]].copy()

    # test init params
    assert sel.cv == 3
    assert sel.scoring == "r2"
    assert sel.threshold == 0.05
    # fit params
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
        threshold=1000,
        random_state=1,
    )
    # fit transformer
    sel.fit(X, y)

    # expected output
    Xtransformed = X[[2, 7, 8]].copy()

    # test init params
    assert sel.cv == 2
    assert sel.scoring == "neg_mean_squared_error"
    assert sel.threshold == 1000
    # fit params
    assert np.round(sel.initial_model_performance_, 0) == -5836.0
    assert sel.features_to_drop_ == [0, 1, 3, 4, 5, 6, 9]
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_cv_generator(df_test):
    X, y = df_test
    cv = StratifiedKFold(n_splits=3)

    X, y = df_test
    sel = SelectByShuffling(
        RandomForestClassifier(random_state=1),
        threshold=0.01,
        random_state=1,
        cv=3,
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = pd.DataFrame(X["var_7"].copy())
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)

    sel = SelectByShuffling(
        RandomForestClassifier(random_state=1),
        threshold=0.01,
        random_state=1,
        cv=cv,
    )
    sel.fit(X, y)
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)

    sel = SelectByShuffling(
        RandomForestClassifier(random_state=1),
        threshold=0.01,
        random_state=1,
        cv=cv.split(X, y),
    )
    sel.fit(X, y)
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


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
    assert sel.threshold == 0.01
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
        "var_10",
        "var_11",
    ]
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_sample_weights():
    X = pd.DataFrame(
        dict(
            x1=[1000, 2000, 1000, 1000, 2000, 3000],
            x2=[1000, 2000, 1000, 1000, 2000, 3000],
        )
    )
    y = pd.Series([1, 0, 0, 1, 1, 0])

    sbs = SelectByShuffling(
        RandomForestClassifier(random_state=42), cv=2, random_state=42
    )

    sample_weight = [1000, 2000, 1000, 1000, 2000, 3000]
    sbs.fit_transform(X, y, sample_weight=sample_weight)
    assert sbs.initial_model_performance_ == 0.125
