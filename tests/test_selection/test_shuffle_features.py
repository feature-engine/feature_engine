import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import ShuffleFeaturesSelector


@pytest.fixture(scope="module")
def df_test():
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_redundant=4,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )

    # trasform arrays into pandas df and series
    colnames = ["var_" + str(i) for i in range(12)]
    X = pd.DataFrame(X, columns=colnames)
    y = pd.Series(y)
    return X, y


@pytest.fixture(scope="module")
def load_diabetes_dataset():
    # Load the diabetes dataset from sklearn
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(diabetes_X)
    y = pd.DataFrame(diabetes_y)
    return X, y


def test_default_parameters(df_test):
    X, y = df_test
    sel = ShuffleFeaturesSelector(RandomForestClassifier(random_state=1))
    sel.fit(X, y)

    # expected result
    Xtransformed = pd.DataFrame(X["var_7"].copy())

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
    assert sel.threshold == 0.01
    assert sel.cv == 3
    assert sel.scoring == "roc_auc"
    # test fit attrs
    assert np.round(sel.initial_model_performance_, 3) == 0.997
    assert sel.selected_features_ == ["var_7"]
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_3_and_r2(load_diabetes_dataset):
    #  test for regression using cv=3, and the r2 as metric.
    X, y = load_diabetes_dataset
    sel = ShuffleFeaturesSelector(estimator=LinearRegression(), scoring="r2", cv=3)
    sel.fit(X, y)

    # expected output
    Xtransformed = pd.DataFrame(X[[1, 2, 3, 4, 5, 8]].copy())

    # test init params
    assert sel.cv == 3
    assert sel.variables == list(X.columns)
    assert sel.scoring == "r2"
    assert sel.threshold == 0.01
    # fit params
    assert np.round(sel.initial_model_performance_, 3) == 0.489
    assert sel.selected_features_ == [1, 2, 3, 4, 5, 8]
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_2_and_mse(load_diabetes_dataset):
    #  test for regression using cv=2, and the neg_mean_squared_error as metric.
    # add suitable threshold for regression mse

    X, y = load_diabetes_dataset
    sel = ShuffleFeaturesSelector(
        estimator=DecisionTreeRegressor(random_state=0),
        scoring="neg_mean_squared_error",
        cv=2,
        threshold=10,
    )
    # fit transformer
    sel.fit(X, y)

    # expected output
    Xtransformed = pd.DataFrame(X[[2, 8]].copy())

    # test init params
    assert sel.cv == 2
    assert sel.variables == list(X.columns)
    assert sel.scoring == "neg_mean_squared_error"
    assert sel.threshold == 10
    # fit params
    assert np.round(sel.initial_model_performance_, 0) == -5836.0
    assert sel.selected_features_ == [2, 8]
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_non_fitted_error(df_test):
    # when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        sel = ShuffleFeaturesSelector()
        sel.transform(df_test)


def test_raises_cv_error():
    with pytest.raises(ValueError):
        ShuffleFeaturesSelector(cv=0)


def test_raises_threshold_error():
    with pytest.raises(ValueError):
        ShuffleFeaturesSelector(threshold=None)
