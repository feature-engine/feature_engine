import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

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


def test_non_fitted_error(df_test):
    # test case 3: when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        transformer = ShuffleFeaturesSelector()
        transformer.transform(df_test)
        
def test_regression_cv_3(df_test):
    #  test for regression using cv=3, and the r2 as metric.
    
    # Load the diabetes dataset from sklearn
    diabetes_X, diabetes_y = load_diabetes(return_X_y=True)
    data = pd.DataFrame(diabetes_X)
    target = pd.DataFrame(diabetes_y)
    # initialize linear regresion estimator
    linear_model = LinearRegression()
    # initialize transformer
    transformer = ShuffleFeaturesSelector(estimator=linear_model, scoring='r2', cv = 3)
    # fit transformer
    X = transformer.fit_transform(data, target)

    # initialization parameters
    assert transformer.cv == 3
    assert transformer.variables == list(data.columns)
    assert transformer.scoring == "r2"
    assert transformer.threshold == 0.01
    
    # fit params
    # Number of selected features should always be less or equal to 
    # the number of input variables
    assert len(transformer.selected_features_) <= len(transformer.variables)
    # Number of keys in attribute should always be equal to number of of input variables
    assert len(transformer.performance_drifts_) == len(transformer.variables)
    
