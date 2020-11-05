import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import RecursiveFeatureElimination


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
    sel = RecursiveFeatureElimination(
        RandomForestClassifier(random_state=1), threshold=0.001)
    sel.fit(X, y)

    # expected result
    Xtransformed = pd.DataFrame(X[['var_0', 'var_6']].copy())

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
    print(sel.initial_model_performance_)
    print(sel.selected_features_)
    # test fit attrs
    assert np.round(sel.initial_model_performance_, 3) == 0.997
    assert sel.selected_features_ == ['var_0', 'var_6']
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)
