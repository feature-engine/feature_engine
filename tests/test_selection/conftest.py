import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, make_classification


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
