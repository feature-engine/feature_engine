import numpy as np
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


@pytest.fixture(scope="module")
def df_test_num_cat():
    np.random.seed(1)

    df = {
        "var_A": ["A"] * 60 + ["B"] * 100 + ["C"] * 40,
        "var_B": ["A"] * 100 + ["B"] * 60 + ["C"] * 40,
        "var_C": np.random.normal(size=200),
        "var_D": np.random.normal(size=200),
        "target": np.concatenate(
            (
                np.random.binomial(1, 0.7, 60),
                np.random.binomial(1, 0.2, 100),
                np.random.binomial(1, 0.8, 40),
            ),
            axis=None,
        ),
    }

    df = pd.DataFrame(df)
    X = df[["var_A", "var_B", "var_C", "var_D"]]
    y = df["target"]
    return X, y
