import random

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
def df_test_with_groups():
    # Parameters
    n_samples = 100  # Total number of samples
    n_groups = 10    # Total number of groups
    n_features = 5   # Number of features

    # Generate random features
    np.random.seed(1)
    features = np.random.randn(n_samples, n_features)

    # Generate random target variable
    target = np.random.randint(0, 100, size=n_samples)

    # Generate groups
    groups = np.repeat(np.arange(1, n_groups + 1), n_samples // n_groups)
    np.random.shuffle(groups)

    # Create DataFrame
    df = pd.DataFrame(features, columns=[f'var_{i+1}' for i in range(n_features)])
    df['target'] = target
    df['group'] = groups

    features = [col for col in df.columns if col.startswith('var')]
    X = df[features]
    y = df['target']
    groups = df['group']

    return X, y, groups


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


@pytest.fixture(scope="module", name="random_uniform_method")
def wrap_random_uniform_method():
    def _random_uniform(x, y):
        return random.uniform(0.5, 1)

    return _random_uniform
