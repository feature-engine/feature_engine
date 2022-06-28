import pandas as pd
import pytest


@pytest.fixture(scope="module")
def df_classification():
    df = {
        "cat_var_A": ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5,
        "cat_var_B": ["A"] * 6
        + ["B"] * 2
        + ["C"] * 2
        + ["B"] * 2
        + ["C"] * 2
        + ["D"] * 6,
        "num_var_A": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        "num_var_B": [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4],
    }

    df = pd.DataFrame(df)
    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    return df, y


@pytest.fixture(scope="module")
def df_regression():
    df = {
        "cat_var_A": ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5,
        "cat_var_B": ["A"] * 6
        + ["B"] * 2
        + ["C"] * 2
        + ["B"] * 2
        + ["C"] * 2
        + ["D"] * 6,
        "num_var_A": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        "num_var_B": [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4],
    }

    df = pd.DataFrame(df)
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    return df, y
