import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope="module")
def dataframe_vartypes():
    data = {
        "Name": ["tom", "nick", "krish", "jack"],
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
        "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="module")
def dataframe_na():
    data = {
        "Name": ["tom", "nick", "krish", np.nan, "peter", np.nan, "fred", "sam"],
        "City": [
            "London",
            "Manchester",
            np.nan,
            np.nan,
            "London",
            "London",
            "Bristol",
            "Manchester",
        ],
        "Studies": [
            "Bachelor",
            "Bachelor",
            np.nan,
            np.nan,
            "Bachelor",
            "PhD",
            "None",
            "Masters",
        ],
        "Age": [20, 21, 19, np.nan, 23, 40, 41, 37],
        "Marks": [0.9, 0.8, 0.7, np.nan, 0.3, np.nan, 0.8, 0.6],
        "dob": pd.date_range("2020-02-24", periods=8, freq="T"),
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="module")
def dataframe_enc():
    df = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)
    return df


@pytest.fixture(scope="module")
def dataframe_enc_rare():
    df = {
        "var_A": ["B"] * 9 + ["A"] * 6 + ["C"] * 4 + ["D"] * 1,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)
    return df


@pytest.fixture(scope="module")
def dataframe_enc_na():
    df = {
        "var_A": ["B"] * 9 + ["A"] * 6 + ["C"] * 4 + ["D"] * 1,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)
    df.loc[0, "var_A"] = np.nan
    return df


@pytest.fixture(scope="module")
def dataframe_enc_big():
    df = {
        "var_A": ["A"] * 6
        + ["B"] * 10
        + ["C"] * 4
        + ["D"] * 10
        + ["E"] * 2
        + ["F"] * 2
        + ["G"] * 6,
        "var_B": ["A"] * 10
        + ["B"] * 6
        + ["C"] * 4
        + ["D"] * 10
        + ["E"] * 2
        + ["F"] * 2
        + ["G"] * 6,
        "var_C": ["A"] * 4
        + ["B"] * 6
        + ["C"] * 10
        + ["D"] * 10
        + ["E"] * 2
        + ["F"] * 2
        + ["G"] * 6,
    }
    df = pd.DataFrame(df)
    return df


@pytest.fixture(scope="module")
def dataframe_enc_big_na():
    df = {
        "var_A": ["A"] * 6
        + ["B"] * 10
        + ["C"] * 4
        + ["D"] * 10
        + ["E"] * 2
        + ["F"] * 2
        + ["G"] * 6,
        "var_B": ["A"] * 10
        + ["B"] * 6
        + ["C"] * 4
        + ["D"] * 10
        + ["E"] * 2
        + ["F"] * 2
        + ["G"] * 6,
        "var_C": ["A"] * 4
        + ["B"] * 6
        + ["C"] * 10
        + ["D"] * 10
        + ["E"] * 2
        + ["F"] * 2
        + ["G"] * 6,
    }
    df = pd.DataFrame(df)
    df.loc[0, "var_A"] = np.nan
    return df


@pytest.fixture(scope="module")
def dataframe_normal_dist():
    np.random.seed(0)
    mu, sigma = 0, 0.1  # mean and standard deviation
    s = np.random.normal(mu, sigma, 100)
    df = pd.DataFrame(s)
    df.columns = ["var"]
    return df


@pytest.fixture(scope="module")
def dataframe_constant_features():
    data = {
        "Name": ["tom", "nick", "krish", "jack"],
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
        "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
        "const_feat_num": [1, 1, 1, 1],
        "const_feat_cat": ["a", "a", "a", "a"],
        "quasi_feat_num": [1, 1, 1, 2],
        "quasi_feat_cat": ["a", "a", "a", "b"],
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="module")
def dataframe_duplicate_features():
    data = {
        "Name": ["tom", "nick", "krish", "jack"],
        "dob2": pd.date_range("2020-02-24", periods=4, freq="T"),
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
        "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
        "City2": ["London", "Manchester", "Liverpool", "Bristol"],
        "dob3": pd.date_range("2020-02-24", periods=4, freq="T"),
        "Age2": [20, 21, 19, 18],
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="module")
def dataframe_duplicate_features_with_na():
    data = {
        "Name": ["tom", "nick", "krish", "jack", np.nan],
        "dob2": pd.date_range("2020-02-24", periods=5, freq="T"),
        "City": ["London", "Manchester", "Liverpool", "Bristol", np.nan],
        "Age": [20, 21, np.nan, 18, 34],
        "Marks": [0.9, 0.8, 0.7, 0.6, 0.5],
        "dob": pd.date_range("2020-02-24", periods=5, freq="T"),
        "City2": ["London", "Manchester", "Liverpool", "Bristol", np.nan],
        "dob3": pd.date_range("2020-02-24", periods=5, freq="T"),
        "Age2": [20, 21, np.nan, 18, 34],
    }

    df = pd.DataFrame(data)
    return df
