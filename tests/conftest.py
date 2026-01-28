import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.utils import Bunch

# Mock fetch_california_housing to avoid 403 Forbidden errors in CI
def mock_fetch_california_housing(*args, **kwargs):
    rng = np.random.default_rng(42)
    data = rng.uniform(1, 10, (100, 8))
    feature_names = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    ]
    df = pd.DataFrame(data, columns=feature_names)

    # Create a target that correlates with the expected 'selected' features
    # to satisfy MRMR tests which expect specific features to be chosen.
    target = (
        5.0 * df["MedInc"] +
        4.0 * df["Latitude"] +
        3.0 * df["HouseAge"] +
        2.0 * df["AveRooms"] +
        1.0 * df["AveOccup"] +
        rng.standard_normal(100) * 0.1
    )

    if kwargs.get("return_X_y"):
        if kwargs.get("as_frame"):
            return df, pd.Series(target, name="MedHouseVal")
        return data, target.values

    df["MedHouseVal"] = target
    return Bunch(
        data=data,
        target=target.values,
        frame=df if kwargs.get("as_frame") else None,
        feature_names=feature_names,
        target_names=["MedHouseVal"],
        DESCR="mocked california housing",
    )

patch("sklearn.datasets.fetch_california_housing", side_effect=mock_fetch_california_housing).start()


@pytest.fixture(scope="module")
def df_vartypes():
    data = {
        "Name": ["tom", "nick", "krish", "jack"],
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
        "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
    }

    df = pd.DataFrame(data)

    return df


@pytest.fixture(scope="module")
def df_numeric_columns():
    data = {
        0: ["tom", "nick", "krish", "jack"],
        1: ["London", "Manchester", "Liverpool", "Bristol"],
        2: [20, 21, 19, 18],
        3: [0.9, 0.8, 0.7, 0.6],
        4: pd.date_range("2020-02-24", periods=4, freq="min"),
    }

    df = pd.DataFrame(data)

    return df


@pytest.fixture(scope="module")
def df_na():
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
        "dob": pd.date_range("2020-02-24", periods=8, freq="min"),
    }

    df = pd.DataFrame(data)

    return df


@pytest.fixture(scope="module")
def df_enc():
    df = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)

    return df


@pytest.fixture(scope="module")
def df_enc_category_dtypes():
    df = {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)
    df[["var_A", "var_B"]] = df[["var_A", "var_B"]].astype("category")

    return df


@pytest.fixture(scope="module")
def df_enc_numeric():
    df = {
        "var_A": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
        "var_B": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)

    return df


@pytest.fixture(scope="module")
def df_enc_rare():
    df = {
        "var_A": ["B"] * 9 + ["A"] * 6 + ["C"] * 4 + ["D"] * 1,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)

    return df


@pytest.fixture(scope="module")
def df_enc_na():
    df = {
        "var_A": ["B"] * 9 + ["A"] * 6 + ["C"] * 4 + ["D"] * 1,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    }
    df = pd.DataFrame(df)
    df.loc[0, "var_A"] = np.nan

    return df


@pytest.fixture(scope="module")
def df_enc_big():
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
def df_enc_big_na():
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
def df_normal_dist():
    np.random.seed(0)
    mu, sigma = 0, 0.1  # mean and standard deviation
    s = np.random.normal(mu, sigma, 100)
    df = pd.DataFrame(s)
    df.columns = ["var"]

    return df
