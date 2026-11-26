"""Data shared by the discretiser tests.

Each fixture returns a fresh dict, so tests can build the dataframe on the
backend under test with ``make_df(data)``. Missing values are written as None,
which both pandas and polars read as missing.
"""

import datetime
from functools import lru_cache

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing


@lru_cache(maxsize=1)
def _california_housing():
    dataset = fetch_california_housing()
    return dataset.feature_names, dataset.data


@pytest.fixture
def data_california():
    feature_names, values = _california_housing()
    return {name: values[:, i].tolist() for i, name in enumerate(feature_names)}


@pytest.fixture
def data_normal_dist():
    # same seed and parameters as the pandas df_normal_dist fixture in
    # tests/conftest.py
    return {"var": np.random.RandomState(0).normal(0, 0.1, 100).tolist()}


@pytest.fixture
def data_vartypes():
    return {
        "Name": ["tom", "nick", "krish", "jack"],
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
        "dob": [datetime.datetime(2020, 2, 24, 0, i) for i in range(4)],
    }


@pytest.fixture
def data_na():
    return {
        "Name": ["tom", "nick", "krish", None, "peter", None, "fred", "sam"],
        "City": [
            "London",
            "Manchester",
            None,
            None,
            "London",
            "London",
            "Bristol",
            "Manchester",
        ],
        "Studies": [
            "Bachelor",
            "Bachelor",
            None,
            None,
            "Bachelor",
            "PhD",
            "None",
            "Masters",
        ],
        "Age": [20, 21, 19, None, 23, 40, 41, 37],
        "Marks": [0.9, 0.8, 0.7, None, 0.3, None, 0.8, 0.6],
        "dob": [datetime.datetime(2020, 2, 24, 0, i) for i in range(8)],
    }
