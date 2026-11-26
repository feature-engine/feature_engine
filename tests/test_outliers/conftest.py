"""Data shared by the outlier transformer tests.

Each fixture returns a fresh dict, so tests can build the dataframe on the
backend under test with ``make_df(data)``. Missing values are written as None,
which both pandas and polars read as missing.
"""

import numpy as np
import pytest


@pytest.fixture
def data_normal_dist():
    # same seed and parameters as the pandas df_normal_dist fixture in
    # tests/conftest.py
    return {"var": np.random.RandomState(0).normal(0, 0.1, 100).tolist()}


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
    }
