"""Data shared by the imputer tests.

Each fixture returns a fresh dict, so tests can build the dataframe on the
backend under test with ``make_df(data)``. Missing values are written as None,
not np.nan: polars treats np.nan as a real float value (not a null), so
mean/std/quantile would not skip it, unlike pandas. None becomes a null on
both backends.
"""

import datetime

import pytest


def _data_na():
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


@pytest.fixture
def data_na():
    return _data_na()


@pytest.fixture
def data_na_dob():
    # dob is never null: exercises a datetime variable that missing_only=True
    # should exclude from variables_.
    data = _data_na()
    data["dob"] = [datetime.datetime(2020, 2, 24, 0, i) for i in range(8)]
    return data
