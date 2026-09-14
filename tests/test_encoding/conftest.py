"""Data shared by the encoder tests.

Each fixture returns a fresh dict, so tests can build the dataframe on the
backend under test with ``make_df(data)``. Missing values are written as None,
which both pandas and polars read as missing.
"""

import pytest

TARGET = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]


@pytest.fixture
def data_enc():
    return {
        "var_A": ["A"] * 6 + ["B"] * 10 + ["C"] * 4,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": list(TARGET),
    }


@pytest.fixture
def data_enc_rare():
    return {
        "var_A": ["B"] * 9 + ["A"] * 6 + ["C"] * 4 + ["D"] * 1,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": list(TARGET),
    }


@pytest.fixture
def data_enc_na():
    return {
        "var_A": [None] + ["B"] * 8 + ["A"] * 6 + ["C"] * 4 + ["D"] * 1,
        "var_B": ["A"] * 10 + ["B"] * 6 + ["C"] * 4,
        "target": list(TARGET),
    }


@pytest.fixture
def data_enc_numeric():
    return {
        "var_A": [1] * 6 + [2] * 10 + [3] * 4,
        "var_B": [1] * 10 + [2] * 6 + [3] * 4,
        "target": list(TARGET),
    }


@pytest.fixture
def data_enc_big():
    return {
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
