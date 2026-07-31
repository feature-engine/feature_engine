"""Shared data fixtures for the benchmark suite.

The dataframes built here are synthetic but representative of the kind of data
Feature-engine transformers are used on: a mix of numerical, categorical and
datetime variables, with missing values.

Data generation happens in fixtures so that it is never included in the
measured section of a benchmark.
"""

import numpy as np
import pandas as pd
import pytest

# Number of rows used for the transformers whose fit/transform is cheap.
BIG_N = 10_000

# Number of rows used for the transformers that train models under the hood
# (decision trees, cross-validation, ...) so benchmarks stay in the millisecond
# to low second range.
SMALL_N = 1_000

# Number of rows used for the row-wise transformers, which are an order of
# magnitude slower per row than the vectorised ones.
TINY_N = 500

N_NUMERICAL = 8
N_CATEGORICAL = 4


def _make_dataframe(n_rows: int, seed: int = 0, with_na: bool = False):
    rng = np.random.default_rng(seed)

    data = {
        f"num_{i}": rng.normal(loc=i, scale=i + 1, size=n_rows)
        for i in range(N_NUMERICAL)
    }

    # A couple of strictly positive variables, needed by log/box-cox style
    # transformers.
    data["pos_0"] = rng.gamma(shape=2.0, scale=3.0, size=n_rows) + 0.1
    data["pos_1"] = rng.gamma(shape=5.0, scale=1.0, size=n_rows) + 0.1

    # A variable bounded between 0 and 1, needed by the arcsin transformer.
    data["frac_0"] = rng.uniform(0.0, 1.0, size=n_rows)

    # Categorical variables with a decreasing cardinality, including rare
    # categories to exercise the rare label encoder.
    for i in range(N_CATEGORICAL):
        n_categories = 5 * (i + 1)
        weights = np.linspace(1.0, 0.02, num=n_categories)
        weights = weights / weights.sum()
        data[f"cat_{i}"] = rng.choice(
            [f"cat_{i}_value_{j}" for j in range(n_categories)],
            size=n_rows,
            p=weights,
        )

    data["date_0"] = pd.date_range("2015-01-01", periods=n_rows, freq="h")
    data["date_1"] = pd.date_range("2018-06-15", periods=n_rows, freq="7min")

    df = pd.DataFrame(data)

    if with_na:
        for column in ["num_0", "num_1", "pos_0", "cat_0", "cat_1"]:
            mask = rng.random(n_rows) < 0.15
            df.loc[mask, column] = np.nan

    return df


def numerical_vars():
    return [f"num_{i}" for i in range(N_NUMERICAL)]


def categorical_vars():
    return [f"cat_{i}" for i in range(N_CATEGORICAL)]


@pytest.fixture(scope="session")
def df_big():
    """Complete dataframe, no missing data."""
    return _make_dataframe(BIG_N, seed=0)


@pytest.fixture(scope="session")
def df_big_na():
    """Complete dataframe with missing data in numerical and categorical vars."""
    return _make_dataframe(BIG_N, seed=1, with_na=True)


@pytest.fixture(scope="session")
def df_small():
    """Smaller dataframe, for the estimator based transformers."""
    return _make_dataframe(SMALL_N, seed=2)


@pytest.fixture(scope="session")
def df_tiny():
    """Smallest dataframe, for the row-wise transformers."""
    return _make_dataframe(TINY_N, seed=7)


@pytest.fixture(scope="session")
def y_binary():
    """Binary target aligned with ``df_small``."""
    rng = np.random.default_rng(3)
    return pd.Series(rng.integers(0, 2, size=SMALL_N), name="target")


@pytest.fixture(scope="session")
def y_binary_big():
    """Binary target aligned with ``df_big``."""
    rng = np.random.default_rng(4)
    return pd.Series(rng.integers(0, 2, size=BIG_N), name="target")


@pytest.fixture(scope="session")
def y_continuous():
    """Continuous target aligned with ``df_small``."""
    rng = np.random.default_rng(5)
    return pd.Series(rng.normal(size=SMALL_N), name="target")


@pytest.fixture(scope="session")
def df_timeseries():
    """Time indexed dataframe with numerical variables only."""
    rng = np.random.default_rng(6)
    index = pd.date_range("2020-01-01", periods=BIG_N, freq="15min")
    return pd.DataFrame(
        {f"num_{i}": rng.normal(size=BIG_N).cumsum() for i in range(4)},
        index=index,
    )
