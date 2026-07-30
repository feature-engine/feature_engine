"""Benchmarks for the discretisation transformers."""

import pytest

from feature_engine.discretisation import (
    ArbitraryDiscretiser,
    DecisionTreeDiscretiser,
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    GeometricWidthDiscretiser,
)

from .conftest import numerical_vars

NUM_VARS = numerical_vars()
POS_VARS = ["pos_0", "pos_1"]


def test_equal_frequency_discretiser_fit(benchmark, df_big):
    disc = EqualFrequencyDiscretiser(q=10, variables=NUM_VARS)
    benchmark(disc.fit, df_big)


@pytest.mark.parametrize("return_boundaries", [False, True])
def test_equal_frequency_discretiser_transform(benchmark, df_big, return_boundaries):
    disc = EqualFrequencyDiscretiser(
        q=10, variables=NUM_VARS, return_boundaries=return_boundaries
    )
    disc.fit(df_big)
    benchmark(disc.transform, df_big)


def test_equal_width_discretiser_fit(benchmark, df_big):
    disc = EqualWidthDiscretiser(bins=10, variables=NUM_VARS)
    benchmark(disc.fit, df_big)


def test_equal_width_discretiser_transform(benchmark, df_big):
    disc = EqualWidthDiscretiser(bins=10, variables=NUM_VARS)
    disc.fit(df_big)
    benchmark(disc.transform, df_big)


def test_geometric_width_discretiser_transform(benchmark, df_big):
    disc = GeometricWidthDiscretiser(bins=10, variables=POS_VARS)
    disc.fit(df_big)
    benchmark(disc.transform, df_big)


def test_arbitrary_discretiser_transform(benchmark, df_big):
    limits = {var: [-1000, -1, 0, 1, 1000] for var in NUM_VARS}
    disc = ArbitraryDiscretiser(binning_dict=limits)
    disc.fit(df_big)
    benchmark(disc.transform, df_big)


def test_decision_tree_discretiser_fit(benchmark, df_small, y_continuous):
    disc = DecisionTreeDiscretiser(
        variables=NUM_VARS, regression=True, cv=2, random_state=0
    )
    benchmark(disc.fit, df_small, y_continuous)


def test_decision_tree_discretiser_transform(benchmark, df_small, y_continuous):
    disc = DecisionTreeDiscretiser(
        variables=NUM_VARS, regression=True, cv=2, random_state=0
    )
    disc.fit(df_small, y_continuous)
    benchmark(disc.transform, df_small)
