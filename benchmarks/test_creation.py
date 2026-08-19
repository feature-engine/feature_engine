"""Benchmarks for the feature creation transformers."""

import pytest

from feature_engine.creation import (
    CyclicalFeatures,
    DecisionTreeFeatures,
    MathFeatures,
    RelativeFeatures,
)

from .conftest import numerical_vars

NUM_VARS = numerical_vars()


@pytest.mark.parametrize(
    "func", [["sum", "mean"], ["sum", "mean", "std", "min", "max"]]
)
def test_math_features_transform(benchmark, df_tiny, func):
    # Keep the original dataset size so results remain comparable with the
    # pre-NumPy baseline recorded for issue #576.
    creator = MathFeatures(variables=NUM_VARS, func=func)
    creator.fit(df_tiny)
    benchmark(creator.transform, df_tiny)


def test_relative_features_transform(benchmark, df_big):
    creator = RelativeFeatures(
        variables=NUM_VARS[:4],
        reference=["num_4"],
        func=["sub", "div"],
    )
    creator.fit(df_big)
    benchmark(creator.transform, df_big)


def test_cyclical_features_transform(benchmark, df_big):
    creator = CyclicalFeatures(variables=NUM_VARS)
    creator.fit(df_big)
    benchmark(creator.transform, df_big)


def test_decision_tree_features_fit(benchmark, df_small, y_continuous):
    creator = DecisionTreeFeatures(
        variables=NUM_VARS[:3],
        features_to_combine=2,
        regression=True,
        cv=2,
        random_state=0,
    )
    benchmark(creator.fit, df_small, y_continuous)


def test_decision_tree_features_transform(benchmark, df_small, y_continuous):
    creator = DecisionTreeFeatures(
        variables=NUM_VARS[:3],
        features_to_combine=2,
        regression=True,
        cv=2,
        random_state=0,
    )
    creator.fit(df_small, y_continuous)
    benchmark(creator.transform, df_small)
