"""End to end benchmarks: several transformers chained in a Pipeline.

These are the closest thing to a real user workflow and catch regressions that
only show up when transformers are combined.
"""

import pytest

from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder, WoEEncoder
from feature_engine.imputation import CategoricalImputer, MeanImputer
from feature_engine.outliers import Winsoriser
from feature_engine.pipeline import Pipeline
from feature_engine.preprocessing import MatchCategories, MatchVariables
from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures
from feature_engine.transformation import YeoJohnsonTransformer

from .conftest import categorical_vars, numerical_vars

NUM_VARS = numerical_vars()
CAT_VARS = categorical_vars()


def _build_pipeline():
    return Pipeline(
        [
            ("cat_imputer", CategoricalImputer(variables=CAT_VARS)),
            ("num_imputer", MeanImputer()),
            ("rare_label", RareLabelEncoder(tol=0.05, n_categories=2)),
            (
                "winsorizer",
                Winsoriser(capping_method="iqr", tail="both", variables=NUM_VARS),
            ),
            ("yeo_johnson", YeoJohnsonTransformer(variables=NUM_VARS)),
            ("one_hot", OneHotEncoder(variables=CAT_VARS, drop_last=True)),
            ("drop_constant", DropConstantFeatures(tol=0.998)),
        ]
    )


def test_pipeline_fit(benchmark, df_big_na, y_binary_big):
    pipe = _build_pipeline()
    benchmark(pipe.fit, df_big_na, y_binary_big)


def test_pipeline_transform(benchmark, df_big_na, y_binary_big):
    pipe = _build_pipeline()
    pipe.fit(df_big_na, y_binary_big)
    benchmark(pipe.transform, df_big_na)


def test_credit_scoring_pipeline_fit(benchmark, df_big, y_binary_big):
    pipe = Pipeline(
        [
            (
                "discretiser",
                EqualFrequencyDiscretiser(q=10, variables=NUM_VARS, return_object=True),
            ),
            ("rare_label", RareLabelEncoder(tol=0.02, n_categories=2)),
            ("woe", WoEEncoder(variables=NUM_VARS + CAT_VARS)),
            ("drop_correlated", DropCorrelatedFeatures(threshold=0.9)),
        ]
    )
    benchmark(pipe.fit, df_big, y_binary_big)


@pytest.mark.parametrize("match_dtypes", [False, True])
def test_match_variables_transform(benchmark, df_big, match_dtypes):
    matcher = MatchVariables(match_dtypes=match_dtypes, verbose=False)
    matcher.fit(df_big)
    benchmark(matcher.transform, df_big)


def test_match_categories_transform(benchmark, df_big):
    matcher = MatchCategories(variables=CAT_VARS)
    matcher.fit(df_big)
    benchmark(matcher.transform, df_big)
