"""Benchmarks for the missing data imputation transformers."""

import pytest

from feature_engine.imputation import (
    ArbitraryImputer,
    CategoricalImputer,
    DropMissingData,
    EndTailImputer,
    MeanImputer,
    MissingIndicator,
    RandomSampleImputer,
)

from .conftest import categorical_vars, numerical_vars

NUM_VARS = numerical_vars()
CAT_VARS = categorical_vars()


@pytest.mark.parametrize("method", ["mean", "median"])
def test_mean_imputer_fit(benchmark, df_big_na, method):
    imputer = MeanImputer(imputation_method=method, variables=NUM_VARS)
    benchmark(imputer.fit, df_big_na)


@pytest.mark.parametrize("method", ["mean", "median"])
def test_mean_imputer_transform(benchmark, df_big_na, method):
    imputer = MeanImputer(imputation_method=method, variables=NUM_VARS)
    imputer.fit(df_big_na)
    benchmark(imputer.transform, df_big_na)


def test_arbitrary_imputer_transform(benchmark, df_big_na):
    imputer = ArbitraryImputer(arbitrary_number=-999, variables=NUM_VARS)
    imputer.fit(df_big_na)
    benchmark(imputer.transform, df_big_na)


@pytest.mark.parametrize("method", ["gaussian", "iqr"])
def test_end_tail_imputer_fit(benchmark, df_big_na, method):
    imputer = EndTailImputer(imputation_method=method, variables=NUM_VARS)
    benchmark(imputer.fit, df_big_na)


def test_end_tail_imputer_transform(benchmark, df_big_na):
    imputer = EndTailImputer(imputation_method="gaussian", variables=NUM_VARS)
    imputer.fit(df_big_na)
    benchmark(imputer.transform, df_big_na)


@pytest.mark.parametrize("method", ["frequent", "missing"])
def test_categorical_imputer_fit(benchmark, df_big_na, method):
    imputer = CategoricalImputer(imputation_method=method, variables=CAT_VARS)
    benchmark(imputer.fit, df_big_na)


@pytest.mark.parametrize("method", ["frequent", "missing"])
def test_categorical_imputer_transform(benchmark, df_big_na, method):
    imputer = CategoricalImputer(imputation_method=method, variables=CAT_VARS)
    imputer.fit(df_big_na)
    benchmark(imputer.transform, df_big_na)


def test_random_sample_imputer_transform(benchmark, df_big_na):
    imputer = RandomSampleImputer(variables=NUM_VARS + CAT_VARS, random_state=0)
    imputer.fit(df_big_na)
    benchmark(imputer.transform, df_big_na)


def test_missing_indicator_transform(benchmark, df_big_na):
    imputer = MissingIndicator(missing_only=True)
    imputer.fit(df_big_na)
    benchmark(imputer.transform, df_big_na)


def test_drop_missing_data_transform(benchmark, df_big_na):
    imputer = DropMissingData()
    imputer.fit(df_big_na)
    benchmark(imputer.transform, df_big_na)
