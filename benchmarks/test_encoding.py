"""Benchmarks for the categorical encoding transformers."""

import pytest

from feature_engine.encoding import (
    CountEncoder,
    DecisionTreeEncoder,
    MeanEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareLabelEncoder,
    StringSimilarityEncoder,
    WoEEncoder,
)

from .conftest import categorical_vars

CAT_VARS = categorical_vars()


@pytest.mark.parametrize("encoding_method", ["count", "frequency"])
def test_count_encoder_fit(benchmark, df_big, encoding_method):
    encoder = CountEncoder(encoding_method=encoding_method, variables=CAT_VARS)
    benchmark(encoder.fit, df_big)


@pytest.mark.parametrize("encoding_method", ["count", "frequency"])
def test_count_encoder_transform(benchmark, df_big, encoding_method):
    encoder = CountEncoder(encoding_method=encoding_method, variables=CAT_VARS)
    encoder.fit(df_big)
    benchmark(encoder.transform, df_big)


@pytest.mark.parametrize("encoding_method", ["ordered", "arbitrary"])
def test_ordinal_encoder_fit(benchmark, df_big, y_binary_big, encoding_method):
    encoder = OrdinalEncoder(encoding_method=encoding_method, variables=CAT_VARS)
    benchmark(encoder.fit, df_big, y_binary_big)


def test_ordinal_encoder_transform(benchmark, df_big, y_binary_big):
    encoder = OrdinalEncoder(encoding_method="ordered", variables=CAT_VARS)
    encoder.fit(df_big, y_binary_big)
    benchmark(encoder.transform, df_big)


def test_mean_encoder_fit(benchmark, df_big, y_binary_big):
    encoder = MeanEncoder(variables=CAT_VARS)
    benchmark(encoder.fit, df_big, y_binary_big)


def test_mean_encoder_transform(benchmark, df_big, y_binary_big):
    encoder = MeanEncoder(variables=CAT_VARS)
    encoder.fit(df_big, y_binary_big)
    benchmark(encoder.transform, df_big)


def test_mean_encoder_smoothing_fit(benchmark, df_big, y_binary_big):
    encoder = MeanEncoder(variables=CAT_VARS, smoothing="auto")
    benchmark(encoder.fit, df_big, y_binary_big)


def test_woe_encoder_fit(benchmark, df_big, y_binary_big):
    encoder = WoEEncoder(variables=CAT_VARS)
    benchmark(encoder.fit, df_big, y_binary_big)


def test_woe_encoder_transform(benchmark, df_big, y_binary_big):
    encoder = WoEEncoder(variables=CAT_VARS)
    encoder.fit(df_big, y_binary_big)
    benchmark(encoder.transform, df_big)


@pytest.mark.parametrize("drop_last", [False, True])
def test_one_hot_encoder_transform(benchmark, df_big, drop_last):
    encoder = OneHotEncoder(variables=CAT_VARS, drop_last=drop_last)
    encoder.fit(df_big)
    benchmark(encoder.transform, df_big)


def test_one_hot_encoder_top_categories_transform(benchmark, df_big):
    encoder = OneHotEncoder(variables=CAT_VARS, top_categories=5)
    encoder.fit(df_big)
    benchmark(encoder.transform, df_big)


def test_rare_label_encoder_fit(benchmark, df_big):
    encoder = RareLabelEncoder(tol=0.05, n_categories=2, variables=CAT_VARS)
    benchmark(encoder.fit, df_big)


def test_rare_label_encoder_transform(benchmark, df_big):
    encoder = RareLabelEncoder(tol=0.05, n_categories=2, variables=CAT_VARS)
    encoder.fit(df_big)
    benchmark(encoder.transform, df_big)


def test_decision_tree_encoder_fit(benchmark, df_small, y_binary):
    encoder = DecisionTreeEncoder(
        variables=CAT_VARS, regression=False, cv=2, random_state=0
    )
    benchmark(encoder.fit, df_small, y_binary)


def test_decision_tree_encoder_transform(benchmark, df_small, y_binary):
    encoder = DecisionTreeEncoder(
        variables=CAT_VARS, regression=False, cv=2, random_state=0
    )
    encoder.fit(df_small, y_binary)
    benchmark(encoder.transform, df_small)


def test_string_similarity_encoder_fit(benchmark, df_small):
    encoder = StringSimilarityEncoder(variables=CAT_VARS)
    benchmark(encoder.fit, df_small)


def test_string_similarity_encoder_transform(benchmark, df_small):
    encoder = StringSimilarityEncoder(variables=CAT_VARS)
    encoder.fit(df_small)
    benchmark(encoder.transform, df_small)
