"""Benchmarks for the mathematical variable transformers and scalers."""

from feature_engine.scaling import MeanNormalisationScaler
from feature_engine.transformation import (
    ArcSinhTransformer,
    ArcsinTransformer,
    BoxCoxTransformer,
    LogTransformer,
    PowerTransformer,
    ReciprocalTransformer,
    YeoJohnsonTransformer,
)

from .conftest import numerical_vars

NUM_VARS = numerical_vars()
POS_VARS = ["pos_0", "pos_1"]


def test_log_transformer_transform(benchmark, df_big):
    transformer = LogTransformer(variables=POS_VARS)
    transformer.fit(df_big)
    benchmark(transformer.transform, df_big)


def test_log_transformer_auto_c_fit(benchmark, df_big):
    # C="auto" makes fit learn the shift needed to make the variables positive.
    transformer = LogTransformer(variables=NUM_VARS, C="auto")
    benchmark(transformer.fit, df_big)


def test_log_transformer_auto_c_transform(benchmark, df_big):
    transformer = LogTransformer(variables=NUM_VARS, C="auto")
    transformer.fit(df_big)
    benchmark(transformer.transform, df_big)


def test_power_transformer_transform(benchmark, df_big):
    transformer = PowerTransformer(variables=POS_VARS, exp=0.5)
    transformer.fit(df_big)
    benchmark(transformer.transform, df_big)


def test_reciprocal_transformer_transform(benchmark, df_big):
    transformer = ReciprocalTransformer(variables=POS_VARS)
    transformer.fit(df_big)
    benchmark(transformer.transform, df_big)


def test_box_cox_transformer_fit(benchmark, df_big):
    transformer = BoxCoxTransformer(variables=POS_VARS)
    benchmark(transformer.fit, df_big)


def test_box_cox_transformer_transform(benchmark, df_big):
    transformer = BoxCoxTransformer(variables=POS_VARS)
    transformer.fit(df_big)
    benchmark(transformer.transform, df_big)


def test_yeo_johnson_transformer_fit(benchmark, df_big):
    transformer = YeoJohnsonTransformer(variables=NUM_VARS)
    benchmark(transformer.fit, df_big)


def test_yeo_johnson_transformer_transform(benchmark, df_big):
    transformer = YeoJohnsonTransformer(variables=NUM_VARS)
    transformer.fit(df_big)
    benchmark(transformer.transform, df_big)


def test_arcsin_transformer_transform(benchmark, df_big):
    transformer = ArcsinTransformer(variables=["frac_0"])
    transformer.fit(df_big)
    benchmark(transformer.transform, df_big)


def test_arcsinh_transformer_transform(benchmark, df_big):
    transformer = ArcSinhTransformer(variables=NUM_VARS)
    transformer.fit(df_big)
    benchmark(transformer.transform, df_big)


def test_mean_normalisation_scaler_fit(benchmark, df_big):
    scaler = MeanNormalisationScaler(variables=NUM_VARS)
    benchmark(scaler.fit, df_big)


def test_mean_normalisation_scaler_transform(benchmark, df_big):
    scaler = MeanNormalisationScaler(variables=NUM_VARS)
    scaler.fit(df_big)
    benchmark(scaler.transform, df_big)
