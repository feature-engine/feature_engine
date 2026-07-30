"""Benchmarks for the time series forecasting feature transformers."""

import pytest

from feature_engine.timeseries.forecasting import (
    ExpandingWindowFeatures,
    LagFeatures,
    WindowFeatures,
)

TS_VARS = [f"num_{i}" for i in range(4)]


@pytest.mark.parametrize("periods", [1, [1, 3, 6, 12]], ids=["single", "multiple"])
def test_lag_features_transform(benchmark, df_timeseries, periods):
    transformer = LagFeatures(variables=TS_VARS, periods=periods)
    transformer.fit(df_timeseries)
    benchmark(transformer.transform, df_timeseries)


def test_lag_features_freq_transform(benchmark, df_timeseries):
    transformer = LagFeatures(variables=TS_VARS, freq=["1h", "1D"])
    transformer.fit(df_timeseries)
    benchmark(transformer.transform, df_timeseries)


@pytest.mark.parametrize("window", [3, [3, 12]], ids=["single", "multiple"])
def test_window_features_transform(benchmark, df_timeseries, window):
    transformer = WindowFeatures(
        variables=TS_VARS, window=window, functions=["mean", "std"]
    )
    transformer.fit(df_timeseries)
    benchmark(transformer.transform, df_timeseries)


def test_expanding_window_features_transform(benchmark, df_timeseries):
    transformer = ExpandingWindowFeatures(variables=TS_VARS, functions=["mean", "max"])
    transformer.fit(df_timeseries)
    benchmark(transformer.transform, df_timeseries)
