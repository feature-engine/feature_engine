"""Benchmarks for the outlier capping and trimming transformers."""

import pytest

from feature_engine.outliers import (
    ArbitraryOutlierCapper,
    OutlierTrimmer,
    Winsoriser,
)

from .conftest import numerical_vars

NUM_VARS = numerical_vars()


@pytest.mark.parametrize("capping_method", ["gaussian", "iqr", "quantiles", "mad"])
def test_winsoriser_fit(benchmark, df_big, capping_method):
    capper = Winsoriser(capping_method=capping_method, tail="both", variables=NUM_VARS)
    benchmark(capper.fit, df_big)


@pytest.mark.parametrize("add_indicators", [False, True])
def test_winsoriser_transform(benchmark, df_big, add_indicators):
    capper = Winsoriser(
        capping_method="iqr",
        tail="both",
        variables=NUM_VARS,
        add_indicators=add_indicators,
    )
    capper.fit(df_big)
    benchmark(capper.transform, df_big)


def test_arbitrary_outlier_capper_transform(benchmark, df_big):
    capper = ArbitraryOutlierCapper(
        max_capping_dict={var: 10 for var in NUM_VARS},
        min_capping_dict={var: -10 for var in NUM_VARS},
    )
    capper.fit(df_big)
    benchmark(capper.transform, df_big)


def test_outlier_trimmer_fit(benchmark, df_big):
    trimmer = OutlierTrimmer(capping_method="iqr", tail="both", variables=NUM_VARS)
    benchmark(trimmer.fit, df_big)


def test_outlier_trimmer_transform(benchmark, df_big):
    trimmer = OutlierTrimmer(capping_method="iqr", tail="both", variables=NUM_VARS)
    trimmer.fit(df_big)
    benchmark(trimmer.transform, df_big)
