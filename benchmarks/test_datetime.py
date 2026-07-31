"""Benchmarks for the datetime feature extraction transformers."""

import pytest

from feature_engine.datetime import (
    DatetimeFeatures,
    DatetimeOrdinal,
    DatetimeSubtraction,
)

DATE_VARS = ["date_0", "date_1"]


@pytest.mark.parametrize(
    "features_to_extract",
    [
        ["year", "month", "day_of_month"],
        None,
        "all",
    ],
    ids=["basic", "default", "all"],
)
def test_datetime_features_transform(benchmark, df_big, features_to_extract):
    transformer = DatetimeFeatures(
        variables=DATE_VARS, features_to_extract=features_to_extract
    )
    transformer.fit(df_big)
    benchmark(transformer.transform, df_big)


def test_datetime_features_from_string_transform(benchmark, df_big):
    # Dates stored as strings: parsing dominates the runtime.
    df = df_big.copy()
    df["date_0"] = df["date_0"].astype(str)
    transformer = DatetimeFeatures(
        variables=["date_0"], features_to_extract=["year", "month", "day_of_month"]
    )
    transformer.fit(df)
    benchmark(transformer.transform, df)


def test_datetime_subtraction_transform(benchmark, df_big):
    transformer = DatetimeSubtraction(variables=["date_0"], reference=["date_1"])
    transformer.fit(df_big)
    benchmark(transformer.transform, df_big)


def test_datetime_ordinal_transform(benchmark, df_big):
    transformer = DatetimeOrdinal(variables=DATE_VARS)
    transformer.fit(df_big)
    benchmark(transformer.transform, df_big)
