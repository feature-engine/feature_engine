"""Benchmarks for the variable handling helpers.

These functions are called by every transformer during fit, so they are on the
hot path of the whole library.
"""

from feature_engine.variable_handling import (
    check_numerical_variables,
    find_all_variables,
    find_categorical_and_numerical_variables,
    find_categorical_variables,
    find_datetime_variables,
    find_numerical_variables,
    retain_variables_if_in_df,
)

from .conftest import numerical_vars

NUM_VARS = numerical_vars()


def test_find_numerical_variables(benchmark, df_big):
    benchmark(find_numerical_variables, df_big)


def test_find_categorical_variables(benchmark, df_big):
    benchmark(find_categorical_variables, df_big)


def test_find_datetime_variables(benchmark, df_big):
    benchmark(find_datetime_variables, df_big)


def test_find_all_variables(benchmark, df_big):
    benchmark(find_all_variables, df_big)


def test_find_categorical_and_numerical_variables(benchmark, df_big):
    benchmark(find_categorical_and_numerical_variables, df_big)


def test_check_numerical_variables(benchmark, df_big):
    benchmark(check_numerical_variables, df_big, NUM_VARS)


def test_retain_variables_if_in_df(benchmark, df_big):
    benchmark(retain_variables_if_in_df, df_big, NUM_VARS + ["not_in_df"])


def test_find_datetime_variables_object_dtype(benchmark, df_big):
    # Datetimes cast as strings: the check needs to try parsing the columns.
    df = df_big.copy()
    df["date_0"] = df["date_0"].astype(str)
    df["date_1"] = df["date_1"].astype(str)
    benchmark(find_datetime_variables, df)
