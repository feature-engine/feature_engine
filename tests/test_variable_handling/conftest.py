from datetime import datetime, timezone

import pandas as pd
import polars as pl
import pytest


def cast_categorical(df, columns):
    """Cast `columns` to the backend's categorical dtype, whichever backend `df`
    (pandas or polars) happens to be. Used to build matched pandas/polars data
    for tests parametrized over both libraries.
    """
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        df[columns] = df[columns].astype("category")
        return df
    return df.with_columns([pl.col(c).cast(pl.Categorical) for c in columns])


# Data shared between the pandas and polars variants of a test. Kept as plain
# dicts/lists (not fixtures) so a test can build both `make_df(BASIC_DATA)` and
# `make_df(BASIC_DATA)` for a different backend without needing to convert
# between frame types.
BASIC_DATA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
}

# Datetime formats that both pandas and polars auto-detect: native
# Datetime/Date columns and ISO-8601 strings. Formats that only pandas'
# flexible, dateutil-backed guessing can parse (e.g. "01-Jan-2010",
# "10/11/12", bare time strings) are exercised separately, in pandas-only
# tests, against the `df_datetime` fixture below.
DATETIME_DATA = {
    **BASIC_DATA,
    "date_range": [datetime(2020, 2, 24, 0, i) for i in range(4)],
    "date_obj0": ["2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27"],
    "date_range_tz": [
        datetime(2020, 2, 24, 0, i, tzinfo=timezone.utc) for i in range(4)
    ],
}


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "date_range": pd.date_range("2020-02-24", periods=4, freq="min"),
            "date_obj0": ["2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27"],
            "date_range_tz": pd.date_range(
                "2020-02-24", periods=4, freq="min"
            ).tz_localize("UTC"),
        }
    )
    df["Name"] = df["Name"].astype("category")
    return df


@pytest.fixture
def df_int(df):
    df = df.copy()
    df.columns = range(1, len(df.columns) + 1)
    return df


@pytest.fixture
def df_datetime(df):
    df = df.copy()

    df["date_obj1"] = ["01-Jan-2010", "24-Feb-1945", "14-Jun-2100", "17-May-1999"]
    df["date_obj2"] = ["10/11/12", "12/31/09", "06/30/95", "03/17/04"]
    df["time_obj"] = ["21:45:23", "09:15:33", "12:34:59", "03:27:02"]

    df["time_objTZ"] = df["time_obj"].add(["+5", "+11", "-3", "-8"])
    df["date_obj1"] = df["date_obj1"].astype("category")
    df["Age"] = df["Age"].astype("O")
    return df
