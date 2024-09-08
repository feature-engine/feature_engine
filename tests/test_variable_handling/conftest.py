import pandas as pd
import pytest


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
