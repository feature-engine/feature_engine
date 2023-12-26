import pandas as pd
import pytest

from feature_engine.variable_handling import (
    find_categorical_variables,
    find_datetime_variables,
    find_numerical_variables,
)


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "date_range": pd.date_range("2020-02-24", periods=4, freq="T"),
            "date_obj0": ["2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27"],
        }
    )
    df["Name"] = df["Name"].astype("category")
    return df


@pytest.fixture
def df_int(df):
    df = df.copy()
    df.columns = [1, 2, 3, 4, 5, 6]
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


def test_numerical_variables_finds_numerical_variables(df, df_int):
    assert find_numerical_variables(df) == ["Age", "Marks"]
    assert find_numerical_variables(df_int) == [3, 4]


def test_numerical_variables_raises_error_when_no_numerical_variables(df, df_int):
    msg = (
        "No numerical variables found in this dataframe. Please check "
        "variable format with pandas dtypes."
    )
    with pytest.raises(TypeError) as record:
        assert find_numerical_variables(df.drop(["Age", "Marks"], axis=1))
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert find_numerical_variables(df_int.drop([3, 4], axis=1))
    assert str(record.value) == msg


def test_categorical_variables_finds_categorical_variables(df, df_int):
    assert find_categorical_variables(df) == ["Name", "City"]
    assert find_categorical_variables(df_int) == [1, 2]


def test_categorical_variables_raises_error_when_no_categorical_variables(df, df_int):
    msg = (
        "No categorical variables found in this dataframe. Please check "
        "variable format with pandas dtypes."
    )
    with pytest.raises(TypeError) as record:
        assert find_categorical_variables(df.drop(["Name", "City"], axis=1))
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert find_categorical_variables(df_int.drop([1, 2], axis=1))
    assert str(record.value) == msg


def test_datetime_variables_finds_datetime_variables(df_datetime):
    vars_dt = [
        "date_range",
        "date_obj0",
        "date_obj1",
        "date_obj2",
        "time_obj",
        "time_objTZ",
    ]

    assert find_datetime_variables(df_datetime) == vars_dt

    assert find_datetime_variables(
        df_datetime[vars_dt].reindex(columns=["date_obj1", "date_range", "date_obj2"]),
    ) == ["date_obj1", "date_range", "date_obj2"]


def test_datetime_variables_raises_error_when_no_datetime_variables(df_datetime):

    msg = "No datetime variables found in this dataframe."

    vars_nondt = ["Marks", "Age", "Name"]

    with pytest.raises(ValueError) as record:
        assert find_datetime_variables(df_datetime.loc[:, vars_nondt])
    assert str(record.value) == msg
