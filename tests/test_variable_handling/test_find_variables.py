import pandas as pd
import pytest

from feature_engine.variable_handling import (
    find_numerical_variables,
    find_categorical_variables,
)

df = pd.DataFrame(
    {
        "Name": ["tom", "nick", "krish", "jack"],
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
        "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
        "date": ["2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27"],
    }
)
df["Name"] = df["Name"].astype("category")

df_int = pd.DataFrame(
    {
        1: ["tom", "nick", "krish", "jack"],
        2: ["London", "Manchester", "Liverpool", "Bristol"],
        3: [20, 21, 19, 18],
        4: [0.9, 0.8, 0.7, 0.6],
        5: pd.date_range("2020-02-24", periods=4, freq="T"),
    }
)
df_int[1] = df_int[1].astype("category")


def test_numerical_variables_finds_numerical_variables():
    assert find_numerical_variables(df) == ["Age", "Marks"]
    assert find_numerical_variables(df_int) == [3, 4]


def test_numerical_variables_raises_error_when_no_numerical_variables():
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


def test_categorical_variables_finds_categorical_variables():
    assert find_categorical_variables(df) == ["Name", "City"]
    assert find_categorical_variables(df_int) == [1, 2]


def test_categorical_variables_raises_error_when_no_categorical_variables():
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
