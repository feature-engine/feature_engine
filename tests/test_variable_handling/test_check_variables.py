import pandas as pd
import pytest

from feature_engine.variable_handling import (
    check_numerical_variables,
    check_categorical_variables,
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


def test_check_numerical_variables_returns_numerical_variables():
    assert check_numerical_variables(df, ["Age", "Marks"]) == ["Age", "Marks"]
    assert check_numerical_variables(df, ["Age"]) == ["Age"]
    assert check_numerical_variables(df, "Age") == ["Age"]
    assert check_numerical_variables(df_int, [3, 4]) == [3, 4]
    assert check_numerical_variables(df_int, [3]) == [3]
    assert check_numerical_variables(df_int, 4) == [4]


def test_check_numerical_variables_raises_errors_when_not_numerical():
    msg = (
        "Some of the variables are not numerical. Please cast them as "
        "numerical before using this transformer."
    )
    with pytest.raises(TypeError) as record:
        assert check_numerical_variables(df, "Name")
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_numerical_variables(df, ["Name"])
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_numerical_variables(df_int, 1)
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_numerical_variables(df_int, [1])
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_numerical_variables(df, ["Name", "Marks"])
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_numerical_variables(df_int, [2, 3])
    assert str(record.value) == msg


def test_check_categorical_variables_returns_categorical_variables():
    assert check_categorical_variables(df, ["Name", "date"]) == ["Name", "date"]
    assert check_categorical_variables(df, ["Name"]) == ["Name"]
    assert check_categorical_variables(df, "date") == ["date"]
    assert check_categorical_variables(df_int, [1, 2]) == [1, 2]
    assert check_categorical_variables(df_int, [2]) == [2]
    assert check_categorical_variables(df_int, 2) == [2]


def test_check_categorical_variables_raises_errors_when_not_categorical():
    msg = (
        "Some of the variables are not categorical. Please cast them as "
        "object or categorical before using this transformer."
    )
    with pytest.raises(TypeError) as record:
        assert check_categorical_variables(df, "Age")
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_categorical_variables(df, ["Age"])
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_categorical_variables(df_int, 3)
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_categorical_variables(df_int, [3])
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_categorical_variables(df, ["Name", "Marks"])
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_categorical_variables(df_int, [2, 3])
    assert str(record.value) == msg
