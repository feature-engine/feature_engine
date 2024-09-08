import pandas as pd
import pytest

from feature_engine.variable_handling import (
    check_all_variables,
    check_categorical_variables,
    check_datetime_variables,
    check_numerical_variables,
)


def test_check_numerical_variables_returns_numerical_variables(df, df_int):
    assert check_numerical_variables(df, ["Age", "Marks"]) == ["Age", "Marks"]
    assert check_numerical_variables(df, ["Age"]) == ["Age"]
    assert check_numerical_variables(df, "Age") == ["Age"]
    assert check_numerical_variables(df_int, [3, 4]) == [3, 4]
    assert check_numerical_variables(df_int, [3]) == [3]
    assert check_numerical_variables(df_int, 4) == [4]


def test_check_numerical_variables_raises_errors_when_not_numerical(df, df_int):
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


def test_check_categorical_variables_returns_categorical_variables(df, df_int):
    assert check_categorical_variables(df, ["Name", "date_obj0"]) == [
        "Name",
        "date_obj0",
    ]
    assert check_categorical_variables(df, ["Name"]) == ["Name"]
    assert check_categorical_variables(df, "date_obj0") == ["date_obj0"]
    assert check_categorical_variables(df_int, [1, 2]) == [1, 2]
    assert check_categorical_variables(df_int, [2]) == [2]
    assert check_categorical_variables(df_int, 2) == [2]

    df[["Age", "Marks"]] = df[["Age", "Marks"]].astype(pd.CategoricalDtype)
    assert check_categorical_variables(df, ["Age", "Marks"]) == ["Age", "Marks"]


def test_check_categorical_variables_raises_errors_when_not_categorical(df, df_int):
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


def test_check_datetime_variables_returns_datetime_variables(df_datetime):
    var_dt = ["date_range"]
    var_dt_str = "date_range"
    vars_convertible_to_dt = ["date_range", "date_obj1", "date_obj2", "time_obj"]
    var_convertible_to_dt = "date_obj1"
    tz_time = "time_objTZ"
    tz_time_obj = "date_range_tz"

    # when variables are specified
    assert check_datetime_variables(df_datetime, var_dt_str) == [var_dt_str]
    assert check_datetime_variables(df_datetime, var_dt) == var_dt
    assert check_datetime_variables(df_datetime, var_convertible_to_dt) == [
        var_convertible_to_dt
    ]
    assert (
        check_datetime_variables(df_datetime, vars_convertible_to_dt)
        == vars_convertible_to_dt
    )
    assert check_datetime_variables(df_datetime, tz_time) == [tz_time]
    assert check_datetime_variables(df_datetime, tz_time_obj) == [tz_time_obj]

    df_datetime[vars_convertible_to_dt] = df_datetime[vars_convertible_to_dt].astype(
        pd.CategoricalDtype
    )
    assert (
        check_datetime_variables(df_datetime, vars_convertible_to_dt)
        == vars_convertible_to_dt
    )


def test_check_datetime_variables_raises_errors_when_not_datetime(df_datetime):
    msg = "Some of the variables are not or cannot be parsed as datetime."

    with pytest.raises(TypeError) as record:
        assert check_datetime_variables(df_datetime, variables="Age")
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_datetime_variables(df_datetime, variables=["Age", "Name"])
    assert str(record.value) == msg

    with pytest.raises(TypeError):
        assert check_datetime_variables(df_datetime, variables=["date_range", "Age"])
    assert str(record.value) == msg


@pytest.mark.parametrize(
    "input_vars",
    [
        ["Name", "City", "Age", "Marks", "dob"],
        [
            "Name",
            "City",
            "Age",
            "Marks",
        ],
        "Name",
        ["Age"],
    ],
)
def test_check_all_variables_returns_all_variables(df_vartypes, input_vars):
    if isinstance(input_vars, list):
        assert check_all_variables(df_vartypes, input_vars) == input_vars
    else:
        assert check_all_variables(df_vartypes, input_vars) == [input_vars]


@pytest.mark.parametrize(
    "input_vars", [["Name", "City", "Absent"], "Absent", ["Absent"]]
)
def test_check_all_variables_raises_errors_when_not_in_dataframe(
    df_vartypes, input_vars
):
    msg_ls = "'Some of the variables are not in the dataframe.'"
    msg_single = "'The variable Absent is not in the dataframe.'"

    with pytest.raises(KeyError) as record:
        assert check_all_variables(df_vartypes, input_vars)
    if isinstance(input_vars, list):
        assert str(record.value) == msg_ls
    else:
        assert str(record.value) == msg_single
