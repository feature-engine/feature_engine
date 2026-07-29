import pandas as pd
import polars as pl
import pytest

from feature_engine.variable_handling import (
    check_all_variables,
    check_categorical_variables,
    check_datetime_variables,
    check_numerical_variables,
)
from tests.test_variable_handling.conftest import (
    BASIC_DATA,
    DATETIME_DATA,
    cast_categorical,
)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_check_numerical_variables_returns_numerical_variables(make_df):
    df = make_df(BASIC_DATA)
    assert check_numerical_variables(df, ["Age", "Marks"]) == ["Age", "Marks"]
    assert check_numerical_variables(df, ["Age"]) == ["Age"]
    assert check_numerical_variables(df, "Age") == ["Age"]


def test_check_numerical_variables_returns_numerical_variables_int_names(df_int):
    # polars requires string column names, so int-named columns are pandas-only
    assert check_numerical_variables(df_int, [3, 4]) == [3, 4]
    assert check_numerical_variables(df_int, [3]) == [3]
    assert check_numerical_variables(df_int, 4) == [4]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_check_numerical_variables_raises_errors_when_not_numerical(make_df):
    df = make_df(BASIC_DATA)
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
        assert check_numerical_variables(df, ["Name", "Marks"])
    assert str(record.value) == msg


def test_check_numerical_variables_raises_errors_int_names(df_int):
    msg = (
        "Some of the variables are not numerical. Please cast them as "
        "numerical before using this transformer."
    )
    with pytest.raises(TypeError) as record:
        assert check_numerical_variables(df_int, 1)
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_numerical_variables(df_int, [1])
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_numerical_variables(df_int, [2, 3])
    assert str(record.value) == msg


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_check_categorical_variables_returns_categorical_variables(make_df):
    df = make_df(BASIC_DATA)
    assert check_categorical_variables(df, ["Name", "City"]) == ["Name", "City"]
    assert check_categorical_variables(df, ["Name"]) == ["Name"]
    assert check_categorical_variables(df, "Name") == ["Name"]


def test_check_categorical_variables_numeric_categories_pandas_only():
    # pandas allows category dtype with numeric categories (pd.Categorical can
    # wrap numbers); polars categoricals are always string-backed, so casting a
    # numeric column to Categorical isn't a realistic polars scenario.
    df = pd.DataFrame(BASIC_DATA)
    df = cast_categorical(df, ["Age", "Marks"])
    assert check_categorical_variables(df, ["Age", "Marks"]) == ["Age", "Marks"]


def test_check_categorical_variables_returns_categorical_variables_int_names(df_int):
    assert check_categorical_variables(df_int, [1, 2]) == [1, 2]
    assert check_categorical_variables(df_int, [2]) == [2]
    assert check_categorical_variables(df_int, 2) == [2]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_check_categorical_variables_raises_errors_when_not_categorical(make_df):
    df = make_df(BASIC_DATA)
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
        assert check_categorical_variables(df, ["Name", "Marks"])
    assert str(record.value) == msg


def test_check_categorical_variables_raises_errors_int_names(df_int):
    msg = (
        "Some of the variables are not categorical. Please cast them as "
        "object or categorical before using this transformer."
    )
    with pytest.raises(TypeError) as record:
        assert check_categorical_variables(df_int, 3)
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_categorical_variables(df_int, [3])
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_categorical_variables(df_int, [2, 3])
    assert str(record.value) == msg


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_check_datetime_variables_returns_datetime_variables(make_df):
    df = make_df(DATETIME_DATA)
    var_dt = ["date_range"]
    var_dt_str = "date_range"
    vars_dt = ["date_range", "date_obj0", "date_range_tz"]
    tz_time = "date_range_tz"

    assert check_datetime_variables(df, var_dt_str) == [var_dt_str]
    assert check_datetime_variables(df, var_dt) == var_dt
    assert check_datetime_variables(df, vars_dt) == vars_dt
    assert check_datetime_variables(df, tz_time) == [tz_time]

    # only the string column can be cast to categorical - native Datetime
    # columns can't be cast to Categorical in polars
    df = cast_categorical(df, ["date_obj0"])
    assert check_datetime_variables(df, "date_obj0") == ["date_obj0"]


def test_check_datetime_variables_returns_pandas_only_string_formats(df_datetime):
    # "01-Jan-2010"-style and "10/11/12"-style strings are only ever recognised
    # through pandas' flexible, dateutil-backed guessing - see the note in
    # check_datetime_variables' docstring.
    vars_convertible_to_dt = ["date_range", "date_obj1", "date_obj2", "time_obj"]
    var_convertible_to_dt = "date_obj1"

    assert check_datetime_variables(df_datetime, var_convertible_to_dt) == [
        var_convertible_to_dt
    ]
    assert (
        check_datetime_variables(df_datetime, vars_convertible_to_dt)
        == vars_convertible_to_dt
    )

    df_datetime[vars_convertible_to_dt] = df_datetime[vars_convertible_to_dt].astype(
        pd.CategoricalDtype
    )
    assert (
        check_datetime_variables(df_datetime, vars_convertible_to_dt)
        == vars_convertible_to_dt
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_check_datetime_variables_raises_errors_when_not_datetime(make_df):
    df = make_df(DATETIME_DATA)
    msg = "Some of the variables are not or cannot be parsed as datetime."

    with pytest.raises(TypeError) as record:
        assert check_datetime_variables(df, variables="Age")
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert check_datetime_variables(df, variables=["Age", "Name"])
    assert str(record.value) == msg

    with pytest.raises(TypeError):
        assert check_datetime_variables(df, variables=["date_range", "Age"])
    assert str(record.value) == msg


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "input_vars",
    [
        ["Name", "City", "Age", "Marks"],
        ["Name", "City", "Age"],
        "Name",
        ["Age"],
    ],
)
def test_check_all_variables_returns_all_variables(make_df, input_vars):
    df = make_df(BASIC_DATA)
    if isinstance(input_vars, list):
        assert check_all_variables(df, input_vars) == input_vars
    else:
        assert check_all_variables(df, input_vars) == [input_vars]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "input_vars", [["Name", "City", "Absent"], "Absent", ["Absent"]]
)
def test_check_all_variables_raises_errors_when_not_in_dataframe(make_df, input_vars):
    df = make_df(BASIC_DATA)
    msg_ls = "'Some of the variables are not in the dataframe.'"
    msg_single = "'The variable Absent is not in the dataframe.'"

    with pytest.raises(KeyError) as record:
        assert check_all_variables(df, input_vars)
    if isinstance(input_vars, list):
        assert str(record.value) == msg_ls
    else:
        assert str(record.value) == msg_single
