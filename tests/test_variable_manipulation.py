import pandas as pd
import pytest

from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
    _find_or_check_categorical_variables,
    _find_or_check_datetime_variables,
    _find_or_check_numerical_variables,
)


def test_check_input_parameter_variables():
    vars_ls = ["var1", "var2", "var1"]
    vars_int_ls = [0, 1, 2, 3]
    vars_none = None
    vars_str = "var1"
    vars_int = 0
    vars_tuple = ("var1", "var2")
    vars_set = {"var1", "var2"}
    vars_dict = {"var1": 1, "var2": 2}

    assert _check_input_parameter_variables(vars_ls) == ["var1", "var2", "var1"]
    assert _check_input_parameter_variables(vars_int_ls) == [0, 1, 2, 3]
    assert _check_input_parameter_variables(vars_none) is None
    assert _check_input_parameter_variables(vars_str) == "var1"
    assert _check_input_parameter_variables(vars_int) == 0

    with pytest.raises(ValueError):
        assert _check_input_parameter_variables(vars_tuple)

    with pytest.raises(ValueError):
        assert _check_input_parameter_variables(vars_set)

    with pytest.raises(ValueError):
        assert _check_input_parameter_variables(vars_dict)


def test_find_or_check_numerical_variables(df_vartypes, df_numeric_columns):
    vars_num = ["Age", "Marks"]
    var_num = "Age"
    vars_mix = ["Age", "Marks", "Name"]
    vars_none = None

    assert _find_or_check_numerical_variables(df_vartypes, vars_num) == vars_num
    assert _find_or_check_numerical_variables(df_vartypes, var_num) == ["Age"]
    assert _find_or_check_numerical_variables(df_vartypes, vars_none) == vars_num

    with pytest.raises(TypeError):
        assert _find_or_check_numerical_variables(df_vartypes, vars_mix)

    with pytest.raises(ValueError):
        assert _find_or_check_numerical_variables(df_vartypes[["Name", "City"]], None)

    assert _find_or_check_numerical_variables(df_numeric_columns, [2, 3]) == [2, 3]
    assert _find_or_check_numerical_variables(df_numeric_columns, 2) == [2]


def test_find_or_check_categorical_variables(df_vartypes, df_numeric_columns):
    vars_cat = ["Name", "City"]
    vars_mix = ["Age", "Marks", "Name"]

    assert _find_or_check_categorical_variables(df_vartypes, vars_cat) == vars_cat
    assert _find_or_check_categorical_variables(df_vartypes, None) == vars_cat

    with pytest.raises(TypeError):
        assert _find_or_check_categorical_variables(df_vartypes, vars_mix)

    with pytest.raises(ValueError):
        assert _find_or_check_categorical_variables(df_vartypes[["Age", "Marks"]], None)

    assert _find_or_check_categorical_variables(df_numeric_columns, [0, 1]) == [0, 1]
    assert _find_or_check_categorical_variables(df_numeric_columns, 1) == [1]

    df_vartypes["Age"] = df_vartypes["Age"].astype("category")
    assert _find_or_check_categorical_variables(df_vartypes, None) == vars_cat + ["Age"]
    assert _find_or_check_categorical_variables(df_vartypes, ["Name", "Age"]) == [
        "Name",
        "Age",
    ]


def test_find_or_check_datetime_variables(df_datetime):
    vars_dt = ["datetime_range"]
    var_dt = "datetime_range"
    vars_nondt = ["Age", "Name"]
    vars_convertible_to_dt = ["datetime_range", "date_obj1", "date_obj2", "time_obj"]
    var_convertible_to_dt = "date_obj1"
    vars_mix = ["datetime_range", "Age"]
    cat_date = pd.DataFrame(
        {"date_obj1_cat": df_datetime["date_obj1"].astype("category")}
    )
    tz_date = pd.DataFrame({"date_obj2Z": df_datetime["date_obj2"].add("T12Z")})

    # check errors raised
    with pytest.raises(ValueError):
        assert _find_or_check_datetime_variables(
            df_datetime.loc[:, vars_nondt], variables=None
        )
    with pytest.raises(ValueError):
        assert _find_or_check_datetime_variables(
            df_datetime[vars_nondt].join(cat_date), variables=None
        )
    with pytest.raises(TypeError):
        assert _find_or_check_datetime_variables(df_datetime, variables="Age")
    with pytest.raises(TypeError):
        assert _find_or_check_datetime_variables(df_datetime, variables=vars_nondt)
    with pytest.raises(TypeError):
        assert _find_or_check_datetime_variables(df_datetime, variables=vars_mix)
    with pytest.raises(TypeError):
        assert _find_or_check_datetime_variables(
            df_datetime[vars_convertible_to_dt].join(cat_date),
            variables=vars_convertible_to_dt + ["date_obj1_cat"],
        )

    # when variables=None
    assert (
        _find_or_check_datetime_variables(df_datetime, variables=None)
        == vars_convertible_to_dt
    )
    assert (
        _find_or_check_datetime_variables(
            df_datetime[vars_convertible_to_dt].reindex(
                columns=["date_obj1", "datetime_range", "date_obj2"]
            ),
            variables=None,
        )
        == ["date_obj1", "datetime_range", "date_obj2"]
    )

    # when variables are specified
    assert _find_or_check_categorical_variables(df_datetime, var_dt) == [var_dt]
    assert _find_or_check_datetime_variables(df_datetime, vars_dt) == vars_dt
    assert _find_or_check_datetime_variables(
        df_datetime, variables=var_convertible_to_dt
    ) == [var_convertible_to_dt]
    assert (
        _find_or_check_datetime_variables(df_datetime, variables=vars_convertible_to_dt)
        == vars_convertible_to_dt
    )
    assert (
        _find_or_check_datetime_variables(
            df_datetime.join(tz_date),
            variables=None,
        )
        == vars_convertible_to_dt + ["date_obj2Z"]
    )


def test_find_all_variables(df_vartypes):
    all_vars = ["Name", "City", "Age", "Marks", "dob"]
    user_vars = ["Name", "City"]
    non_existing_vars = ["Grades"]

    assert _find_all_variables(df_vartypes) == all_vars
    assert _find_all_variables(df_vartypes, ["Name", "City"]) == user_vars

    with pytest.raises(KeyError):
        assert _find_all_variables(df_vartypes, non_existing_vars)
