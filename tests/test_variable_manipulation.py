import pandas as pd
import pytest

from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_all_variables,
    _find_or_check_categorical_variables,
    _find_or_check_datetime_variables,
    _find_or_check_numerical_variables,
    _find_categorical_and_numerical_variables,
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
        assert _find_or_check_numerical_variables(df_vartypes, "City")

    with pytest.raises(TypeError):
        assert _find_or_check_numerical_variables(df_numeric_columns, 0)

    with pytest.raises(TypeError):
        assert _find_or_check_numerical_variables(df_numeric_columns, [1, 3])

    with pytest.raises(TypeError):
        assert _find_or_check_numerical_variables(df_vartypes, vars_mix)

    with pytest.raises(ValueError):
        assert _find_or_check_numerical_variables(df_vartypes, variables=[])

    with pytest.raises(ValueError):
        assert _find_or_check_numerical_variables(df_vartypes[["Name", "City"]], None)

    assert _find_or_check_numerical_variables(df_numeric_columns, [2, 3]) == [2, 3]
    assert _find_or_check_numerical_variables(df_numeric_columns, 2) == [2]


def _cast_var_as_type(df, var, new_type):
    df_copy = df.copy()
    df_copy[var] = df[var].astype(new_type)
    return df_copy


def test_find_or_check_categorical_variables(
    df_vartypes, df_datetime, df_numeric_columns
):
    vars_cat = ["Name", "City"]
    vars_mix = ["Age", "Marks", "Name"]

    # errors when vars entered by user are not categorical
    with pytest.raises(TypeError):
        assert _find_or_check_categorical_variables(df_vartypes, "Marks")
    with pytest.raises(TypeError):
        assert _find_or_check_categorical_variables(df_datetime, "datetime_range")
    with pytest.raises(TypeError):
        assert _find_or_check_categorical_variables(df_datetime, ["datetime_range"])
    with pytest.raises(TypeError):
        assert _find_or_check_categorical_variables(df_numeric_columns, 3)
    with pytest.raises(TypeError):
        assert _find_or_check_categorical_variables(df_numeric_columns, [0, 2])
    with pytest.raises(TypeError):
        assert _find_or_check_categorical_variables(df_vartypes, vars_mix)

    # error when user enters empty list
    with pytest.raises(ValueError):
        assert _find_or_check_categorical_variables(df_vartypes, variables=[])

    # error when df has no categorical variables
    with pytest.raises(ValueError):
        assert _find_or_check_categorical_variables(df_vartypes[["Age", "Marks"]], None)
    with pytest.raises(ValueError):
        assert _find_or_check_categorical_variables(
            df_datetime[["date_obj1", "time_obj"]], None
        )

    # when variables=None
    assert _find_or_check_categorical_variables(df_vartypes, None) == vars_cat
    assert _find_or_check_categorical_variables(df_datetime, None) == ["Name"]

    # when vars are specified
    assert _find_or_check_categorical_variables(df_vartypes, "Name") == ["Name"]
    assert _find_or_check_categorical_variables(df_datetime, "date_obj1") == [
        "date_obj1"
    ]
    assert _find_or_check_categorical_variables(df_vartypes, vars_cat) == vars_cat
    assert _find_or_check_categorical_variables(df_datetime, ["Name", "date_obj1"]) == [
        "Name",
        "date_obj1",
    ]

    # vars specified, column name is integer
    assert _find_or_check_categorical_variables(df_numeric_columns, [0, 1]) == [0, 1]
    assert _find_or_check_categorical_variables(df_numeric_columns, 0) == [0]
    assert _find_or_check_categorical_variables(df_numeric_columns, 1) == [1]

    # datetime vars cast as category
    # object-like datetime
    df_datetime["date_obj1"] = df_datetime["date_obj1"].astype("category")
    assert _find_or_check_categorical_variables(df_datetime, None) == ["Name"]
    assert _find_or_check_categorical_variables(df_datetime, ["Name", "date_obj1"]) == [
        "Name",
        "date_obj1",
    ]
    # datetime64
    df_datetime["datetime_range"] = df_datetime["datetime_range"].astype("category")
    assert _find_or_check_categorical_variables(df_datetime, None) == ["Name"]
    assert _find_or_check_categorical_variables(
        df_datetime, ["Name", "datetime_range"]
    ) == ["Name", "datetime_range"]

    # time-aware datetime var
    tz_time = pd.DataFrame(
        {"time_objTZ": df_datetime["time_obj"].add(["+5", "+11", "-3", "-8"])}
    )
    with pytest.raises(ValueError):
        assert _find_or_check_categorical_variables(tz_time, None)
    assert _find_or_check_categorical_variables(tz_time, "time_objTZ") == ["time_objTZ"]


@pytest.mark.parametrize(
    "_num_var, _cat_type",
    [("Age", "category"), ("Age", "O"), ("Marks", "category"), ("Marks", "O")],
)
def test_find_or_check_categorical_variables_when_numeric_is_cast_as_category_or_object(
    df_vartypes, _num_var, _cat_type
):
    df_vartypes = _cast_var_as_type(df_vartypes, _num_var, _cat_type)
    assert _find_or_check_categorical_variables(df_vartypes, _num_var) == [_num_var]
    assert _find_or_check_categorical_variables(df_vartypes, None) == [
        "Name",
        "City",
        _num_var,
    ]
    assert _find_or_check_categorical_variables(df_vartypes, ["Name", _num_var]) == [
        "Name",
        _num_var,
    ]


def test_find_or_check_datetime_variables(df_datetime):
    var_dt = ["datetime_range"]
    var_dt_str = "datetime_range"
    vars_nondt = ["Age", "Name"]
    vars_convertible_to_dt = ["datetime_range", "date_obj1", "date_obj2", "time_obj"]
    var_convertible_to_dt = "date_obj1"
    vars_mix = ["datetime_range", "Age"]
    tz_time = pd.DataFrame(
        {"time_objTZ": df_datetime["time_obj"].add(["+5", "+11", "-3", "-8"])}
    )

    # error when df has no datetime variables
    with pytest.raises(ValueError):
        assert _find_or_check_datetime_variables(
            df_datetime.loc[:, vars_nondt], variables=None
        )

    # errors when vars entered by user are not datetime
    with pytest.raises(TypeError):
        assert _find_or_check_datetime_variables(df_datetime, variables="Age")
    with pytest.raises(TypeError):
        assert _find_or_check_datetime_variables(df_datetime, variables=vars_nondt)
    with pytest.raises(TypeError):
        assert _find_or_check_datetime_variables(df_datetime, variables=vars_mix)

    # error when user enters empty list
    with pytest.raises(ValueError):
        assert _find_or_check_datetime_variables(df_datetime, variables=[])

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
    assert _find_or_check_datetime_variables(df_datetime, var_dt_str) == [var_dt_str]
    assert _find_or_check_datetime_variables(df_datetime, var_dt) == var_dt
    assert _find_or_check_datetime_variables(
        df_datetime, variables=var_convertible_to_dt
    ) == [var_convertible_to_dt]
    assert (
        _find_or_check_datetime_variables(df_datetime, variables=vars_convertible_to_dt)
        == vars_convertible_to_dt
    )
    assert (
        _find_or_check_datetime_variables(
            df_datetime.join(tz_time),
            variables=None,
        )
        == vars_convertible_to_dt + ["time_objTZ"]
    )

    # datetime var cast as categorical
    df_datetime["date_obj1"] = df_datetime["date_obj1"].astype("category")
    assert _find_or_check_datetime_variables(df_datetime, variables="date_obj1") == [
        "date_obj1"
    ]
    assert (
        _find_or_check_datetime_variables(df_datetime, variables=vars_convertible_to_dt)
        == vars_convertible_to_dt
    )


@pytest.mark.parametrize("_num_var, _cat_type", [("Age", "category"), ("Age", "O")])
def test_find_or_check_datetime_variables_when_numeric_is_cast_as_category_or_object(
    df_datetime, _num_var, _cat_type
):
    df_datetime = _cast_var_as_type(df_datetime, _num_var, _cat_type)
    with pytest.raises(TypeError):
        assert _find_or_check_datetime_variables(df_datetime, variables=_num_var)
    with pytest.raises(TypeError):
        assert _find_or_check_datetime_variables(df_datetime, variables=[_num_var])
    with pytest.raises(ValueError) as errinfo:
        assert _find_or_check_datetime_variables(
            df_datetime[[_num_var]], variables=None
        )
    assert str(errinfo.value) == "No datetime variables found in this dataframe."
    assert _find_or_check_datetime_variables(df_datetime, variables=None) == [
        "datetime_range",
        "date_obj1",
        "date_obj2",
        "time_obj",
    ]


def test_find_all_variables(df_vartypes):
    all_vars = ["Name", "City", "Age", "Marks", "dob"]
    user_vars = ["Name", "City"]
    non_existing_vars = ["Grades"]

    assert _find_all_variables(df_vartypes) == all_vars
    assert _find_all_variables(df_vartypes, ["Name", "City"]) == user_vars

    with pytest.raises(KeyError):
        assert _find_all_variables(df_vartypes, non_existing_vars)


def test_find_categorical_and_numeric_variables_one_categorical_variables(
        df_enc
):
    assert (_find_categorical_and_numerical_variables(
        df_enc, "var_A") == (["var_A"], [])
            )


def test_find_categorical_and_numerics_variables_one_numeric_variable(
        df_enc_numeric
):
    assert (_find_categorical_and_numerical_variables(
        df_enc_numeric, "var_B") == ([], ["var_B"])
            )


def test_error_find_categorical_and_numerical_vars_datetime_var(
        df_datetime
):
    # TODO: Fix test. Not returning type error.
    with pytest.raises(TypeError):
        _find_categorical_and_numerical_variables(
            df_datetime, ["datetime_range"]
        )


def test_find_categorical_and_numeric_vars_df_contains_num_and_cat_vars(
        df_enc_categorical_and_numeric
):
    assert (_find_categorical_and_numerical_variables(
        df_enc_categorical_and_numeric, None)
        == (["var_A", "var_B"], ["var_C", "var_D", "target"]))


def test_find_categorical_and_numeric_variables_df_contains_num_vars(
        df_enc_numeric
):
    assert (_find_categorical_and_numerical_variables(
        df_enc_numeric, None)
        == ([], ["var_A", "var_B", "target"]))


def test_find_categorical_and_numeric_variables_df_contains_cat_vars(
        df_enc
):
    df_new = df_enc[["var_A", "var_B"]].copy()
    assert (_find_categorical_and_numerical_variables(
        df_new, None)
        == (["var_A", "var_B"], []))


def test_error_find_categorical_and_numeric_variables_pass_empty_list(
        df_enc
):
    with pytest.raises(ValueError):
        _find_categorical_and_numerical_variables(
            df_enc, []
        )


def test_find_categorical_and_numerical_variables_user_passes_num_and_cat_vars(
        df_enc_categorical_and_numeric
):
    assert (_find_categorical_and_numerical_variables(
        df_enc_categorical_and_numeric, ["var_A", "var_C", "target"])
        == (["var_A"], ["var_C", "target"]))


def test_find_categorical_and_numeric_variables_user_passes_cat_vars(
        df_enc_categorical_and_numeric
):
    assert (_find_categorical_and_numerical_variables(
        df_enc_categorical_and_numeric, ["var_A", "var_B"])
        == (["var_A", "var_B"], []))


def test_find_categorical_and_numeric_variables_user_passes_num_vars(
        df_enc_categorical_and_numeric
):
    assert (_find_categorical_and_numerical_variables(
        df_enc_categorical_and_numeric, ["var_C", "target"])
        == ([], ["var_C", "target"]))


#TODO:
# add tests for new function: find_numerical_and_categorical_variables
# test the following:
# user passes 1 variable that is categorical
# user passes 1 variable that is numerical
# user passes q variable in format datetime, function raises error
# user passes None, df contains both numerical and categorical
# user passes None, df contains numerical
# user passes None, df contains categorical
# user passes empty list, function raises error
# user passes list with num and cat vars
# user passes list with cat vars
# user passes list with num vars
