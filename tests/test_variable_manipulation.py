
import pytest
from feature_engine.variable_manipulation import (_define_variables,
                                                  _find_all_variables,
                                                  _find_categorical_variables,
                                                  _find_numerical_variables)


def test_define_variables():
    vars_ls = ["var1", "var2", "var1"]
    vars_none = None
    vars_str = "var1"
    vars_tuple = ("var1", "var2")
    vars_set = {"var1", "var2"}
    vars_dict = {"var1":1, "var2":2}

    assert _define_variables(vars_ls) == vars_ls
    assert _define_variables(vars_none) == vars_none
    assert _define_variables(vars_str) == [vars_str]

    with pytest.raises(ValueError):
        assert _define_variables(vars_tuple)

    with pytest.raises(ValueError):
        assert _define_variables(vars_set)
    
    with pytest.raises(ValueError):
        assert _define_variables(vars_dict)


def test_find_numerical_variables(df_vartypes):
    vars_num = ["Age", "Marks"]
    vars_mix = ["Age", "Marks", "Name"]
    vars_none = None

    assert _find_numerical_variables(df_vartypes, vars_num) == vars_num
    assert _find_numerical_variables(df_vartypes, vars_none) == vars_num

    with pytest.raises(TypeError):
        assert _find_numerical_variables(df_vartypes, vars_mix)

    with pytest.raises(ValueError):
        assert _find_numerical_variables(df_vartypes[["Name", "City"]], None)


def test_find_categorical_variables(df_vartypes):
    vars_cat = ["Name", "City"]
    vars_mix = ["Age", "Marks", "Name"]
    vars_none = None

    assert _find_categorical_variables(df_vartypes, vars_cat) == vars_cat
    assert _find_categorical_variables(df_vartypes, vars_none) == vars_cat

    with pytest.raises(TypeError):
        assert _find_categorical_variables(df_vartypes, vars_mix)

    with pytest.raises(ValueError):
        assert _find_categorical_variables(df_vartypes[["Age", "Marks"]], None)


def test_find_all_variables(df_vartypes):
    all_vars = ["Name", "City", "Age", "Marks", "dob"]
    user_vars = ["Name", "City"]
    non_existing_vars = ["Grades"]

    assert _find_all_variables(df_vartypes) == all_vars
    assert _find_all_variables(df_vartypes, ["Name", "City"]) == user_vars

    with pytest.raises(TypeError):
        assert _find_all_variables(df_vartypes, non_existing_vars)
