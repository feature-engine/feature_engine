import numpy as np
import pandas as pd
import pytest

from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _convert_variable_to_datetime,
    _convert_variables_to_datetime,
    _find_all_variables,
    _find_or_check_categorical_variables,
    _find_or_check_datetime_variables,
    _find_or_check_numerical_variables,
)


def test_convert_variable_to_datetime(df_vartypes2):

    # check dtypes upon conversion
    assert _convert_variable_to_datetime(df_vartypes2.Name).dtype == "O"
    assert (
        _convert_variable_to_datetime(
            df_vartypes2.City.astype("category")
        ).dtype.categories.dtype
        == "O"
    )
    assert (
        _convert_variable_to_datetime(
            df_vartypes2.Age.astype("category")
        ).dtype.categories.dtype
        == "int64"
    )
    assert _convert_variable_to_datetime(
        df_vartypes2.dob.astype("category")
    ).dtype.categories.dtype == np.dtype("datetime64[ns]")
    assert _convert_variable_to_datetime(df_vartypes2.dob).dtype == np.dtype(
        "datetime64[ns]"
    )
    assert _convert_variable_to_datetime(df_vartypes2.Age).dtype == "int64"
    assert _convert_variable_to_datetime(df_vartypes2.City).dtype == "O"
    assert _convert_variable_to_datetime(df_vartypes2.doa).dtype == np.dtype(
        "datetime64[ns]"
    )
    assert _convert_variable_to_datetime(
        df_vartypes2.doa.astype("category")
    ).dtype.categories.dtype == np.dtype("datetime64[ns]")

    # add a UTC timezone awareness
    assert hasattr(_convert_variable_to_datetime(df_vartypes2.dof.add("T12Z")), "dt")

    # check different datetime formats passing kwargs
    pd.testing.assert_series_equal(
        _convert_variable_to_datetime(df_vartypes2.doa),
        pd.Series(
            np.array(
                ["2010-12-01", "1945-02-24", "2100-06-14", "1999-05-17"],
                dtype=np.datetime64,
            ),
            name="doa",
        ),
    )
    pd.testing.assert_series_equal(
        _convert_variable_to_datetime(df_vartypes2.dof, dayfirst=True),
        pd.Series(
            np.array(
                ["2012-11-10", "2009-09-09", "1995-05-25", "2004-03-17"],
                dtype=np.datetime64,
            ),
            name="dof",
        ),
    )
    pd.testing.assert_series_equal(
        _convert_variable_to_datetime(df_vartypes2.dof, yearfirst=True),
        pd.Series(
            np.array(
                ["2010-11-12", "2009-09-09", "1995-05-25", "2004-03-17"],
                dtype=np.datetime64,
            ),
            name="dof",
        ),
    )

    # check behavior with missing values
    series_dt_na = pd.Series(["Dec-1990", np.NaN])
    series_dt_conv = pd.Series(np.array(["1990-12-01", "NaT"], dtype=np.datetime64))
    pd.testing.assert_series_equal(
        _convert_variable_to_datetime(series_dt_na), series_dt_conv
    )
    pd.testing.assert_series_equal(
        _convert_variable_to_datetime(series_dt_na.astype("category")),
        series_dt_conv.astype("category"),
    )


def test_convert_variables_to_datetime(df_vartypes2):

    vars_to_dt1 = ["Name", "Age", "doa"]
    vars_to_dt2 = ["Name", "Age", "dob"]
    assert list(
        _convert_variables_to_datetime(df_vartypes2, vars_to_dt1).dtypes.values
    ) == [
        "O",
        "O",
        "int64",
        "float64",
        np.dtype("datetime64[ns]"),
        np.dtype("datetime64[ns]"),
        "O",
    ]
    assert list(
        _convert_variables_to_datetime(df_vartypes2, vars_to_dt2).dtypes.values
    ) == ["O", "O", "int64", "float64", np.dtype("datetime64[ns]"), "O", "O"]
    assert list(_convert_variables_to_datetime(df_vartypes2, None).dtypes.values) == [
        "O",
        "O",
        "int64",
        "float64",
        np.dtype("datetime64[ns]"),
        np.dtype("datetime64[ns]"),
        np.dtype("datetime64[ns]"),
    ]
    pd.testing.assert_frame_equal(
        _convert_variables_to_datetime(df_vartypes2, variables=["dof"], dayfirst=True),
        df_vartypes2.drop("dof", axis=1).join(
            pd.DataFrame(
                {
                    "dof": np.array(
                        ["2012-11-10", "2009-09-09", "1995-05-25", "2004-03-17"],
                        dtype=np.datetime64,
                    )
                }
            )
        ),
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
    assert _find_or_check_numerical_variables(df_vartypes, var_num) == [var_num]
    assert _find_or_check_numerical_variables(df_vartypes, vars_none) == vars_num

    with pytest.raises(TypeError):
        assert _find_or_check_numerical_variables(df_vartypes, vars_mix)

    with pytest.raises(ValueError):
        assert _find_or_check_numerical_variables(df_vartypes[["Name", "City"]], None)

    assert _find_or_check_numerical_variables(df_numeric_columns, [2, 3]) == [2, 3]
    assert _find_or_check_numerical_variables(df_numeric_columns, 2) == [2]


def test_find_or_check_categorical_variables(df_vartypes2, df_numeric_columns):
    vars_cat1 = ["Name", "City"]
    vars_cat2 = ["Name", "City", "doa"]
    vars_cat3 = ["Name", "City", "Age"]
    vars_noncat1 = ["Age", "Marks"]
    vars_noncat2 = ["Age", "doa"]
    vars_mix = ["Age", "Marks", "Name"]

    assert _find_or_check_categorical_variables(df_vartypes2, vars_cat1) == vars_cat1
    assert _find_or_check_categorical_variables(df_vartypes2, None) == vars_cat1

    with pytest.raises(TypeError):
        assert _find_or_check_categorical_variables(df_vartypes2, vars_mix)
        assert _find_or_check_categorical_variables(df_vartypes2, vars_cat2)

    with pytest.raises(ValueError):
        assert _find_or_check_categorical_variables(df_vartypes2[vars_noncat1], None)
        assert _find_or_check_categorical_variables(df_vartypes2[vars_noncat2], None)

    assert _find_or_check_categorical_variables(df_numeric_columns, [0, 1]) == [0, 1]
    assert _find_or_check_categorical_variables(df_numeric_columns, 1) == [1]

    assert (
        _find_or_check_categorical_variables(
            df_vartypes2.astype({"Age": "category"}), None
        )
        == vars_cat3
    )
    assert (
        _find_or_check_categorical_variables(
            df_vartypes2.astype({"Age": "category"}), vars_cat3
        )
        == vars_cat3
    )


def test_find_or_check_datetime_variables(df_vartypes2):
    vars_dt = ["dob"]
    var_dt = "dob"
    vars_nondt = ["Age", "Marks", "Name"]
    vars_convertible_to_dt = ["dob", "doa", "dof"]
    vars_mix = ["dob", "Age", "Marks"]

    assert _find_or_check_datetime_variables(df_vartypes2, vars_dt) == vars_dt
    assert _find_or_check_datetime_variables(df_vartypes2, var_dt) == [var_dt]
    assert (
        _find_or_check_datetime_variables(df_vartypes2, None) == vars_convertible_to_dt
    )
    assert _find_or_check_datetime_variables(
        df_vartypes2.join(pd.DataFrame({"dofZ": df_vartypes2.dof.add("T12Z")})),
        variables=None,
    ) == vars_convertible_to_dt + ["dofZ"]

    with pytest.raises(TypeError):
        _find_or_check_datetime_variables(df_vartypes2, variables="Age")
        _find_or_check_datetime_variables(df_vartypes2, variables=vars_mix)

    with pytest.raises(ValueError):
        _find_or_check_datetime_variables(df_vartypes2.loc[:, vars_nondt], None)


def test_find_all_variables(df_vartypes):
    all_vars = list(df_vartypes.columns)
    user_vars = ["Name", "City"]
    non_existing_vars = ["Grades"]

    assert _find_all_variables(df_vartypes) == all_vars
    assert _find_all_variables(df_vartypes, ["Name", "City"]) == user_vars

    with pytest.raises(KeyError):
        assert _find_all_variables(df_vartypes, non_existing_vars)
