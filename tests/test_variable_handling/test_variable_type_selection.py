import pandas as pd
import pytest

from feature_engine.variable_handling.variable_type_selection import (
    _filter_out_variables_not_in_dataframe,
    find_all_variables,
    find_categorical_and_numerical_variables,
    find_or_check_categorical_variables,
    find_or_check_datetime_variables,
    find_or_check_numerical_variables,
)


def test_find_or_check_numerical_variables(df_vartypes, df_numeric_columns):
    vars_num = ["Age", "Marks"]
    var_num = "Age"
    vars_mix = ["Age", "Marks", "Name"]
    vars_none = None

    assert find_or_check_numerical_variables(df_vartypes, vars_num) == vars_num
    assert find_or_check_numerical_variables(df_vartypes, var_num) == ["Age"]
    assert find_or_check_numerical_variables(df_vartypes, vars_none) == vars_num

    with pytest.raises(TypeError):
        assert find_or_check_numerical_variables(df_vartypes, "City")

    with pytest.raises(TypeError):
        assert find_or_check_numerical_variables(df_numeric_columns, 0)

    with pytest.raises(TypeError):
        assert find_or_check_numerical_variables(df_numeric_columns, [1, 3])

    with pytest.raises(TypeError):
        assert find_or_check_numerical_variables(df_vartypes, vars_mix)

    with pytest.raises(ValueError):
        assert find_or_check_numerical_variables(df_vartypes, variables=[])

    with pytest.raises(ValueError):
        assert find_or_check_numerical_variables(df_vartypes[["Name", "City"]], None)

    assert find_or_check_numerical_variables(df_numeric_columns, [2, 3]) == [2, 3]
    assert find_or_check_numerical_variables(df_numeric_columns, 2) == [2]


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
        assert find_or_check_categorical_variables(df_vartypes, "Marks")
    with pytest.raises(TypeError):
        assert find_or_check_categorical_variables(df_datetime, "datetime_range")
    with pytest.raises(TypeError):
        assert find_or_check_categorical_variables(df_datetime, ["datetime_range"])
    with pytest.raises(TypeError):
        assert find_or_check_categorical_variables(df_numeric_columns, 3)
    with pytest.raises(TypeError):
        assert find_or_check_categorical_variables(df_numeric_columns, [0, 2])
    with pytest.raises(TypeError):
        assert find_or_check_categorical_variables(df_vartypes, vars_mix)

    # error when user enters empty list
    with pytest.raises(ValueError):
        assert find_or_check_categorical_variables(df_vartypes, variables=[])

    # error when df has no categorical variables
    with pytest.raises(ValueError):
        assert find_or_check_categorical_variables(df_vartypes[["Age", "Marks"]], None)
    with pytest.raises(ValueError):
        assert find_or_check_categorical_variables(
            df_datetime[["date_obj1", "time_obj"]], None
        )

    # when variables=None
    assert find_or_check_categorical_variables(df_vartypes, None) == vars_cat
    assert find_or_check_categorical_variables(df_datetime, None) == ["Name"]

    # when vars are specified
    assert find_or_check_categorical_variables(df_vartypes, "Name") == ["Name"]
    assert find_or_check_categorical_variables(df_datetime, "date_obj1") == [
        "date_obj1"
    ]
    assert find_or_check_categorical_variables(df_vartypes, vars_cat) == vars_cat
    assert find_or_check_categorical_variables(df_datetime, ["Name", "date_obj1"]) == [
        "Name",
        "date_obj1",
    ]

    # vars specified, column name is integer
    assert find_or_check_categorical_variables(df_numeric_columns, [0, 1]) == [0, 1]
    assert find_or_check_categorical_variables(df_numeric_columns, 0) == [0]
    assert find_or_check_categorical_variables(df_numeric_columns, 1) == [1]

    # datetime vars cast as category
    # object-like datetime
    df_datetime["date_obj1"] = df_datetime["date_obj1"].astype("category")
    assert find_or_check_categorical_variables(df_datetime, None) == ["Name"]
    assert find_or_check_categorical_variables(df_datetime, ["Name", "date_obj1"]) == [
        "Name",
        "date_obj1",
    ]
    # datetime64
    df_datetime["datetime_range"] = df_datetime["datetime_range"].astype("category")
    assert find_or_check_categorical_variables(df_datetime, None) == ["Name"]
    assert find_or_check_categorical_variables(
        df_datetime, ["Name", "datetime_range"]
    ) == ["Name", "datetime_range"]

    # time-aware datetime var
    tz_time = pd.DataFrame(
        {"time_objTZ": df_datetime["time_obj"].add(["+5", "+11", "-3", "-8"])}
    )
    with pytest.raises(ValueError):
        assert find_or_check_categorical_variables(tz_time, None)
    assert find_or_check_categorical_variables(tz_time, "time_objTZ") == ["time_objTZ"]


@pytest.mark.parametrize(
    "_num_var, _cat_type",
    [("Age", "category"), ("Age", "O"), ("Marks", "category"), ("Marks", "O")],
)
def test_find_or_check_categorical_variables_when_numeric_is_cast_as_category_or_object(
    df_vartypes, _num_var, _cat_type
):
    df_vartypes = _cast_var_as_type(df_vartypes, _num_var, _cat_type)
    assert find_or_check_categorical_variables(df_vartypes, _num_var) == [_num_var]
    assert find_or_check_categorical_variables(df_vartypes, None) == [
        "Name",
        "City",
        _num_var,
    ]
    assert find_or_check_categorical_variables(df_vartypes, ["Name", _num_var]) == [
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
        assert find_or_check_datetime_variables(
            df_datetime.loc[:, vars_nondt], variables=None
        )

    # errors when vars entered by user are not datetime
    with pytest.raises(TypeError):
        assert find_or_check_datetime_variables(df_datetime, variables="Age")
    with pytest.raises(TypeError):
        assert find_or_check_datetime_variables(df_datetime, variables=vars_nondt)
    with pytest.raises(TypeError):
        assert find_or_check_datetime_variables(df_datetime, variables=vars_mix)

    # error when user enters empty list
    with pytest.raises(ValueError):
        assert find_or_check_datetime_variables(df_datetime, variables=[])

    # when variables=None
    assert (
        find_or_check_datetime_variables(df_datetime, variables=None)
        == vars_convertible_to_dt
    )
    assert find_or_check_datetime_variables(
        df_datetime[vars_convertible_to_dt].reindex(
            columns=["date_obj1", "datetime_range", "date_obj2"]
        ),
        variables=None,
    ) == ["date_obj1", "datetime_range", "date_obj2"]

    # when variables are specified
    assert find_or_check_datetime_variables(df_datetime, var_dt_str) == [var_dt_str]
    assert find_or_check_datetime_variables(df_datetime, var_dt) == var_dt
    assert find_or_check_datetime_variables(
        df_datetime, variables=var_convertible_to_dt
    ) == [var_convertible_to_dt]
    assert (
        find_or_check_datetime_variables(df_datetime, variables=vars_convertible_to_dt)
        == vars_convertible_to_dt
    )
    assert find_or_check_datetime_variables(
        df_datetime.join(tz_time),
        variables=None,
    ) == vars_convertible_to_dt + ["time_objTZ"]

    # datetime var cast as categorical
    df_datetime["date_obj1"] = df_datetime["date_obj1"].astype("category")
    assert find_or_check_datetime_variables(df_datetime, variables="date_obj1") == [
        "date_obj1"
    ]
    assert (
        find_or_check_datetime_variables(df_datetime, variables=vars_convertible_to_dt)
        == vars_convertible_to_dt
    )


@pytest.mark.parametrize("_num_var, _cat_type", [("Age", "category"), ("Age", "O")])
def test_find_or_check_datetime_variables_when_numeric_is_cast_as_category_or_object(
    df_datetime, _num_var, _cat_type
):
    df_datetime = _cast_var_as_type(df_datetime, _num_var, _cat_type)
    with pytest.raises(TypeError):
        assert find_or_check_datetime_variables(df_datetime, variables=_num_var)
    with pytest.raises(TypeError):
        assert find_or_check_datetime_variables(df_datetime, variables=[_num_var])
    with pytest.raises(ValueError) as errinfo:
        assert find_or_check_datetime_variables(df_datetime[[_num_var]], variables=None)
    assert str(errinfo.value) == "No datetime variables found in this dataframe."
    assert find_or_check_datetime_variables(df_datetime, variables=None) == [
        "datetime_range",
        "date_obj1",
        "date_obj2",
        "time_obj",
    ]


def test_find_all_variables(df_vartypes):
    all_vars = ["Name", "City", "Age", "Marks", "dob"]
    all_vars_no_dt = ["Name", "City", "Age", "Marks"]
    user_vars = ["Name", "City"]
    non_existing_vars = ["Grades"]

    assert find_all_variables(df_vartypes) == all_vars
    assert find_all_variables(df_vartypes, exclude_datetime=True) == all_vars_no_dt
    assert find_all_variables(df_vartypes, ["Name", "City"]) == user_vars

    with pytest.raises(KeyError):
        assert find_all_variables(df_vartypes, non_existing_vars)


filter_dict = [
    (
        pd.DataFrame(columns=["A", "B", "C", "D", "E"]),
        ["A", "C", "B", "G", "H"],
        ["A", "C", "B"],
        ["X", "Y"],
    ),
    (pd.DataFrame(columns=[1, 2, 3, 4, 5]), [1, 2, 4, 6], [1, 2, 4], [6, 7]),
    (pd.DataFrame(columns=[1, 2, 3, 4, 5]), 1, [1], 7),
    (pd.DataFrame(columns=["A", "B", "C", "D", "E"]), "C", ["C"], "G"),
]


@pytest.mark.parametrize("df, variables, overlap, not_in_col", filter_dict)
def test_filter_out_variables_not_in_dataframe(df, variables, overlap, not_in_col):
    """Test the filter of variables not in the columns of the dataframe."""
    assert _filter_out_variables_not_in_dataframe(df, variables) == overlap

    with pytest.raises(ValueError):
        assert _filter_out_variables_not_in_dataframe(df, not_in_col)


def test_find_categorical_and_numerical_variables(df_vartypes):

    # Case 1: user passes 1 variable that is categorical
    assert find_categorical_and_numerical_variables(df_vartypes, ["Name"]) == (
        ["Name"],
        [],
    )
    assert find_categorical_and_numerical_variables(df_vartypes, "Name") == (
        ["Name"],
        [],
    )

    # Case 2: user passes 1 variable that is numerical
    assert find_categorical_and_numerical_variables(df_vartypes, ["Age"]) == (
        [],
        ["Age"],
    )
    assert find_categorical_and_numerical_variables(df_vartypes, "Age") == (
        [],
        ["Age"],
    )

    # Case 3: user passes 1 categorical and 1 numerical variable
    assert find_categorical_and_numerical_variables(df_vartypes, ["Age", "Name"]) == (
        ["Name"],
        ["Age"],
    )

    # Case 4: automatically identify variables
    assert find_categorical_and_numerical_variables(df_vartypes, None) == (
        ["Name", "City"],
        ["Age", "Marks"],
    )
    assert find_categorical_and_numerical_variables(
        df_vartypes[["Name", "City"]], None
    ) == (["Name", "City"], [])
    assert find_categorical_and_numerical_variables(
        df_vartypes[["Age", "Marks"]], None
    ) == ([], ["Age", "Marks"])

    # Case 5: error when no variable is numerical or categorical
    with pytest.raises(TypeError):
        find_categorical_and_numerical_variables(df_vartypes["dob"].to_frame(), None)

    with pytest.raises(TypeError):
        find_categorical_and_numerical_variables(df_vartypes["dob"].to_frame(), ["dob"])

    with pytest.raises(TypeError):
        find_categorical_and_numerical_variables(df_vartypes["dob"].to_frame(), "dob")

    # Case 6: user passes empty list
    with pytest.raises(ValueError):
        find_categorical_and_numerical_variables(df_vartypes, [])

    # Case 7: datetime cast as object
    df = df_vartypes.copy()
    df["dob"] = df["dob"].astype("O")

    # datetime variable is skipped when automatically finding variables, but
    # selected if user passes it in list
    assert find_categorical_and_numerical_variables(df, None) == (
        ["Name", "City"],
        ["Age", "Marks"],
    )
    assert find_categorical_and_numerical_variables(df, ["Name", "Marks", "dob"]) == (
        ["Name", "dob"],
        ["Marks"],
    )

    # Case 8: variables cast as category
    df = df_vartypes.copy()
    df["City"] = df["City"].astype("category")
    assert find_categorical_and_numerical_variables(df, None) == (
        ["Name", "City"],
        ["Age", "Marks"],
    )
    assert find_categorical_and_numerical_variables(df, "City") == (["City"], [])
