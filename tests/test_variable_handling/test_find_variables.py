import pandas as pd
import pytest

from feature_engine.variable_handling import (
    find_all_variables,
    find_categorical_and_numerical_variables,
    find_categorical_variables,
    find_datetime_variables,
    find_numerical_variables,
)

# --- find_numerical_variables --- #


def test_numerical_variables_finds_variables(df, df_int):
    assert find_numerical_variables(df) == ["Age", "Marks"]
    assert find_numerical_variables(df_int) == [3, 4]


def test_numerical_variables_raises_error(df, df_int):
    msg = "No numerical variables found in this dataframe."
    with pytest.raises(TypeError, match=msg):
        find_numerical_variables(df.drop(["Age", "Marks"], axis=1))

    with pytest.raises(TypeError, match=msg):
        find_numerical_variables(df_int.drop([3, 4], axis=1))


def test_numerical_variables_raises_warning(df, df_int):
    msg = "No numerical variables found in this dataframe."

    # Test with a regular DataFrame
    with pytest.warns(UserWarning, match=msg):
        find_numerical_variables(df.drop(["Age", "Marks"], axis=1), return_empty=True)

    # Test with integer-only DataFrame
    with pytest.warns(UserWarning, match=msg):
        find_numerical_variables(df_int.drop([3, 4], axis=1), return_empty=True)


def test_numerical_variables_returns_empty_list(df, df_int):
    assert (
        find_numerical_variables(df.drop(["Age", "Marks"], axis=1), return_empty=True)
        == []
    )
    assert (
        find_numerical_variables(df_int.drop([3, 4], axis=1), return_empty=True) == []
    )


# --- find_categorical_variables --- #


def test_categorical_variables_finds_variables(df, df_int):
    assert find_categorical_variables(df) == ["Name", "City"]
    assert find_categorical_variables(df_int) == [1, 2]


def test_categorical_variables_raises_error(df, df_int):
    msg = "No categorical variables found in this dataframe."
    with pytest.raises(TypeError, match=msg):
        find_categorical_variables(df.drop(["Name", "City"], axis=1))

    with pytest.raises(TypeError, match=msg):
        find_categorical_variables(df_int.drop([1, 2], axis=1))


def test_categorical_variables_raises_warning(df, df_int):
    msg = "No categorical variables found in this dataframe."

    # Test with a regular DataFrame
    with pytest.warns(UserWarning, match=msg):
        find_categorical_variables(df.drop(["Name", "City"], axis=1), return_empty=True)

    # Test with integer-only DataFrame
    with pytest.warns(UserWarning, match=msg):
        find_categorical_variables(df_int.drop([1, 2], axis=1), return_empty=True)


def test_categorical_variables_returns_empty_list(df, df_int):
    assert (
        find_categorical_variables(df.drop(["Name", "City"], axis=1), return_empty=True)
        == []
    )
    assert (
        find_categorical_variables(df_int.drop([1, 2], axis=1), return_empty=True) == []
    )


# --- find_datetime_variables --- #


def test_datetime_variables_finds_variables(df_datetime):
    vars_dt = [
        "date_range",
        "date_obj0",
        "date_range_tz",
        "date_obj1",
        "date_obj2",
        "time_obj",
        "time_objTZ",
    ]

    assert find_datetime_variables(df_datetime) == vars_dt

    assert find_datetime_variables(
        df_datetime[vars_dt].reindex(columns=["date_obj1", "date_range", "date_obj2"]),
    ) == ["date_obj1", "date_range", "date_obj2"]


def test_datetime_variables_raises_error(df_datetime):
    msg = "No datetime variables found in this dataframe."

    vars_nondt = ["Marks", "Age", "Name"]

    with pytest.raises(TypeError, match=msg):
        find_datetime_variables(df_datetime.loc[:, vars_nondt])


def test_datetime_variables_raises_warning(df_datetime):
    msg = "No datetime variables found in this dataframe."
    vars_nondt = ["Marks", "Age", "Name"]
    with pytest.warns(UserWarning, match=msg):
        find_datetime_variables(df_datetime.loc[:, vars_nondt], return_empty=True)


def test_datetime_variables_returns_empty_list(df_datetime):
    vars_nondt = ["Marks", "Age", "Name"]
    assert (
        find_datetime_variables(df_datetime.loc[:, vars_nondt], return_empty=True) == []
    )


# --- find_all_variables --- #


def test_find_all_variables(df):
    all_vars = [
        "Name",
        "City",
        "Age",
        "Marks",
        "date_range",
        "date_obj0",
        "date_range_tz",
    ]
    assert find_all_variables(df, exclude_datetime=False) == all_vars


def test_find_all_variables_excludes_dt(df):
    all_vars_no_dt = ["Name", "City", "Age", "Marks"]
    assert find_all_variables(df, exclude_datetime=True) == all_vars_no_dt


def test_find_all_variables_raises_error(df):
    dt_vars = [
        "date_range",
        "date_obj0",
        "date_range_tz",
    ]
    df = df[dt_vars]
    msg = "No variables found in this dataframe"
    with pytest.raises(TypeError, match=msg):
        find_all_variables(df, exclude_datetime=True)


def test_find_all_variables_raises_warning(df):
    dt_vars = [
        "date_range",
        "date_obj0",
        "date_range_tz",
    ]
    df = df[dt_vars]
    msg = "No variables found in this dataframe"
    with pytest.warns(UserWarning, match=msg):
        find_all_variables(df, exclude_datetime=True, return_empty=True)


def test_find_all_variables_returns_empty(df):
    dt_vars = [
        "date_range",
        "date_obj0",
        "date_range_tz",
    ]
    df = df[dt_vars]
    assert find_all_variables(df, exclude_datetime=True, return_empty=True) == []


# --- find_categorical_and_numerical_variables --- #


def test_numcat_user_passes_varlist(df_vartypes):
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


def test_numcat_when_var_is_none(df_vartypes):
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


@pytest.fixture(scope="module")
def dfdt():
    X = pd.DataFrame()
    X["date1"] = pd.date_range("2020-02-24", periods=1000, freq="min")
    X["date2"] = pd.date_range("2021-09-29", periods=1000, freq="h")
    X["date3"] = ["2020-02-24"] * 1000
    return X


def test_numcat_raises_no_var_error(dfdt):
    # Case 5: error when no variable is numerical or categorical
    msg = "There are no numerical or categorical variables"
    with pytest.raises(TypeError, match=msg):
        find_categorical_and_numerical_variables(dfdt, None)
    msg = "The variable entered is neither numerical nor categorical."
    with pytest.raises(TypeError, match=msg):
        find_categorical_and_numerical_variables(dfdt, "date1")


def test_numcat_raises_no_var_warn(dfdt):
    # Case 6: warning when no variable is numerical or categorical
    msg = "There are no numerical or categorical variables"
    with pytest.warns(UserWarning, match=msg):
        find_categorical_and_numerical_variables(
            dfdt,
            None,
            return_empty=True,
        )
    msg = "The variable entered is neither numerical nor"
    with pytest.warns(UserWarning, match=msg):
        find_categorical_and_numerical_variables(
            dfdt, variables="date1", return_empty=True
        )


def test_numcat_returns_empty_lists(dfdt):
    assert find_categorical_and_numerical_variables(
        dfdt,
        None,
        return_empty=True,
    ) == ([], [])
    assert find_categorical_and_numerical_variables(
        dfdt,
        "date1",
        return_empty=True,
    ) == ([], [])


def test_numcat_on_user_empty_list(df_vartypes):
    # Case 7: user passes empty list
    with pytest.raises(ValueError):
        find_categorical_and_numerical_variables(df_vartypes, [])

    with pytest.warns(UserWarning):
        find_categorical_and_numerical_variables(df_vartypes, [], return_empty=True)

    assert find_categorical_and_numerical_variables(
        df_vartypes, [], return_empty=True
    ) == ([], [])


def test_numcat_when_dt_as_object(df_vartypes):
    # Case 8: datetime cast as object
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


def test_numcat_vars_as_category(df_vartypes):
    # Case 9: variables cast as category
    df = df_vartypes.copy()
    df["City"] = df["City"].astype("category")
    assert find_categorical_and_numerical_variables(df, None) == (
        ["Name", "City"],
        ["Age", "Marks"],
    )
    assert find_categorical_and_numerical_variables(df, "City") == (["City"], [])
