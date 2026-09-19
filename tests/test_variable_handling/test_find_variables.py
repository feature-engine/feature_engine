import pandas as pd
import polars as pl
import pytest

from feature_engine.variable_handling import (
    find_all_variables,
    find_categorical_and_numerical_variables,
    find_categorical_variables,
    find_datetime_variables,
    find_numerical_variables,
)
from tests.test_variable_handling.conftest import (
    BASIC_DATA,
    DATETIME_DATA,
    cast_categorical,
)

# --- find_numerical_variables --- #


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_numerical_variables_finds_variables(make_df):
    df = make_df(BASIC_DATA)
    assert find_numerical_variables(df) == ["Age", "Marks"]


def test_numerical_variables_finds_variables_with_int_column_names(df_int):
    # polars requires string column names. int-named columns are pandas-only
    assert find_numerical_variables(df_int) == [3, 4]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_numerical_variables_raises_error(make_df):
    df = make_df(BASIC_DATA)
    msg = "No numerical variables found in this dataframe."
    with pytest.raises(TypeError, match=msg):
        find_numerical_variables(df[["Name", "City"]])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_numerical_variables_raises_warning(make_df):
    df = make_df(BASIC_DATA)
    msg = "No numerical variables found in this dataframe."
    with pytest.warns(UserWarning, match=msg):
        find_numerical_variables(df[["Name", "City"]], return_empty=True)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_numerical_variables_returns_empty_list(make_df):
    df = make_df(BASIC_DATA)
    assert find_numerical_variables(df[["Name", "City"]], return_empty=True) == []


# --- find_categorical_variables --- #


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_categorical_variables_finds_variables(make_df):
    df = make_df(BASIC_DATA)
    assert find_categorical_variables(df) == ["Name", "City"]


def test_categorical_variables_finds_variables_with_int_column_names(df_int):
    assert find_categorical_variables(df_int) == [1, 2]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_categorical_variables_raises_error(make_df):
    df = make_df(BASIC_DATA)
    msg = "No categorical variables found in this dataframe."
    with pytest.raises(TypeError, match=msg):
        find_categorical_variables(df[["Age", "Marks"]])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_categorical_variables_raises_warning(make_df):
    df = make_df(BASIC_DATA)
    msg = "No categorical variables found in this dataframe."
    with pytest.warns(UserWarning, match=msg):
        find_categorical_variables(df[["Age", "Marks"]], return_empty=True)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_categorical_variables_returns_empty_list(make_df):
    df = make_df(BASIC_DATA)
    assert find_categorical_variables(df[["Age", "Marks"]], return_empty=True) == []


# --- find_datetime_variables --- #


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_variables_finds_variables(make_df):
    df = make_df(DATETIME_DATA)
    vars_dt = ["date_range", "date_obj0", "date_range_tz"]
    assert find_datetime_variables(df) == vars_dt

    assert find_datetime_variables(
        df[["date_obj0", "date_range", "date_range_tz"]],
    ) == ["date_obj0", "date_range", "date_range_tz"]


def test_datetime_variables_finds_pandas_only_string_formats(df_datetime):
    # "01-Jan-2010"-style, "10/11/12"-style and bare-time strings are
    # recognised through flexible, dateutil-backed guessing.
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


def test_datetime_variables_finds_flexible_string_formats_in_polars_too():
    # flexible, dateutil-backed date guessing is backend-agnostic, so polars
    # now also recognises non-ISO formats it previously could not.
    df = pl.DataFrame(
        {
            "var_num": [1, 2, 3],
            "date_obj1": ["01-Jan-2010", "24-Feb-1945", "14-Jun-2100"],
            "date_obj2": ["10/11/12", "12/31/09", "06/30/95"],
        }
    )
    assert find_datetime_variables(df) == ["date_obj1", "date_obj2"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_variables_raises_error(make_df):
    df = make_df(DATETIME_DATA)
    msg = "No datetime variables found in this dataframe."
    vars_nondt = ["Marks", "Age", "Name"]
    with pytest.raises(TypeError, match=msg):
        find_datetime_variables(df[vars_nondt])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_variables_raises_warning(make_df):
    df = make_df(DATETIME_DATA)
    msg = "No datetime variables found in this dataframe."
    vars_nondt = ["Marks", "Age", "Name"]
    with pytest.warns(UserWarning, match=msg):
        find_datetime_variables(df[vars_nondt], return_empty=True)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_datetime_variables_returns_empty_list(make_df):
    df = make_df(DATETIME_DATA)
    vars_nondt = ["Marks", "Age", "Name"]
    assert find_datetime_variables(df[vars_nondt], return_empty=True) == []


# --- find_all_variables --- #


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_find_all_variables(make_df):
    df = make_df(BASIC_DATA)
    assert find_all_variables(df, exclude_datetime=False) == list(BASIC_DATA.keys())


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_find_all_variables_excludes_dt(make_df):
    df = make_df(DATETIME_DATA)
    all_vars_no_dt = ["Name", "City", "Age", "Marks"]
    assert find_all_variables(df, exclude_datetime=True) == all_vars_no_dt


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_find_all_variables_raises_error(make_df):
    dt_vars = ["date_range", "date_obj0", "date_range_tz"]
    df = make_df(DATETIME_DATA)[dt_vars]
    msg = "No variables found in this dataframe"
    with pytest.raises(TypeError, match=msg):
        find_all_variables(df, exclude_datetime=True)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_find_all_variables_raises_warning(make_df):
    dt_vars = ["date_range", "date_obj0", "date_range_tz"]
    df = make_df(DATETIME_DATA)[dt_vars]
    msg = "No variables found in this dataframe"
    with pytest.warns(UserWarning, match=msg):
        find_all_variables(df, exclude_datetime=True, return_empty=True)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_find_all_variables_returns_empty(make_df):
    dt_vars = ["date_range", "date_obj0", "date_range_tz"]
    df = make_df(DATETIME_DATA)[dt_vars]
    assert find_all_variables(df, exclude_datetime=True, return_empty=True) == []


# --- find_categorical_and_numerical_variables --- #


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_numcat_user_passes_varlist(make_df):
    df = make_df(BASIC_DATA)

    # Case 1: user passes 1 variable that is categorical
    assert find_categorical_and_numerical_variables(df, ["Name"]) == (["Name"], [])
    assert find_categorical_and_numerical_variables(df, "Name") == (["Name"], [])

    # Case 2: user passes 1 variable that is numerical
    assert find_categorical_and_numerical_variables(df, ["Age"]) == ([], ["Age"])
    assert find_categorical_and_numerical_variables(df, "Age") == ([], ["Age"])

    # Case 3: user passes 1 categorical and 1 numerical variable
    assert find_categorical_and_numerical_variables(df, ["Age", "Name"]) == (
        ["Name"],
        ["Age"],
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_numcat_when_var_is_none(make_df):
    df = make_df(BASIC_DATA)

    assert find_categorical_and_numerical_variables(df, None) == (
        ["Name", "City"],
        ["Age", "Marks"],
    )
    assert find_categorical_and_numerical_variables(df[["Name", "City"]], None) == (
        ["Name", "City"],
        [],
    )
    assert find_categorical_and_numerical_variables(df[["Age", "Marks"]], None) == (
        [],
        ["Age", "Marks"],
    )


@pytest.mark.parametrize(
    "make_df, assert_error", [(pd.DataFrame, TypeError), (pl.DataFrame, TypeError)]
)
def test_numcat_raises_no_var_error(make_df, assert_error):
    # Case 5: error when no variable is numerical or categorical
    df = make_df(
        {
            "date1": DATETIME_DATA["date_range"],
            "date2": DATETIME_DATA["date_range_tz"],
        }
    )
    msg = "There are no numerical or categorical variables"
    with pytest.raises(assert_error, match=msg):
        find_categorical_and_numerical_variables(df, None)
    msg = "The variable entered is neither numerical nor categorical."
    with pytest.raises(assert_error, match=msg):
        find_categorical_and_numerical_variables(df, "date1")


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_numcat_raises_no_var_warn(make_df):
    df = make_df(
        {
            "date1": DATETIME_DATA["date_range"],
            "date2": DATETIME_DATA["date_range_tz"],
        }
    )
    msg = "There are no numerical or categorical variables"
    with pytest.warns(UserWarning, match=msg):
        find_categorical_and_numerical_variables(df, None, return_empty=True)
    msg = "The variable entered is neither numerical nor"
    with pytest.warns(UserWarning, match=msg):
        find_categorical_and_numerical_variables(
            df, variables="date1", return_empty=True
        )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_numcat_returns_empty_lists(make_df):
    df = make_df(
        {
            "date1": DATETIME_DATA["date_range"],
            "date2": DATETIME_DATA["date_range_tz"],
        }
    )
    assert find_categorical_and_numerical_variables(
        df, None, return_empty=True
    ) == ([], [])
    assert find_categorical_and_numerical_variables(
        df, "date1", return_empty=True
    ) == ([], [])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_numcat_on_user_empty_list(make_df):
    df = make_df(BASIC_DATA)

    msg = "The list of variables provided is empty. If this was"
    with pytest.raises(ValueError, match=msg):
        find_categorical_and_numerical_variables(df, [])

    msg = "The list of variables provided is empty. Returning "
    with pytest.warns(UserWarning, match=msg):
        find_categorical_and_numerical_variables(df, [], return_empty=True)

    assert find_categorical_and_numerical_variables(df, [], return_empty=True) == (
        [],
        [],
    )


def test_numcat_when_dt_as_object(df_vartypes):
    # Case 8: datetime cast as object - pandas-only, `df_vartypes["dob"]` is a
    # pandas datetime64 column relying on pandas' `.astype("O")`, which has no
    # polars equivalent (polars has no generic object dtype to cast into).
    df = df_vartypes.copy()
    df["dob"] = df["dob"].astype("O")

    assert find_categorical_and_numerical_variables(df, None) == (
        ["Name", "City"],
        ["Age", "Marks"],
    )
    assert find_categorical_and_numerical_variables(df, ["Name", "Marks", "dob"]) == (
        ["Name"],
        ["Marks"],
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_numcat_vars_as_category(make_df):
    # Case 9: variables cast as category
    df = make_df(BASIC_DATA)
    df = cast_categorical(df, ["City"])
    assert find_categorical_and_numerical_variables(df, None) == (
        ["Name", "City"],
        ["Age", "Marks"],
    )
    assert find_categorical_and_numerical_variables(df, "City") == (["City"], [])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_numcat_agrees_with_find_categorical_on_date_like_category(make_df):
    # Regression test: the single-variable path used to disagree with
    # find_categorical_variables on a date-like category column.
    df = make_df({"date_cat": DATETIME_DATA["date_obj0"], "num": BASIC_DATA["Age"]})
    df = cast_categorical(df, ["date_cat"])

    assert find_categorical_variables(df, return_empty=True) == []
    assert find_categorical_and_numerical_variables(df, None) == ([], ["num"])
    assert find_categorical_and_numerical_variables(
        df, "date_cat", return_empty=True
    ) == ([], [])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_numcat_exclude_datetime_false_keeps_date_like_category(make_df):
    # exclude_datetime=False must be honoured consistently across all three
    # entry points, including the single-variable branch.
    df = make_df({"date_cat": DATETIME_DATA["date_obj0"], "num": BASIC_DATA["Age"]})
    df = cast_categorical(df, ["date_cat"])

    assert find_categorical_variables(df, exclude_datetime=False) == ["date_cat"]
    assert find_categorical_and_numerical_variables(
        df, None, exclude_datetime=False
    ) == (["date_cat"], ["num"])
    assert find_categorical_and_numerical_variables(
        df, "date_cat", exclude_datetime=False
    ) == (["date_cat"], [])
