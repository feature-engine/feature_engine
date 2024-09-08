import pytest

from feature_engine.variable_handling import (
    find_all_variables,
    find_categorical_and_numerical_variables,
    find_categorical_variables,
    find_datetime_variables,
    find_numerical_variables,
)


def test_numerical_variables_finds_numerical_variables(df, df_int):
    assert find_numerical_variables(df) == ["Age", "Marks"]
    assert find_numerical_variables(df_int) == [3, 4]


def test_numerical_variables_raises_error_when_no_numerical_variables(df, df_int):
    msg = (
        "No numerical variables found in this dataframe. Please check "
        "variable format with pandas dtypes."
    )
    with pytest.raises(TypeError) as record:
        assert find_numerical_variables(df.drop(["Age", "Marks"], axis=1))
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert find_numerical_variables(df_int.drop([3, 4], axis=1))
    assert str(record.value) == msg


def test_categorical_variables_finds_categorical_variables(df, df_int):
    assert find_categorical_variables(df) == ["Name", "City"]
    assert find_categorical_variables(df_int) == [1, 2]


def test_categorical_variables_raises_error_when_no_categorical_variables(df, df_int):
    msg = (
        "No categorical variables found in this dataframe. Please check "
        "variable format with pandas dtypes."
    )
    with pytest.raises(TypeError) as record:
        assert find_categorical_variables(df.drop(["Name", "City"], axis=1))
    assert str(record.value) == msg

    with pytest.raises(TypeError) as record:
        assert find_categorical_variables(df_int.drop([1, 2], axis=1))
    assert str(record.value) == msg


def test_datetime_variables_finds_datetime_variables(df_datetime):
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


def test_datetime_variables_raises_error_when_no_datetime_variables(df_datetime):
    msg = "No datetime variables found in this dataframe."

    vars_nondt = ["Marks", "Age", "Name"]

    with pytest.raises(ValueError) as record:
        assert find_datetime_variables(df_datetime.loc[:, vars_nondt])
    assert str(record.value) == msg


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
    all_vars_no_dt = ["Name", "City", "Age", "Marks"]

    assert find_all_variables(df, exclude_datetime=False) == all_vars
    assert find_all_variables(df, exclude_datetime=True) == all_vars_no_dt


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
