import pandas as pd
import pytest

from feature_engine.variable_handling.variable_selection import (
    _filter_out_variables_not_in_dataframe,
    find_categorical_and_numerical_variables,
)

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
