import pandas as pd
import polars as pl
import pytest

from feature_engine.variable_handling.retain_variables import retain_variables_if_in_df

test_dict = [
    (["A", "C", "B", "G", "H"], ["A", "C", "B"], ["X", "Y"]),
    ("C", ["C"], "G"),
]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("variables, overlap, col_not_in_df", test_dict)
def test_retain_variables_if_in_df(make_df, variables, overlap, col_not_in_df):
    df = make_df({"A": [1], "B": [1], "C": [1], "D": [1], "E": [1]})

    msg = "None of the variables in the list are present in the dataframe."

    assert retain_variables_if_in_df(df, variables) == overlap

    with pytest.raises(ValueError, match=msg):
        retain_variables_if_in_df(df, col_not_in_df)


def test_retain_variables_if_in_df_int_column_names():
    # polars requires string column names. int-named columns are pandas-only
    df = pd.DataFrame({1: [1], 2: [1], 3: [1], 4: [1], 5: [1]})

    msg = "None of the variables in the list are present in the dataframe."

    assert retain_variables_if_in_df(df, [1, 2, 4, 6]) == [1, 2, 4]
    assert retain_variables_if_in_df(df, 1) == [1]

    with pytest.raises(ValueError, match=msg):
        retain_variables_if_in_df(df, [6, 7])

    with pytest.raises(ValueError, match=msg):
        retain_variables_if_in_df(df, 7)
