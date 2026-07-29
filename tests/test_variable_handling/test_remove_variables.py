import pandas as pd
import polars as pl
import pytest

from feature_engine.variable_handling.retain_variables import retain_variables_if_in_df


def make_empty_df(is_pandas: bool, columns):
    if is_pandas:
        return pd.DataFrame(columns=columns)
    return pl.DataFrame(schema=columns)


test_dict = [
    (["A", "C", "B", "G", "H"], ["A", "C", "B"], ["X", "Y"]),
    ("C", ["C"], "G"),
]


@pytest.mark.parametrize("is_pandas", [True, False])
@pytest.mark.parametrize("variables, overlap, col_not_in_df", test_dict)
def test_retain_variables_if_in_df(is_pandas, variables, overlap, col_not_in_df):
    df = make_empty_df(is_pandas, ["A", "B", "C", "D", "E"])

    msg = "None of the variables in the list are present in the dataframe."

    assert retain_variables_if_in_df(df, variables) == overlap

    with pytest.raises(ValueError) as record:
        retain_variables_if_in_df(df, col_not_in_df)
    assert str(record.value) == msg


def test_retain_variables_if_in_df_int_column_names():
    # polars requires string column names, so int-named columns are pandas-only
    df = pd.DataFrame(columns=[1, 2, 3, 4, 5])

    msg = "None of the variables in the list are present in the dataframe."

    assert retain_variables_if_in_df(df, [1, 2, 4, 6]) == [1, 2, 4]
    assert retain_variables_if_in_df(df, 1) == [1]

    with pytest.raises(ValueError) as record:
        retain_variables_if_in_df(df, [6, 7])
    assert str(record.value) == msg

    with pytest.raises(ValueError) as record:
        retain_variables_if_in_df(df, 7)
    assert str(record.value) == msg
