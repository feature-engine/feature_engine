import pandas as pd
import pytest

from feature_engine.variable_handling.retain_variables import retain_variables_if_in_df

test_dict = [
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


@pytest.mark.parametrize("df, variables, overlap, col_not_in_df", test_dict)
def test_retain_variables_if_in_df(df, variables, overlap, col_not_in_df):

    msg = "None of the variables in the list are present in the dataframe."

    assert retain_variables_if_in_df(df, variables) == overlap

    with pytest.raises(ValueError) as record:
        retain_variables_if_in_df(df, col_not_in_df)
    assert str(record.value) == msg
