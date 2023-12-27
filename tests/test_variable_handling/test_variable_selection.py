import pandas as pd
import pytest

from feature_engine.variable_handling.variable_selection import (
    _filter_out_variables_not_in_dataframe,
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
