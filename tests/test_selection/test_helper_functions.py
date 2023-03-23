import numpy as np
import pandas as pd
from feature_engine.selection._helper_functions import _sort_variables

test_df = pd.DataFrame(
    {
        "z_var": [0, 0, 0, 3, 4, 5, 6, 6, 6, 9, np.nan],
        "a_var": [0, 1, 2, 3, 4, 5, 6, 7, 8, np.nan, np.nan],
        "c_var": [1, 1, 1, 1, 1, 1, 1, 1, np.nan, np.nan, np.nan],
        "b_var": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
)
var_list = test_df.columns.to_list()


def test_order_alphabetically():
    ordered_vars = _sort_variables(test_df, var_list, order_by="alphabetic")
    expected = ["a_var", "b_var", "c_var", "z_var"]
    assert ordered_vars == expected


def test_order_None():
    ordered_vars = _sort_variables(test_df, var_list, order_by=None)
    assert ordered_vars == var_list


def test_order_nan():
    ordered_vars = _sort_variables(test_df, var_list, order_by="nan")
    expected = ["b_var", "z_var", "a_var", "c_var"]
    assert ordered_vars == expected


def test_order_unique():
    ordered_vars = _sort_variables(test_df, var_list, order_by="unique")
    expected = ["b_var", "a_var", "z_var", "c_var"]
    assert ordered_vars == expected
