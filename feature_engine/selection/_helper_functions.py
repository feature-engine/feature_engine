from typing import List
import pandas as pd


def _sort_variables(X: pd.DataFrame, variables: List, order_by: str):
    """Helper function for sorting columns."""
    if order_by == "nan":
        ordered_vars = list(X[variables].isna().sum(0).sort_values().index)
    elif order_by == "unique":
        ordered_vars = list(X[variables].nunique(0).sort_values(ascending=False).index)
    elif order_by == "alphabetic":
        ordered_vars = list(X[variables].sort_index(axis=1).columns)
    else:
        ordered_vars = variables
    return ordered_vars
