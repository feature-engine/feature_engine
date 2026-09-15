"""Helpers for tests that run on several dataframe backends.
Use them together with the `make_df` fixture in `tests/conftest.py`.
"""

import narwhals as nw
import pandas as pd
import polars as pl


def frame_to_dict(X):
    """Return the dataframe contents as ``{column: list of values}``.

    pandas represents missing values as NaN and polars as None, so NaN (and
    pd.NA) are normalised to None and the same expected values work for both
    backends.
    """
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    return {
        col: [none_if_missing(v) for v in values] for col, values in result.items()
    }


def null_count(X, col):
    """Return the number of missing values in column ``col``."""
    return nw.from_native(X, eager_only=True).get_column(col).null_count()


def make_series(make_df, values, name=None):
    """Build a Series on the same backend as ``make_df``."""
    if make_df is pd.DataFrame:
        return pd.Series(values, name=name)
    return pl.Series(name=name or "", values=values)


def none_if_missing(value):
    if value is pd.NA or (isinstance(value, float) and value != value):
        return None
    return value
