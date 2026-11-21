import re

import pandas as pd
import polars as pl
import pytest

from feature_engine.dataframe_checks import check_X
from feature_engine.timeseries.forecasting._forecasting_helpers import (
    check_index,
    sort_by_index,
)
from tests.backend_helpers import frame_to_dict

DATES = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])
MSG_INDEX_NA = (
    "The dataframe's index contains NaN values or missing data. Only dataframes "
    "with complete indexes are compatible with this transformer."
)
MSG_INDEX_DUPLICATED = (
    "The dataframe's index does not contain unique values. Only dataframes with "
    "unique values in the index are compatible with this transformer."
)


@pytest.mark.parametrize("freq", [None, "1D"])
def test_check_index_accepts_complete_and_unique_pandas_index(freq):
    check_index(check_X(pd.DataFrame({"a": [1, 2, 3]}, index=DATES)), freq)


@pytest.mark.parametrize(
    "index, msg",
    [
        (list(DATES[:2]) + [pd.NaT], MSG_INDEX_NA),
        (list(DATES[:2]) + [DATES[0]], MSG_INDEX_DUPLICATED),
    ],
)
def test_check_index_raises_error_if_pandas_index_incomplete_or_duplicated(
    index, msg
):
    X = check_X(pd.DataFrame({"a": [1, 2, 3]}, index=index))
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        check_index(X)


def test_check_index_raises_error_if_freq_with_polars():
    X = check_X(pl.DataFrame({"a": [1, 2, 3]}))
    check_index(X)

    msg = (
        "freq is only supported with pandas dataframes, because it uses the "
        "dataframe's DatetimeIndex. With other dataframes, leave freq=None "
        "to shift the rows by periods. Got 1D instead."
    )
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        check_index(X, "1D")


def test_sort_by_index_sorts_pandas_rows_by_the_index():
    X = pd.DataFrame({"a": [3, 1, 2]}, index=DATES)
    Xt = sort_by_index(check_X(X))

    expected = pd.DataFrame({"a": [1, 2, 3]}, index=DATES.sort_values())
    pd.testing.assert_frame_equal(Xt, expected)
    assert X["a"].tolist() == [3, 1, 2]


def test_sort_by_index_keeps_the_row_order_with_polars():
    X = check_X(pl.DataFrame({"a": [3, 1, 2]}))
    assert frame_to_dict(sort_by_index(X).to_native()) == {"a": [3, 1, 2]}
