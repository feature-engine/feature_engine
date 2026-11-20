import re

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from feature_engine.dataframe_checks import check_X
from feature_engine.datetime._datetime_helpers import (
    datetime_to_numpy,
    index_to_datetime_numpy,
)

DATES = ["2020-01-01 10:00:00", "2020-06-01 23:30:00", "1945-07-15 06:07:08"]
EXPECTED = np.array(DATES, dtype="datetime64[s]")


def _cast(X, expr):
    return nw.from_native(X).with_columns(expr).to_native()


def test_datetime_to_numpy_parses_strings_and_keeps_datetimes(make_df):
    X = _cast(make_df({"s": DATES, "d": DATES}), nw.col("d").str.to_datetime())
    arrays = datetime_to_numpy(check_X(X), ["s", "d"])
    assert len(arrays) == 2
    for values in arrays:
        assert values.dtype.kind == "M"
        np.testing.assert_array_equal(values.astype("datetime64[s]"), EXPECTED)


def test_datetime_to_numpy_parses_categorical_variables(make_df):
    X = _cast(make_df({"c": DATES}), nw.col("c").cast(nw.Categorical))
    (values,) = datetime_to_numpy(check_X(X), ["c"])
    np.testing.assert_array_equal(values.astype("datetime64[s]"), EXPECTED)


def test_datetime_to_numpy_with_format(make_df):
    X = make_df({"s": ["15/07/1945", "01/06/2020"]})
    (values,) = datetime_to_numpy(check_X(X), ["s"], format="%d/%m/%Y")
    np.testing.assert_array_equal(
        values.astype("datetime64[D]"),
        np.array(["1945-07-15", "2020-06-01"], dtype="datetime64[D]"),
    )


def test_datetime_to_numpy_returns_nat_for_missing_values(make_df):
    X = make_df({"s": [DATES[0], None, DATES[2]]})
    (values,) = datetime_to_numpy(check_X(X), ["s"])
    assert np.isnat(values).tolist() == [False, True, False]
    assert values[0].astype("datetime64[s]") == EXPECTED[0]


@pytest.mark.parametrize(
    "local_time, expected",
    [
        (True, ["2020-01-01T10:00:00", "2020-06-01T23:30:00"]),
        (False, ["2020-01-01T09:00:00", "2020-06-01T21:30:00"]),
    ],
)
def test_datetime_to_numpy_with_time_zones(make_df, local_time, expected):
    X = _cast(
        make_df({"d": DATES[:2]}),
        nw.col("d").str.to_datetime().dt.replace_time_zone("Europe/Madrid"),
    )
    (values,) = datetime_to_numpy(check_X(X), ["d"], local_time=local_time)
    np.testing.assert_array_equal(
        values.astype("datetime64[s]"), np.array(expected, dtype="datetime64[s]")
    )


def test_datetime_to_numpy_with_dateutil_fallback(make_df):
    X = make_df({"s": ["01-Jan-2010", None, "24-Feb-1945"]})
    (values,) = datetime_to_numpy(check_X(X), ["s"], dateutil_fallback=True)
    assert np.isnat(values).tolist() == [False, True, False]
    np.testing.assert_array_equal(
        values[[0, 2]].astype("datetime64[D]"),
        np.array(["2010-01-01", "1945-02-24"], dtype="datetime64[D]"),
    )


def test_datetime_to_numpy_without_fallback_raises_on_ambiguous_polars_strings():
    # pandas guesses the format of each string; polars needs a single format.
    X = check_X(pl.DataFrame({"s": ["01-Jan-2010", "24-Feb-1945"]}))
    with pytest.raises(pl.exceptions.ComputeError):
        datetime_to_numpy(X, ["s"])


def test_datetime_to_numpy_with_pandas_parsing_options():
    X = pd.DataFrame({"s": ["10/11/12", "01/02/09"]})
    (values,) = datetime_to_numpy(X, ["s"], dayfirst=True)
    np.testing.assert_array_equal(
        values.astype("datetime64[D]"),
        np.array(["2012-11-10", "2009-02-01"], dtype="datetime64[D]"),
    )


@pytest.mark.parametrize(
    "index",
    [
        pd.to_datetime(DATES),
        pd.Index(DATES),
        pd.DatetimeIndex(pd.to_datetime(DATES)).tz_localize("Europe/Madrid"),
    ],
)
def test_index_to_datetime_numpy(index):
    X = pd.DataFrame({"a": [1, 2, 3]}, index=index)
    values = index_to_datetime_numpy(X)
    np.testing.assert_array_equal(values.astype("datetime64[s]"), EXPECTED)


def test_index_to_datetime_numpy_returns_nat_for_missing_values():
    X = pd.DataFrame({"a": [1, 2, 3]}, index=[DATES[0], None, DATES[2]])
    values = index_to_datetime_numpy(X)
    assert np.isnat(values).tolist() == [False, True, False]


@pytest.mark.parametrize("index", [[1, 2, 3], ["a", "b", "c"]])
def test_index_to_datetime_numpy_raises_error_if_index_not_datetime(index):
    X = pd.DataFrame({"a": [1, 2, 3]}, index=index)
    with pytest.raises(
        TypeError, match=re.escape("The dataframe index is not datetime.")
    ):
        index_to_datetime_numpy(X)


def test_index_to_datetime_numpy_raises_error_if_not_pandas():
    X = check_X(pl.DataFrame({"a": [1, 2, 3]}))
    msg = (
        "variables='index' requires a pandas dataframe, since only pandas "
        f"dataframes have an index. Got {pl.DataFrame} instead."
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        index_to_datetime_numpy(X)
