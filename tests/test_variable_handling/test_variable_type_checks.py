from datetime import date

import narwhals as nw
import pandas as pd
import polars as pl

from feature_engine.variable_handling._variable_type_checks import (
    _is_categorical_and_is_datetime,
    _is_categorical_and_is_not_datetime,
    _is_categories_num,
    _is_convertible_to_dt,
    _is_convertible_to_num,
    _is_date_or_datetime,
    _looks_like_date_string,
)


def nw_series(values, dtype=None):
    s = pl.Series("x", values)
    if dtype is not None:
        s = s.cast(dtype)
    return nw.from_native(s, series_only=True)


def nw_pandas_series(values, dtype=None):
    s = pd.Series(values, dtype=dtype)
    return nw.from_native(s, series_only=True)


def test_is_date_or_datetime():
    """A dtype is a date or datetime if it is narwhals' Date or Datetime type."""
    assert _is_date_or_datetime(nw_series([date(2020, 1, 1)]).dtype) is True
    assert (
        _is_date_or_datetime(nw_series(["2020-01-01"]).str.to_datetime().dtype)
        is True
    )
    assert _is_date_or_datetime(nw_series(["a", "b"]).dtype) is False
    assert _is_date_or_datetime(nw_series([1, 2, 3]).dtype) is False


def test_looks_like_date_string():
    """A string looks like a date if dateutil finds at least 2 date/time fields
    in it - this rejects bare numbers that dateutil would otherwise happily
    "parse" as a single field (e.g. a day), while still accepting real dates
    in non-ISO formats and bare times.
    """
    # real dates, including non-ISO formats
    assert _looks_like_date_string("2020-01-01") is True
    assert _looks_like_date_string("01-Jan-2010") is True
    assert _looks_like_date_string("10/11/12") is True

    # bare times
    assert _looks_like_date_string("21:45:23") is True
    assert _looks_like_date_string("08:00") is True

    # partial dates
    assert _looks_like_date_string("Jan 2020") is True

    # bare numbers dateutil could misparse as a single date/time field
    assert _looks_like_date_string("20") is False
    assert _looks_like_date_string("1999") is False
    assert _looks_like_date_string("12") is False

    # non-date garbage
    assert _looks_like_date_string("hello") is False
    assert _looks_like_date_string("") is False

    # non-string values (e.g. from a mixed-type pandas Object column) must not
    # raise, they simply aren't date strings
    assert _looks_like_date_string(20) is False
    assert _looks_like_date_string(1.5) is False
    assert _looks_like_date_string(None) is False


def test_is_convertible_to_num():
    """A series is convertible to numeric if every non-null value can be cast
    to float.
    """
    assert _is_convertible_to_num(nw_series(["20", "21", "19"])) is True
    assert _is_convertible_to_num(nw_series(["a", "b"])) is False
    assert (
        _is_convertible_to_num(nw_series(["20", "21"], dtype=pl.Categorical))
        is True
    )

    # object dtype columns (pandas-only concept - narwhals classifies a plain
    # object dtype column of ints as `nw.Object`, not `nw.String`)
    assert _is_convertible_to_num(nw_pandas_series([1, 2], dtype="object")) is True
    assert (
        _is_convertible_to_num(
            nw_pandas_series([pd.Timestamp("2020-01-01")], dtype="object")
        )
        is False
    )


def test_is_convertible_to_dt():
    """A series is convertible to datetime if every non-null value is either a
    real date/datetime object, or a string that looks like a date.
    """
    assert _is_convertible_to_dt(nw_series(["2020-01-01", "2020-01-02"])) is True
    assert _is_convertible_to_dt(nw_series(["a", "b"])) is False
    assert _is_convertible_to_dt(nw_series(["20", "21"])) is False

    # flexible, dateutil-backed date guessing works for every backend now, not
    # just pandas - so non-ISO formats are recognised here too
    assert _is_convertible_to_dt(nw_series(["01-Jan-2010"])) is True
    assert _is_convertible_to_dt(nw_series(["10/11/12"])) is True

    # an object dtype column holding actual datetime objects (e.g. pandas
    # Timestamps) is trivially convertible, without needing to parse anything
    assert (
        _is_convertible_to_dt(
            nw_pandas_series([pd.Timestamp("2020-01-01")], dtype="object")
        )
        is True
    )


def test_is_categories_num():
    """A categorical series' categories are numeric if their dtype is numeric -
    only possible for pandas, since polars categories are always string-backed.
    """
    non_numeric_cat = nw_series(["a", "b", "c"], dtype=pl.Categorical)
    assert _is_categories_num(non_numeric_cat) is False

    numeric_cat = nw_pandas_series([20, 21, 19, 18], dtype="category")
    assert _is_categories_num(numeric_cat) is True


def test_is_categorical_and_is_datetime():
    """A series is categorical-and-datetime if it is a Categorical/String/Object
    column whose values are dates, but not an Enum (an explicit category set is
    never treated as a datetime) or a numeric-backed categorical.
    """
    assert (
        _is_categorical_and_is_datetime(
            nw_series(["2020-01-01", "2020-01-02"], dtype=pl.Categorical)
        )
        is True
    )
    assert (
        _is_categorical_and_is_datetime(nw_series(["a", "b"], dtype=pl.Categorical))
        is False
    )
    assert _is_categorical_and_is_datetime(nw_series(["2020-01-01"])) is True
    assert _is_categorical_and_is_datetime(nw_series(["20", "21"])) is False
    assert _is_categorical_and_is_datetime(nw_series(["a", "b"])) is False

    # an explicit Enum is always treated as categorical, never as datetime
    enum_dtype = pl.Enum(["2020-01-01", "2020-01-02"])
    assert (
        _is_categorical_and_is_datetime(
            nw_series(["2020-01-01", "2020-01-02"], dtype=enum_dtype)
        )
        is False
    )

    # numeric should be False
    assert _is_categorical_and_is_datetime(nw_series([1, 2, 3])) is False

    # a numeric-backed categorical (pandas-only - polars categories are always
    # string-backed) can never be a datetime, regardless of the categories
    numeric_cat = nw_pandas_series([20, 21, 19, 18], dtype="category")
    assert _is_categorical_and_is_datetime(numeric_cat) is False

    # a string-dtype pandas column with datetime-like values
    assert (
        _is_categorical_and_is_datetime(
            nw_pandas_series(["2020-01-01", "2020-01-02"], dtype="string")
        )
        is True
    )

    # object dtype column holding actual Timestamp objects
    assert (
        _is_categorical_and_is_datetime(
            nw_pandas_series([pd.Timestamp("2020-01-01")], dtype="object")
        )
        is True
    )

    # object dtype column holding plain ints - not a datetime
    assert (
        _is_categorical_and_is_datetime(nw_pandas_series([1, 2], dtype="object"))
        is False
    )


def test_is_categorical_and_is_not_datetime():
    """A series is categorical-and-not-datetime if it is a Categorical/String/
    Object/Enum column whose values are not dates.
    """
    assert (
        _is_categorical_and_is_not_datetime(
            nw_series(["2020-01-01", "2020-01-02"], dtype=pl.Categorical)
        )
        is False
    )
    assert (
        _is_categorical_and_is_not_datetime(
            nw_series(["a", "b"], dtype=pl.Categorical)
        )
        is True
    )
    assert _is_categorical_and_is_not_datetime(nw_series(["2020-01-01"])) is False
    assert _is_categorical_and_is_not_datetime(nw_series(["20", "21"])) is True
    assert _is_categorical_and_is_not_datetime(nw_series(["a", "b"])) is True

    # an explicit Enum is always treated as categorical
    assert (
        _is_categorical_and_is_not_datetime(
            nw_series(["a", "b"], dtype=pl.Enum(["a", "b"]))
        )
        is True
    )

    # numeric should be False
    assert _is_categorical_and_is_not_datetime(nw_series([1, 2, 3])) is False

    # a numeric-backed categorical is categorical-and-not-datetime
    numeric_cat = nw_pandas_series([20, 21, 19, 18], dtype="category")
    assert _is_categorical_and_is_not_datetime(numeric_cat) is True

    # object dtype column of plain ints
    assert (
        _is_categorical_and_is_not_datetime(nw_pandas_series([1, 2], dtype="object"))
        is True
    )

    # object dtype column holding actual Timestamp objects - is a datetime, so
    # not "categorical and not datetime"
    assert (
        _is_categorical_and_is_not_datetime(
            nw_pandas_series([pd.Timestamp("2020-01-01")], dtype="object")
        )
        is False
    )

    # string-dtype pandas column not convertible to numeric or datetime
    assert (
        _is_categorical_and_is_not_datetime(
            nw_pandas_series(["a", "b"], dtype="string")
        )
        is True
    )
