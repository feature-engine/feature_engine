from datetime import date

import narwhals as nw
import pandas as pd
import polars as pl

from feature_engine.variable_handling._variable_type_checks import (
    _nw_is_categorical_and_is_datetime,
    _nw_is_categorical_and_is_not_datetime,
    _nw_is_convertible_to_dt,
    _nw_is_convertible_to_num,
    _nw_is_date_or_datetime,
)


def nw_series(values, dtype=None):
    s = pl.Series("x", values)
    if dtype is not None:
        s = s.cast(dtype)
    return nw.from_native(s, series_only=True)


def nw_pandas_series(values, dtype=None):
    s = pd.Series(values, dtype=dtype)
    return nw.from_native(s, series_only=True)


def test_nw_is_date_or_datetime():
    assert _nw_is_date_or_datetime(nw_series([date(2020, 1, 1)]).dtype) is True
    assert (
        _nw_is_date_or_datetime(nw_series(["2020-01-01"]).str.to_datetime().dtype)
        is True
    )
    assert _nw_is_date_or_datetime(nw_series(["a", "b"]).dtype) is False
    assert _nw_is_date_or_datetime(nw_series([1, 2, 3]).dtype) is False


def test_nw_is_convertible_to_num():
    assert _nw_is_convertible_to_num(nw_series(["20", "21", "19"])) is True
    assert _nw_is_convertible_to_num(nw_series(["a", "b"])) is False
    assert (
        _nw_is_convertible_to_num(nw_series(["20", "21"], dtype=pl.Categorical))
        is True
    )

    # object dtype columns (pandas-only concept - narwhals classifies a plain
    # object dtype column of ints as `nw.Object`, not `nw.String`)
    assert _nw_is_convertible_to_num(nw_pandas_series([1, 2], dtype="object")) is True
    assert (
        _nw_is_convertible_to_num(
            nw_pandas_series([pd.Timestamp("2020-01-01")], dtype="object")
        )
        is False
    )


def test_nw_is_convertible_to_dt():
    assert _nw_is_convertible_to_dt(nw_series(["2020-01-01", "2020-01-02"])) is True
    assert _nw_is_convertible_to_dt(nw_series(["a", "b"])) is False
    assert _nw_is_convertible_to_dt(nw_series(["20", "21"])) is False

    # flexible, dateutil-backed date guessing works for every backend now, not
    # just pandas - so non-ISO formats are recognised here too
    assert _nw_is_convertible_to_dt(nw_series(["01-Jan-2010"])) is True
    assert _nw_is_convertible_to_dt(nw_series(["10/11/12"])) is True

    # an object dtype column holding actual datetime objects (e.g. pandas
    # Timestamps) is trivially convertible, without needing to parse anything
    assert (
        _nw_is_convertible_to_dt(
            nw_pandas_series([pd.Timestamp("2020-01-01")], dtype="object")
        )
        is True
    )


def test_nw_is_categorical_and_is_datetime():
    assert (
        _nw_is_categorical_and_is_datetime(
            nw_series(["2020-01-01", "2020-01-02"], dtype=pl.Categorical)
        )
        is True
    )
    assert (
        _nw_is_categorical_and_is_datetime(nw_series(["a", "b"], dtype=pl.Categorical))
        is False
    )
    assert _nw_is_categorical_and_is_datetime(nw_series(["2020-01-01"])) is True
    assert _nw_is_categorical_and_is_datetime(nw_series(["20", "21"])) is False
    assert _nw_is_categorical_and_is_datetime(nw_series(["a", "b"])) is False

    # an explicit Enum is always treated as categorical, never as datetime
    enum_dtype = pl.Enum(["2020-01-01", "2020-01-02"])
    assert (
        _nw_is_categorical_and_is_datetime(
            nw_series(["2020-01-01", "2020-01-02"], dtype=enum_dtype)
        )
        is False
    )

    # numeric should be False
    assert _nw_is_categorical_and_is_datetime(nw_series([1, 2, 3])) is False

    # a numeric-backed categorical (pandas-only - polars categories are always
    # string-backed) can never be a datetime, regardless of the categories
    numeric_cat = nw_pandas_series([20, 21, 19, 18], dtype="category")
    assert _nw_is_categorical_and_is_datetime(numeric_cat) is False

    # a string-dtype pandas column with datetime-like values
    assert (
        _nw_is_categorical_and_is_datetime(
            nw_pandas_series(["2020-01-01", "2020-01-02"], dtype="string")
        )
        is True
    )

    # object dtype column holding actual Timestamp objects
    assert (
        _nw_is_categorical_and_is_datetime(
            nw_pandas_series([pd.Timestamp("2020-01-01")], dtype="object")
        )
        is True
    )

    # object dtype column holding plain ints - not a datetime
    assert (
        _nw_is_categorical_and_is_datetime(nw_pandas_series([1, 2], dtype="object"))
        is False
    )


def test_nw_is_categorical_and_is_not_datetime():
    assert (
        _nw_is_categorical_and_is_not_datetime(
            nw_series(["2020-01-01", "2020-01-02"], dtype=pl.Categorical)
        )
        is False
    )
    assert (
        _nw_is_categorical_and_is_not_datetime(
            nw_series(["a", "b"], dtype=pl.Categorical)
        )
        is True
    )
    assert _nw_is_categorical_and_is_not_datetime(nw_series(["2020-01-01"])) is False
    assert _nw_is_categorical_and_is_not_datetime(nw_series(["20", "21"])) is True
    assert _nw_is_categorical_and_is_not_datetime(nw_series(["a", "b"])) is True

    # an explicit Enum is always treated as categorical
    assert (
        _nw_is_categorical_and_is_not_datetime(
            nw_series(["a", "b"], dtype=pl.Enum(["a", "b"]))
        )
        is True
    )

    # numeric should be False
    assert _nw_is_categorical_and_is_not_datetime(nw_series([1, 2, 3])) is False

    # a numeric-backed categorical is categorical-and-not-datetime
    numeric_cat = nw_pandas_series([20, 21, 19, 18], dtype="category")
    assert _nw_is_categorical_and_is_not_datetime(numeric_cat) is True

    # object dtype column of plain ints
    assert (
        _nw_is_categorical_and_is_not_datetime(nw_pandas_series([1, 2], dtype="object"))
        is True
    )

    # object dtype column holding actual Timestamp objects - is a datetime, so
    # not "categorical and not datetime"
    assert (
        _nw_is_categorical_and_is_not_datetime(
            nw_pandas_series([pd.Timestamp("2020-01-01")], dtype="object")
        )
        is False
    )

    # string-dtype pandas column not convertible to numeric or datetime
    assert (
        _nw_is_categorical_and_is_not_datetime(
            nw_pandas_series(["a", "b"], dtype="string")
        )
        is True
    )
