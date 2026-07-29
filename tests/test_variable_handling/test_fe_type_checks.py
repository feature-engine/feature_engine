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


def test_is_categories_num(df):
    assert _is_categories_num(df["Name"]) is False

    df["Age"] = df["Age"].astype("category")
    assert _is_categories_num(df["Age"]) is True


def test_is_convertible_to_num(df):
    assert _is_convertible_to_num(df["Name"]) is False
    assert _is_convertible_to_num(df["date_obj0"]) is False

    df["age_str"] = ["20", "21", "19", "18"]
    assert _is_convertible_to_num(df["age_str"]) is True


def test_is_convertible_to_dt(df):
    assert _is_convertible_to_dt(df["date_obj0"]) is True
    assert _is_convertible_to_dt(df["date_range"]) is True
    assert _is_convertible_to_dt(df["Name"]) is False

    df["age_str"] = ["20", "21", "19", "18"]
    assert _is_convertible_to_dt(df["age_str"]) is False


def test_is_categorical_and_is_datetime(df, df_datetime):
    assert _is_categorical_and_is_datetime(df["date_obj0"]) is True
    assert _is_categorical_and_is_datetime(df["Name"]) is False
    assert _is_categorical_and_is_datetime(df_datetime["date_obj1"]) is True

    df["age_str"] = ["20", "21", "19", "18"]
    assert _is_categorical_and_is_datetime(df["age_str"]) is False

    df = df.copy()
    # from pandas 3 onwards, object types that contain strings are not recognised as
    # objects any more
    df["Age"] = df["Age"].astype("O")
    assert _is_categorical_and_is_datetime(df["Age"]) is False

    # Object Datetime
    s_obj_dt = pd.Series([pd.Timestamp("2020-01-01")], dtype="object")
    assert _is_categorical_and_is_datetime(s_obj_dt) is True

    # StringDtype Datetime (if convertible)
    s_str_dt = pd.Series(["2020-01-01", "2020-01-02"], dtype="string")
    assert _is_categorical_and_is_datetime(s_str_dt) is True

    # Numeric (should be False for both if and elif branches)
    s_num = pd.Series([1, 2, 3])
    assert _is_categorical_and_is_datetime(s_num) is False

    # Categorical (should hit the 'if' branch)
    s_cat = pd.Series(["a", "b"], dtype="category")
    assert _is_categorical_and_is_datetime(s_cat) is False


def test_is_categorical_and_is_not_datetime(df):
    assert _is_categorical_and_is_not_datetime(df["date_obj0"]) is False
    assert _is_categorical_and_is_not_datetime(df["date_obj0"]) is False
    assert _is_categorical_and_is_not_datetime(df["Name"]) is True

    df["age_str"] = ["20", "21", "19", "18"]
    assert _is_categorical_and_is_not_datetime(df["age_str"]) is True

    # Object Integer
    s_obj_int = pd.Series([1, 2], dtype="object")
    assert _is_categorical_and_is_not_datetime(s_obj_int) is True

    # Object Datetime should be False
    s_obj_dt = pd.Series([pd.Timestamp("2020-01-01")], dtype="object")
    assert _is_categorical_and_is_not_datetime(s_obj_dt) is False

    # StringDtype (not convertible to numeric/datetime) should be True
    s_str = pd.Series(["a", "b"], dtype="string")
    assert _is_categorical_and_is_not_datetime(s_str) is True

    # Numeric should be False
    s_num = pd.Series([1, 2, 3])
    assert _is_categorical_and_is_not_datetime(s_num) is False

    # Categorical should be True (it hits the 'if' branch)
    s_cat = pd.Series(["a", "b"], dtype="category")
    assert _is_categorical_and_is_not_datetime(s_cat) is True


# ---------------------------------------------------------------------------
# narwhals (polars) equivalents
# ---------------------------------------------------------------------------


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


def test_nw_is_convertible_to_dt():
    assert _nw_is_convertible_to_dt(nw_series(["2020-01-01", "2020-01-02"])) is True
    assert _nw_is_convertible_to_dt(nw_series(["a", "b"])) is False
    assert _nw_is_convertible_to_dt(nw_series(["20", "21"])) is False

    # polars has no dateutil-style guesser, unlike pandas, so non-ISO date
    # strings are not recognised
    assert _nw_is_convertible_to_dt(nw_series(["01-Jan-2010"])) is False


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
