import pytest

from feature_engine.variable_handling._variable_type_checks import (
    _is_categorical_and_is_datetime,
    _is_categorical_and_is_not_datetime,
    _is_convertible_to_dt,

)

def test_is_convertible_to_num(df):
    assert _is_convertible_to_dt(df["Name"]) is False
    assert _is_convertible_to_dt(df["date_obj0"]) is True

def test_is_convertible_to_dt(df):
    assert _is_convertible_to_dt(df["date_obj0"]) is True
    assert _is_convertible_to_dt(df["date_range"]) is True
    assert _is_convertible_to_dt(df["Name"]) is False

def test_is_categorical_and_is_datetime(df):
    assert _is_categorical_and_is_datetime(df["date_obj0"]) is True
    assert _is_categorical_and_is_datetime(df["Name"]) is False

def test_is_categorical_and_is_not_datetime(df):
    assert _is_categorical_and_is_not_datetime(df["date_obj0"]) is False
    assert _is_categorical_and_is_not_datetime(df["date_obj0"]) is False
    assert _is_categorical_and_is_not_datetime(df["Name"]) is True
