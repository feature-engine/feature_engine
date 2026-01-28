from feature_engine.variable_handling._variable_type_checks import (
    _is_categorical_and_is_datetime,
    _is_categorical_and_is_not_datetime,
    _is_categories_num,
    _is_convertible_to_dt,
    _is_convertible_to_num,
)


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


def test_is_categorical_and_is_not_datetime(df):
    assert _is_categorical_and_is_not_datetime(df["date_obj0"]) is False
    assert _is_categorical_and_is_not_datetime(df["date_obj0"]) is False
    assert _is_categorical_and_is_not_datetime(df["Name"]) is True

    df["age_str"] = ["20", "21", "19", "18"]
    assert _is_categorical_and_is_not_datetime(df["age_str"]) is True
