import warnings

import pandas as pd
import pytest

from feature_engine.variable_handling import (
    find_all_variables,
    find_categorical_and_numerical_variables,
    find_categorical_variables,
    find_datetime_variables,
    find_numerical_variables,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "Age": [20, 21, 22],
            "Marks": [85, 90, 95],
            "Gender": ["M", "F", "M"],
            "Date": pd.date_range("2020-01-01", periods=3),
        }
    )


@pytest.fixture
def df_empty():
    return pd.DataFrame(
        {
            "Gender": ["M", "F", "M"],
            "Date": pd.date_range("2020-01-01", periods=3),
        }
    )


# --- find_numerical_variables --- #


def test_find_numerical_variables_error(df):
    msg = (
        "No numerical variables found in this dataframe. Check "
        "variable format with pandas dtypes or set allow_empty to True "
        "to return an empty list instead."
    )
    with pytest.raises(TypeError, match=msg):
        find_numerical_variables(df_empty())


def test_find_numerical_variables_allow_empty_returns_empty_list(df_empty):
    result = find_numerical_variables(df_empty(), allow_empty=True)
    assert result == []


def test_find_numerical_variables_warns(df_empty):
    with pytest.warns(UserWarning, match="No numerical variables found"):
        find_numerical_variables(df_empty(), allow_empty=True)


# --- find_categorical_variables --- #


def test_find_categorical_variables_error(df):
    df_num_only = df[["Age", "Marks"]]
    msg = (
        "No categorical variables found in this dataframe. Check variable "
        "format with pandas dtypes or set allow_empty to True to return an "
        "empty list instead."
    )
    with pytest.raises(TypeError, match=msg):
        find_categorical_variables(df_num_only)


def test_find_categorical_variables_allow_empty_returns_empty_list(df_num_only):
    result = find_categorical_variables(df_num_only, allow_empty=True)
    assert result == []


def test_find_categorical_variables_warns(df_num_only):
    with pytest.warns(UserWarning, match="No categorical variables found"):
        find_categorical_variables(df_num_only, allow_empty=True)


# --- find_datetime_variables --- #


def test_find_datetime_variables_error(df):
    df_no_date = df[["Age", "Marks", "Gender"]]
    msg = "No datetime variables found in this dataframe."
    with pytest.raises(ValueError, match=msg):
        find_datetime_variables(df_no_date)


def test_find_datetime_variables_allow_empty_returns_empty_list(df_no_date):
    result = find_datetime_variables(df_no_date, allow_empty=True)
    assert result == []


def test_find_datetime_variables_warns(df_no_date):
    with pytest.warns(UserWarning, match="No datetime variables found"):
        find_datetime_variables(df_no_date, allow_empty=True)


# --- find_all_variables --- #


def test_find_all_variables_empty_warn(df_empty):
    with pytest.warns(UserWarning, match="No variables found"):
        result = find_all_variables(df_empty(), allow_empty=True)
        assert result == []


def test_find_all_variables_error(df_empty):
    msg = "No variables found in this dataframe."
    with pytest.raises(ValueError, match=msg):
        find_all_variables(df_empty())


# --- find_categorical_and_numerical_variables --- #


def test_find_cat_and_num_variables_empty_warn(df_empty):
    cat_vars, num_vars = find_categorical_and_numerical_variables(
        df_empty(), allow_empty=True
    )
    assert cat_vars == []
    assert num_vars == []


def test_find_cat_and_num_variables_error(df_empty):
    msg = "There are no numerical or categorical variables in the dataframe"
    with pytest.raises(TypeError, match=msg):
        find_categorical_and_numerical_variables(df_empty())


def test_find_cat_and_num_variables_single_variable_warn(df_empty):
    df_invalid = pd.DataFrame({"col": [None, None]})
    cat_vars, num_vars = find_categorical_and_numerical_variables(
        df_invalid, variables="col", allow_empty=True
    )
    assert cat_vars == []
    assert num_vars == []


def test_find_cat_and_num_variables_list_warn(df_empty):
    df_invalid = pd.DataFrame({"col": [None, None]})
    cat_vars, num_vars = find_categorical_and_numerical_variables(
        df_invalid, variables=["col"], allow_empty=True
    )
    assert cat_vars == []
    assert num_vars == []


def test_find_cat_and_num_variables_list_error(df_empty):
    df_invalid = pd.DataFrame({"col": [None, None]})
    msg = "Some of the variables are neither numerical nor categorical."
    with pytest.raises(TypeError, match=msg):
        find_categorical_and_numerical_variables(df_invalid, variables=["col"])
