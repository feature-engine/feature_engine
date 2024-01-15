import pytest
import pandas as pd

from feature_engine.selection.base_selection_functions import (
    _select_all_variables,
    _select_numerical_variables,
)


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "date_range": pd.date_range("2020-02-24", periods=4, freq="T"),
            "date_obj0": ["2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27"],
        }
    )
    df["Name"] = df["Name"].astype("category")
    return df


def test_select_all_variables(df):
    # select all variables
    assert (
        _select_all_variables(
            df, variables=None, confirm_variables=False, exclude_datetime=False
        )
        == df.columns.to_list()
    )

    # select all variables except datetime
    assert _select_all_variables(
        df, variables=None, confirm_variables=False, exclude_datetime=True
    ) == ["Name", "City", "Age", "Marks"]

    # select subset of variables, without confirm
    subset = ["Name", "City", "Age", "Marks"]
    assert _select_all_variables(
        df, variables=subset, confirm_variables=False, exclude_datetime=True
    ) == subset

    # select subset of variables, with confirm
    subset = ["Name", "City", "Age", "Marks", "Hola"]
    assert _select_all_variables(
        df, variables=subset, confirm_variables=True, exclude_datetime=True
    ) == subset[:-1]


def test_select_numerical_variables(df):
    # select all numerical variables
    assert (
            _select_numerical_variables(
                df, variables=None, confirm_variables=False,
            )
            == ["Age", "Marks"]
    )

    # select subset of variables, without confirm
    subset = ["Marks"]
    assert _select_numerical_variables(
        df, variables=subset, confirm_variables=False,
    ) == subset

    # select subset of variables, with confirm
    subset = ["Marks", "Hola"]
    assert _select_numerical_variables(
        df, variables=subset, confirm_variables=True,
    ) == subset[:-1]