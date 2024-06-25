import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from feature_engine.selection.base_selection_functions import (
    _select_all_variables,
    _select_numerical_variables,
    find_correlated_features,
    single_feature_performance,
)


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "date_range": pd.date_range("2020-02-24", periods=4, freq="min"),
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
    assert (
        _select_all_variables(
            df, variables=subset, confirm_variables=False, exclude_datetime=True
        )
        == subset
    )

    # select subset of variables, with confirm
    subset = ["Name", "City", "Age", "Marks", "Hola"]
    assert (
        _select_all_variables(
            df, variables=subset, confirm_variables=True, exclude_datetime=True
        )
        == subset[:-1]
    )


def test_select_numerical_variables(df):
    # select all numerical variables
    assert _select_numerical_variables(
        df,
        variables=None,
        confirm_variables=False,
    ) == ["Age", "Marks"]

    # select subset of variables, without confirm
    subset = ["Marks"]
    assert (
        _select_numerical_variables(
            df,
            variables=subset,
            confirm_variables=False,
        )
        == subset
    )

    # select subset of variables, with confirm
    subset = ["Marks", "Hola"]
    assert (
        _select_numerical_variables(
            df,
            variables=subset,
            confirm_variables=True,
        )
        == subset[:-1]
    )


def test_find_correlated_features():
    # given a correlation-threshold of 0.7
    # a is uncorrelated,
    # b is collinear to c and d,
    # c and d are collinear only to b.
    X = pd.DataFrame()
    X["a"] = [1, -1, 0, 0, 0, 0]
    X["b"] = [0, 0, 1, -1, 1, -1]
    X["c"] = [0, 0, 1, -1, 0, 0]
    X["d"] = [0, 0, 0, 0, 1, -1]

    groups, drop, dict_ = find_correlated_features(
        X, variables=["a", "b", "c", "d"], method="pearson", threshold=0.7
    )

    assert groups == [{"b", "c", "d"}]
    assert drop == ["c", "d"]
    assert dict_ == {"b": {"c", "d"}}

    groups, drop, dict_ = find_correlated_features(
        X, variables=["a", "c", "b", "d"], method="pearson", threshold=0.7
    )

    assert groups == [{"c", "b"}]
    assert drop == ["b"]
    assert dict_ == {"c": {"b"}}


def test_single_feature_performance(df_test):
    X, y = df_test
    rf = RandomForestClassifier(random_state=1)
    variables = X.columns.to_list()

    result = single_feature_performance(X, y, variables, rf, 3, "roc_auc")

    expected = {
        "var_0": 0.5957642619540211,
        "var_1": 0.5365534287221033,
        "var_2": 0.5001855546283257,
        "var_3": 0.4752954458526748,
        "var_4": 0.9780875304971691,
        "var_5": 0.5065441419357082,
        "var_6": 0.9758243290622809,
        "var_7": 0.994571685008432,
        "var_8": 0.5164434795458892,
        "var_9": 0.9543427678969847,
        "var_10": 0.47404183834906727,
        "var_11": 0.5227164067525513,
    }
    assert result == expected
