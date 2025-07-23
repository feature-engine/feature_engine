import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold


from feature_engine.selection.base_selection_functions import (
    _select_all_variables,
    _select_numerical_variables,
    find_correlated_features,
    find_feature_importance,
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
    rf = RandomForestClassifier(n_estimators=5, random_state=1)
    variables = X.columns.to_list()

    mean_, std_ = single_feature_performance(
        X=X,
        y=y,
        variables=variables,
        estimator=rf,
        cv=3,
        scoring="roc_auc",
    )

    expected_mean = {
        "var_0": 0.5813469607144305,
        "var_1": 0.5325152703164752,
        "var_2": 0.5023573007759755,
        "var_3": 0.47596844810700234,
        "var_4": 0.9696712897767115,
        "var_5": 0.5078009005719849,
        "var_6": 0.966096275433625,
        "var_7": 0.9918595739378872,
        "var_8": 0.521667767752105,
        "var_9": 0.9476311088509884,
        "var_10": 0.4871054926777818,
        "var_11": 0.5180029642379039,
    }
    expected_std = {
        "var_0": 0.0035430274728173775,
        "var_1": 0.0046697767238672565,
        "var_2": 0.023714708852568194,
        "var_3": 0.04219857610624132,
        "var_4": 0.010364079344188424,
        "var_5": 0.03203946151605523,
        "var_6": 0.0063709642968091335,
        "var_7": 0.0014579159677989356,
        "var_8": 0.027570153897628277,
        "var_9": 0.014363240810578251,
        "var_10": 0.020283618255582142,
        "var_11": 0.02707242215734807,
    }
    assert mean_ == expected_mean
    assert std_ == expected_std


def test_single_feature_performance_cv_generator(df_test):
    X, y = df_test
    rf = RandomForestClassifier(n_estimators=5, random_state=1)
    variables = X.columns.to_list()
    cv = StratifiedKFold(n_splits=3)
    for cv_ in [cv, cv.split(X, y)]:
        mean_, _ = single_feature_performance(
            X=X,
            y=y,
            variables=variables,
            estimator=rf,
            cv=cv_,
            scoring="roc_auc",
        )

        expected_mean = {
            "var_0": 0.5813469607144305,
            "var_1": 0.5325152703164752,
            "var_2": 0.5023573007759755,
            "var_3": 0.47596844810700234,
            "var_4": 0.9696712897767115,
            "var_5": 0.5078009005719849,
            "var_6": 0.966096275433625,
            "var_7": 0.9918595739378872,
            "var_8": 0.521667767752105,
            "var_9": 0.9476311088509884,
            "var_10": 0.4871054926777818,
            "var_11": 0.5180029642379039,
        }
        assert mean_ == expected_mean


def test_single_feature_performance_with_groups(df_test_with_groups):
    X, y, groups = df_test_with_groups
    rf = RandomForestClassifier(n_estimators=5, random_state=1)
    variables = X.columns.to_list()
    scoring = "neg_mean_absolute_error"
    cv = GroupKFold(n_splits=3)
    cv_indices = cv.split(X=X, y=y, groups=groups)

    expected_mean_, expected_std_ = single_feature_performance(
        X=X,
        y=y,
        variables=variables,
        estimator=rf,
        cv=cv_indices,
        scoring=scoring,
    )

    mean_, std_ = single_feature_performance(
        X=X,
        y=y,
        variables=variables,
        estimator=rf,
        cv=cv,
        scoring=scoring,
        groups=groups,
    )

    assert mean_ == expected_mean_
    assert std_ == expected_std_


def test_find_feature_importance(df_test):
    X, y = df_test
    rf = RandomForestClassifier(n_estimators=3, random_state=3)
    cv = StratifiedKFold(n_splits=3)
    scoring = "recall"

    expected_mean = pd.Series(
        data=[0.01, 0.0, 0.0, 0.0, 0.1, 0.01, 0.07, 0.8, 0.01, 0.01, 0.0, 0.01],
        index=[
            "var_0",
            "var_1",
            "var_2",
            "var_3",
            "var_4",
            "var_5",
            "var_6",
            "var_7",
            "var_8",
            "var_9",
            "var_10",
            "var_11",
        ],
    )
    expected_std = pd.Series(
        data=[
            0.0045,
            0.0044,
            0.0013,
            0.0,
            0.1583,
            0.0059,
            0.0666,
            0.1234,
            0.0037,
            0.0016,
            0.0011,
            0.0055,
        ],
        index=[
            "var_0",
            "var_1",
            "var_2",
            "var_3",
            "var_4",
            "var_5",
            "var_6",
            "var_7",
            "var_8",
            "var_9",
            "var_10",
            "var_11",
        ],
    )

    mean_, std_ = find_feature_importance(
        X=X,
        y=y,
        estimator=rf,
        cv=cv,
        scoring=scoring,
    )
    pd.testing.assert_series_equal(mean_.round(2), expected_mean)
    pd.testing.assert_series_equal(std_.round(4), expected_std)

    mean_, std_ = find_feature_importance(
        X=X,
        y=y,
        estimator=rf,
        cv=cv.split(X, y),
        scoring=scoring,
    )
    pd.testing.assert_series_equal(mean_.round(2), expected_mean)
    pd.testing.assert_series_equal(std_.round(4), expected_std)


def test_find_feature_importancewith_groups(df_test_with_groups):
    X, y, groups = df_test_with_groups
    rf = RandomForestClassifier(n_estimators=3, random_state=1)
    cv = GroupKFold(n_splits=3)
    scoring = "neg_mean_absolute_error"
    cv_indices = cv.split(X=X, y=y, groups=groups)

    expected_mean_, expected_std_ = find_feature_importance(
        X=X,
        y=y,
        estimator=rf,
        cv=cv_indices,
        scoring=scoring,
    )

    mean_, std_ = find_feature_importance(
        X=X,
        y=y,
        estimator=rf,
        cv=cv,
        scoring=scoring,
        groups=groups
    )

    pd.testing.assert_series_equal(mean_, expected_mean_)
    pd.testing.assert_series_equal(std_, expected_std_)
