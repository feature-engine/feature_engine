import pandas as pd
import pytest

from feature_engine.selection import SelectByTargetMeanPerformance


def df_classification():
    df = {
        "cat_var_A": ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5,
        "cat_var_B": ["A"] * 6
        + ["B"] * 2
        + ["C"] * 2
        + ["B"] * 2
        + ["C"] * 2
        + ["D"] * 6,
        "num_var_A": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        "num_var_B": [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4],
    }

    df = pd.DataFrame(df)
    df = pd.concat([df, df, df, df], axis=0)

    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y = pd.concat([y, y, y, y], axis=0)

    return df, y


def df_regression():
    df = {
        "cat_var_A": ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5,
        "cat_var_B": ["A"] * 6
        + ["B"] * 2
        + ["C"] * 2
        + ["B"] * 2
        + ["C"] * 2
        + ["D"] * 6,
        "num_var_A": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        "num_var_B": [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4],
    }

    df = pd.DataFrame(df)
    df = pd.concat([df, df, df, df], axis=0)

    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    y = pd.concat([y, y, y, y], axis=0)
    return df, y


def test_classification():

    X, y = df_classification()

    sel = SelectByTargetMeanPerformance(
        variables=None,
        scoring="accuracy",
        threshold=None,
        bins=2,
        strategy="equal_width",
        cv=2,
    )

    sel.fit(X, y)

    # expected result
    Xtransformed = X[["cat_var_A", "num_var_A"]]

    performance_dict = {
        "cat_var_A": 1.0,
        "cat_var_B": 0.8,
        "num_var_A": 1.0,
        "num_var_B": 0.8,
    }
    features_to_drop = ["cat_var_B", "num_var_B"]

    assert sel.features_to_drop_ == features_to_drop
    assert sel.feature_performance_ == performance_dict
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)

    sel = SelectByTargetMeanPerformance(
        variables=["cat_var_A", "cat_var_B", "num_var_A", "num_var_B"],
        scoring="roc_auc",
        threshold=0.9,
        bins=2,
        strategy="equal_frequency",
        cv=2,
    )

    sel.fit(X, y)

    # expected result
    Xtransformed = X[["cat_var_A", "cat_var_B", "num_var_A"]]

    performance_dict = {
        "cat_var_A": 1.0,
        "cat_var_B": 0.92,
        "num_var_A": 1.0,
        "num_var_B": 0.8,
    }
    features_to_drop = ["num_var_B"]

    assert sel.features_to_drop_ == features_to_drop
    assert sel.feature_performance_ == performance_dict
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression():

    X, y = df_regression()

    sel = SelectByTargetMeanPerformance(
        variables=None,
        bins=2,
        scoring="r2",
        regression=True,
        cv=2,
        strategy="equal_width",
        threshold=None,
    )

    sel.fit(X, y)

    # expected result
    Xtransformed = X[["cat_var_A", "cat_var_B", "num_var_A"]]
    performance_dict = {
        "cat_var_A": 1.0,
        "cat_var_B": 0.8533333333333333,
        "num_var_A": 0.8,
        "num_var_B": 0.512,
    }

    assert sel.features_to_drop_ == ["num_var_B"]
    assert sel.feature_performance_ == performance_dict
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)

    X, y = df_regression()

    sel = SelectByTargetMeanPerformance(
        variables=["cat_var_A", "cat_var_B", "num_var_A", "num_var_B"],
        bins=2,
        scoring="neg_root_mean_squared_error",
        regression=True,
        cv=2,
        strategy="equal_width",
        threshold=-0.2,
    )

    sel.fit(X, y)

    # expected result
    Xtransformed = X["cat_var_A"].to_frame()
    performance_dict = {
        "cat_var_A": 0.0,
        "cat_var_B": -0.42817441928883765,
        "num_var_A": -0.5,
        "num_var_B": -0.7810249675906654,
    }

    assert sel.features_to_drop_ == ["cat_var_B", "num_var_A", "num_var_B"]
    assert sel.feature_performance_ == performance_dict
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_error_wrong_params():
    with pytest.raises(ValueError):
        SelectByTargetMeanPerformance(scoring="mean_squared")
    with pytest.raises(ValueError):
        SelectByTargetMeanPerformance(scoring=1)
    with pytest.raises(ValueError):
        SelectByTargetMeanPerformance(threshold="hola")
    with pytest.raises(ValueError):
        SelectByTargetMeanPerformance(bins="hola")
    with pytest.raises(ValueError):
        SelectByTargetMeanPerformance(strategy="hola")


def test_raises_error_if_evaluating_single_variable_and_threshold_is_None(df_test):
    X, y = df_test

    sel = SelectByTargetMeanPerformance(variables=["var_1"], threshold=None)

    with pytest.raises(ValueError):
        sel.fit(X, y)


def test_test_selector_with_one_variable():

    X, y = df_regression()

    sel = SelectByTargetMeanPerformance(
        variables=["cat_var_A"],
        bins=2,
        scoring="neg_root_mean_squared_error",
        regression=True,
        cv=2,
        strategy="equal_width",
        threshold=-0.2,
    )

    sel.fit(X, y)

    # expected result
    performance_dict = {"cat_var_A": 0.0}

    assert sel.features_to_drop_ == []
    assert sel.feature_performance_ == performance_dict
    pd.testing.assert_frame_equal(sel.transform(X), X)

    X, y = df_regression()

    sel = SelectByTargetMeanPerformance(
        variables=["cat_var_B"],
        bins=2,
        scoring="neg_root_mean_squared_error",
        regression=True,
        cv=2,
        strategy="equal_width",
        threshold=-0.2,
    )

    sel.fit(X, y)

    # expected result
    Xtransformed = X.drop(columns=["cat_var_B"])
    performance_dict = {"cat_var_B": -0.42817441928883765}

    assert sel.features_to_drop_ == ["cat_var_B"]
    assert sel.feature_performance_ == performance_dict
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)
