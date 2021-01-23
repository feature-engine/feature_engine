# import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from feature_engine.selection import SelectByTargetMeanPerformance


def test_numerical_variables_roc_auc(df_test):
    X, y = df_test

    sel = SelectByTargetMeanPerformance(
        variables=None,
        scoring="roc_auc_score",
        threshold=0.6,
        bins=5,
        strategy="equal_width",
        cv=3,
        random_state=1,
    )

    sel.fit(X, y)

    # expected result
    Xtransformed = X[["var_0", "var_4", "var_6", "var_7", "var_9"]]
    # performance_dict = {
    #     "var_0": 0.628,
    #     "var_1": 0.548,
    #     "var_2": 0.513,
    #     "var_3": 0.474,
    #     "var_4": 0.973,
    #     "var_5": 0.496,
    #     "var_6": 0.97,
    #     "var_7": 0.992,
    #     "var_8": 0.536,
    #     "var_9": 0.931,
    #     "var_10": 0.466,
    #     "var_11": 0.517,
    # }

    # test init params
    assert sel.variables == list(X.columns)
    assert sel.scoring == "roc_auc_score"
    assert sel.threshold == 0.6
    assert sel.bins == 5
    assert sel.strategy == "equal_width"
    assert sel.cv == 3
    assert sel.random_state == 1

    # test fit attrs
    assert sel.variables_categorical_ == []
    assert sel.variables_numerical_ == list(X.columns)
    assert sel.features_to_drop_ == [
        "var_1",
        "var_2",
        "var_3",
        "var_5",
        "var_8",
        "var_10",
        "var_11",
    ]
    # assert all(
    #     np.round(sel.feature_performance_[f], 3) == performance_dict[f]
    #     for f in sel.feature_performance_.keys()
    # )
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_categorical_variables_roc_auc(df_test_num_cat):
    X, y = df_test_num_cat
    X = X[["var_A", "var_B"]]

    sel = SelectByTargetMeanPerformance(
        variables=None,
        scoring="roc_auc_score",
        threshold=0.78,
        cv=2,
        random_state=1,
    )

    sel.fit(X, y)

    # expected result
    Xtransformed = X["var_A"].to_frame()
    # performance_dict = {"var_A": 0.841, "var_B": 0.776}

    # test init params
    assert sel.variables == list(X.columns)
    assert sel.scoring == "roc_auc_score"
    assert sel.threshold == 0.78
    assert sel.cv == 2
    assert sel.random_state == 1

    # test fit attrs
    assert sel.variables_categorical_ == list(X.columns)
    assert sel.variables_numerical_ == []
    assert sel.features_to_drop_ == ["var_B"]
    # assert all(
    #     np.round(sel.feature_performance_[f], 3) == performance_dict[f]
    #     for f in sel.feature_performance_.keys()
    # )
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_df_cat_and_num_variables_roc_auc(df_test_num_cat):
    X, y = df_test_num_cat

    sel = SelectByTargetMeanPerformance(
        variables=None,
        scoring="roc_auc_score",
        threshold=0.6,
        bins=3,
        strategy="equal_width",
        cv=2,
        random_state=1,
    )

    sel.fit(X, y)

    # expected result
    Xtransformed = X[["var_A", "var_B"]]
    # performance_dict = {"var_A": 0.841, "var_B": 0.776,
    # "var_C": 0.481, "var_D": 0.496}

    # test init params
    assert sel.variables == list(X.columns)
    assert sel.scoring == "roc_auc_score"
    assert sel.threshold == 0.60
    assert sel.cv == 2
    assert sel.random_state == 1

    # test fit attrs
    assert sel.variables_categorical_ == ["var_A", "var_B"]
    assert sel.variables_numerical_ == ["var_C", "var_D"]
    assert sel.features_to_drop_ == ["var_C", "var_D"]
    # assert all(
    #     np.round(sel.feature_performance_[f], 3) == performance_dict[f]
    #     for f in sel.feature_performance_.keys()
    # )
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_df_cat_and_num_variables_r2(df_test_num_cat):
    X, y = df_test_num_cat

    sel = SelectByTargetMeanPerformance(
        variables=None,
        scoring="r2_score",
        threshold=0.1,
        bins=3,
        strategy="equal_frequency",
        cv=2,
        random_state=1,
    )

    sel.fit(X, y)

    # expected result
    Xtransformed = X[["var_A", "var_B"]]
    # performance_dict = {
    #     "var_A": 0.392,
    #     "var_B": 0.250,
    #     "var_C": -0.004,
    #     "var_D": -0.052,
    # }

    # test init params
    assert sel.variables == list(X.columns)
    assert sel.scoring == "r2_score"
    assert sel.threshold == 0.1
    assert sel.cv == 2
    assert sel.bins == 3
    assert sel.strategy == "equal_frequency"
    assert sel.random_state == 1

    # test fit attrs
    assert sel.variables_categorical_ == ["var_A", "var_B"]
    assert sel.variables_numerical_ == ["var_C", "var_D"]
    assert sel.features_to_drop_ == ["var_C", "var_D"]
    # assert all(
    #     np.round(sel.feature_performance_[f], 3) == performance_dict[f]
    #     for f in sel.feature_performance_.keys()
    # )
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_error_wrong_params():
    with pytest.raises(ValueError):
        SelectByTargetMeanPerformance(scoring="mean_squared")
    with pytest.raises(ValueError):
        SelectByTargetMeanPerformance(scoring=1)
    with pytest.raises(ValueError):
        SelectByTargetMeanPerformance(threshold="hola")
    with pytest.raises(TypeError):
        SelectByTargetMeanPerformance(bins="hola")
    with pytest.raises(ValueError):
        SelectByTargetMeanPerformance(strategy="hola")
    with pytest.raises(ValueError):
        SelectByTargetMeanPerformance(cv="hola")
    with pytest.raises(ValueError):
        SelectByTargetMeanPerformance(cv=1)


def test_error_if_y_not_passed(df_test):
    X, y = df_test
    with pytest.raises(TypeError):
        SelectByTargetMeanPerformance().fit(X)


def test_error_if_input_not_df(df_test):
    X, y = df_test
    with pytest.raises(TypeError):
        SelectByTargetMeanPerformance().fit(X.to_dict(), y)


def test_error_if_fit_input_not_dataframe(df_test):
    with pytest.raises(TypeError):
        SelectByTargetMeanPerformance().fit({"Name": ["Karthik"]})


def test_not_fitted_error(df_test):
    with pytest.raises(NotFittedError):
        transformer = SelectByTargetMeanPerformance()
        transformer.transform(df_test)
