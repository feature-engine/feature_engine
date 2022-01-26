import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection.base_recursive_selector import BaseRecursiveSelector


_input_params = [
    (RandomForestClassifier(), 'roc_auc', 3, 0.1, None),
    (LinearRegression(), "neg_mean_squared_error", KFold(), 0.01, ['var_a', 'var_b']),
    (DecisionTreeRegressor(), 'r2', StratifiedKFold(), 0.5, ['var_a']),
    (RandomForestClassifier(), 'accuracy', 5, 0.002, 'var_a'),
]


@pytest.mark.parametrize(
    "_estimator, _scoring, _cv, _threshold, _variables", _input_params
)
def test_input_params_assignment(
    _estimator, _scoring, _cv, _threshold, _variables
):
    sel = BaseRecursiveSelector(
        estimator=_estimator,
        scoring=_scoring,
        cv=_cv,
        threshold=_threshold,
        variables=_variables,
    )

    assert sel.estimator==_estimator
    assert sel.scoring==_scoring
    assert sel.cv==_cv
    assert sel.threshold==_threshold
    assert sel.variables==_variables


def test_raises_error_when_no_estimator_passed():
    with pytest.raises(TypeError):
        BaseRecursiveSelector()


_thresholds = [None, [0.1], "a_string"]

@pytest.mark.parametrize("_thresholds", _thresholds)
def test_raises_threshold_error(_thresholds):
    with pytest.raises(ValueError):
        BaseRecursiveSelector(RandomForestClassifier(), threshold=_thresholds)

_not_a_df = [
    "not_a_df",
    [1, 2, 3, "some_data"],
    pd.Series([-2, 1.5, 8.94], name="not_a_df"),
]


@pytest.mark.parametrize("_not_a_df", _not_a_df)
def test_raises_error_when_fitting_not_a_df(_not_a_df):
    transformer = BaseRecursiveSelector(RandomForestClassifier())
    # trying to fit not a df
    with pytest.raises(TypeError):
        transformer.fit(_not_a_df)


# _variables = ["var_1", ["var_2"],["var_1", "var_2", "var_3", "var_11"], None]
#
#
# @pytest.mark.parametrize("_variables", _variables)
# def test_variables_params(_variables, df_test):
#     X, y = df_test
#     sel = BaseRecursiveSelector(
#         RandomForestClassifier(), variables=_variables
#     ).fit(X, y)
#
#     if _variables is not None:
#         assert sel.variables == _variables
#         if isinstance(_variables, list):
#             assert sel.variables_ == _variables
#         else:
#             assert sel.variables_ == [_variables]
#     else:
#         assert sel.variables is None
#         assert sel.variables_ == ["var_" + str(i) for i in range(12)]
#
#     # test selector excludes non-numerical variables automatically
#     X['cat_var'] = ['A']*1000
#     sel = BaseRecursiveSelector(
#         RandomForestClassifier(), variables=None
#     ).fit(X, y)
#     assert sel.variables is None
#     assert sel.variables_ == ["var_" + str(i) for i in range(12)]


def test_raises_error_when_user_passes_categorical_var(df_test):
    X, y = df_test

    # add categorical variable
    X['cat_var'] = ['A'] * 1000

    with pytest.raises(TypeError):
        BaseRecursiveSelector(
            RandomForestClassifier(), variables=["var_1", "var_2", "cat_var"]
        ).fit(X, y)

    with pytest.raises(TypeError):
        BaseRecursiveSelector(
            RandomForestClassifier(), variables="cat_var").fit(X, y)
