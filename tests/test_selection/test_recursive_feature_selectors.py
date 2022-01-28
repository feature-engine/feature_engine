import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import (
    RecursiveFeatureAddition,
    RecursiveFeatureElimination,
)
from feature_engine.selection.base_recursive_selector import BaseRecursiveSelector

_selectors = [
    BaseRecursiveSelector,
    RecursiveFeatureElimination,
    RecursiveFeatureAddition,
]

_input_params = [
    (RandomForestClassifier(), "roc_auc", 3, 0.1, None),
    (Lasso(), "neg_mean_squared_error", KFold(), 0.01, ["var_a", "var_b"]),
    (DecisionTreeRegressor(), "r2", StratifiedKFold(), 0.5, ["var_a"]),
    (RandomForestClassifier(), "accuracy", 5, 0.002, "var_a"),
]


@pytest.mark.parametrize("_selector", _selectors)
@pytest.mark.parametrize(
    "_estimator, _scoring, _cv, _threshold, _variables", _input_params
)
def test_input_params_assignment(
    _selector, _estimator, _scoring, _cv, _threshold, _variables
):
    sel = _selector(
        estimator=_estimator,
        scoring=_scoring,
        cv=_cv,
        threshold=_threshold,
        variables=_variables,
    )

    assert sel.estimator == _estimator
    assert sel.scoring == _scoring
    assert sel.cv == _cv
    assert sel.threshold == _threshold
    assert sel.variables == _variables


@pytest.mark.parametrize("_selector", _selectors)
def test_raises_error_when_no_estimator_passed(_selector):
    with pytest.raises(TypeError):
        _selector()


_thresholds = [None, [0.1], "a_string"]


@pytest.mark.parametrize("_selector", _selectors)
@pytest.mark.parametrize("_thresholds", _thresholds)
def test_raises_threshold_error(_selector, _thresholds):
    with pytest.raises(ValueError):
        _selector(RandomForestClassifier(), threshold=_thresholds)


_not_a_df = [
    "not_a_df",
    [1, 2, 3, "some_data"],
    pd.Series([-2, 1.5, 8.94], name="not_a_df"),
]


@pytest.mark.parametrize("_selector", _selectors)
@pytest.mark.parametrize("_not_a_df", _not_a_df)
def test_raises_error_when_fitting_not_a_df(_selector, _not_a_df):
    transformer = _selector(RandomForestClassifier())
    # trying to fit not a df
    with pytest.raises(TypeError):
        transformer.fit(_not_a_df)


_variables = ["var_1", ["var_2"], ["var_1", "var_2", "var_3", "var_11"], None]


@pytest.mark.parametrize("_selector", _selectors)
@pytest.mark.parametrize("_variables", _variables)
def test_variables_params(_selector, _variables, df_test):
    X, y = df_test

    sel = _selector(LogisticRegression(max_iter=2), variables=_variables).fit(X, y)

    if _variables is not None:
        assert sel.variables == _variables

        if isinstance(_variables, list):
            assert sel.variables_ == _variables
        else:
            assert sel.variables_ == [_variables]
    else:
        assert sel.variables is None
        assert sel.variables_ == ["var_" + str(i) for i in range(12)]

    # test selector excludes non-numerical variables automatically
    X["cat_var"] = ["A"] * 1000
    sel = _selector(LogisticRegression(max_iter=2), variables=None).fit(X, y)
    assert sel.variables is None
    assert sel.variables_ == ["var_" + str(i) for i in range(12)]


@pytest.mark.parametrize("_selector", _selectors)
def test_raises_error_when_user_passes_categorical_var(_selector, df_test):
    X, y = df_test

    # add categorical variable
    X["cat_var"] = ["A"] * 1000

    with pytest.raises(TypeError):
        _selector(
            RandomForestClassifier(), variables=["var_1", "var_2", "cat_var"]
        ).fit(X, y)

    with pytest.raises(TypeError):
        _selector(RandomForestClassifier(), variables="cat_var").fit(X, y)


_estimators_and_results = [
    (
        RandomForestClassifier(random_state=1),
        Lasso(alpha=0.01, random_state=1),
        0.9971,
        0.8489,
    ),
    (
        LogisticRegression(random_state=1),
        DecisionTreeRegressor(random_state=1),
        0.9966,
        0.9399,
    ),
]


@pytest.mark.parametrize("_selector", _selectors)
@pytest.mark.parametrize("_classifier, _regressor, _roc, _r2", _estimators_and_results)
def test_fit_initial_model_performance(
    _selector, _classifier, _regressor, _roc, _r2, df_test
):
    X, y = df_test

    sel = _selector(_classifier).fit(X, y)

    assert np.round(sel.initial_model_performance_, 4) == _roc

    sel = _selector(
        _regressor,
        scoring="r2",
    ).fit(X, y)

    assert np.round(sel.initial_model_performance_, 4) == _r2


_estimators_importance = [
    (
        RandomForestClassifier(random_state=1),
        [
            0.0238,
            0.0042,
            0.0022,
            0.0021,
            0.2583,
            0.0034,
            0.2012,
            0.38,
            0.0145,
            0.1044,
            0.0035,
            0.0024,
        ],
    ),
    (
        LogisticRegression(random_state=1),
        [
            1.4106,
            0.1924,
            0.0876,
            0.066,
            0.5421,
            0.0825,
            0.5658,
            2.1938,
            1.5259,
            0.1173,
            0.1673,
            0.1792,
        ],
    ),
    (
        Lasso(alpha=0.01, random_state=1),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2126, 0.0557, 0.0, 0.0, 0.0],
    ),
    (
        DecisionTreeRegressor(random_state=1),
        [
            0.0016,
            0.0,
            0.002,
            0.002,
            0.0013,
            0.001,
            0.0026,
            0.976,
            0.0106,
            0.0,
            0.0006,
            0.0022,
        ],
    ),
]


@pytest.mark.parametrize("_estimator, _importance", _estimators_importance)
def test_feature_importances(_estimator, _importance, df_test):
    X, y = df_test

    # Test Base Recursive Selector
    sel = BaseRecursiveSelector(_estimator).fit(X, y)
    assert list(np.round(sel.feature_importances_.values, 4)) == _importance

    sel = RecursiveFeatureAddition(_estimator).fit(X, y)
    _importance.sort(reverse=True)
    assert list(np.round(sel.feature_importances_.values, 4)) == _importance

    sel = RecursiveFeatureElimination(_estimator).fit(X, y)
    _importance.sort(reverse=False)
    assert list(np.round(sel.feature_importances_.values, 4)) == _importance


_cv_constructor = [KFold(), StratifiedKFold()]


@pytest.mark.parametrize("_selector", _selectors)
@pytest.mark.parametrize("_cv", _cv_constructor)
def test_feature_KFold_constructor(_selector, _cv, df_test):
    X, y = df_test

    sel = _selector(Lasso(alpha=0.01, random_state=1), cv=_cv).fit(X, y)

    assert hasattr(sel, "initial_model_performance_")
    assert hasattr(sel, "feature_importances_")
