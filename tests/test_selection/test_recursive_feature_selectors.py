import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import (
    RecursiveFeatureAddition,
    RecursiveFeatureElimination,
)

_selectors = [
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


_thresholds = [None, [0.1], "a_string"]


@pytest.mark.parametrize("_selector", _selectors)
@pytest.mark.parametrize("_thresholds", _thresholds)
def test_raises_threshold_error(_selector, _thresholds):
    with pytest.raises(ValueError):
        _selector(RandomForestClassifier(), threshold=_thresholds)


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

    sel = _selector(_classifier, threshold=-100).fit(X, y)

    assert np.round(sel.initial_model_performance_, 4) == _roc

    sel = _selector(
        _regressor,
        scoring="r2",
    ).fit(X, y)

    assert np.round(sel.initial_model_performance_, 4) == _r2


_estimators_importance = [
    (
        RandomForestClassifier(n_estimators=5, random_state=1),
        [
            0.49881322433327063,
            0.24234595114295532,
            0.17684337500037525,
            0.025427859893734316,
            0.02436691418502239,
            0.00842195125881095,
            0.006712051147370731,
            0.00553740334514697,
            0.004739876867868088,
            0.00321013573560038,
            0.001870728276222728,
            0.0017105288136222314,
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

    sel = RecursiveFeatureAddition(_estimator, threshold=-100).fit(X, y)
    _importance.sort(reverse=True)
    assert (
        list(np.round(sel.feature_importances_.values, 2)) == np.round(_importance, 2)
    ).all()

    sel = RecursiveFeatureElimination(_estimator, threshold=-100).fit(X, y)
    _importance.sort(reverse=False)
    assert (
        list(np.round(sel.feature_importances_.values, 2)) == np.round(_importance, 2)
    ).all()
