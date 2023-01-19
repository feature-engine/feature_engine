import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor


from feature_engine.selection import ProbeFeatureSelection


_input_params = [
    (RandomForestClassifier(), "precision", "all", 3, 6, 4),
    (Lasso(), "neg_mean_squared_error", "binary", 7, 4, 100),
    (LogisticRegression(), "roc_auc", "normal", 5, 2, 73),
    (DecisionTreeRegressor(), "r2", "uniform", 4, 10, 84),
]

@pytest.mark.parametrize(
    "_estimator, _scoring, _distribution, _cv, _n_probes, _random_state", _input_params
)
def test_input_params_assignment(
    _estimator, _scoring, _distribution, _cv, _n_probes, _random_state
):

    sel = ProbeFeatureSelection(
        estimator=_estimator,
        scoring=_scoring,
        distribution=_distribution,
        cv=_cv,
        n_probes=_n_probes,
        random_state=_random_state,
    )

    assert sel.estimator == _estimator
    assert sel.scoring == _scoring
    assert sel.distribution == _distribution
    assert sel.cv == _cv
    assert sel.n_probes == _n_probes
    assert sel.random_state == _random_state


def test_generate_probe_feature(df_test):

    X, y = df_test

    sel = ProbeFeatureSelection(
        estimator=RandomForestClassifier(),
        scoring="recall",
        distribution="all",
        cv=3,
        n_probes=3,
        random_state=33,
        confirm_variables=False,
    )

    sel.fit(X, y)
    res = sel.probe_features_.head().round(3)

    # expected results
    expected_results = {
        "gaussian_probe_0": [-0.957, -4.809, -4.606, -1.711, -0.65],
        "binary_probe_0": [0, 1, 0, 0, 0],
        "uniform_probe_0": [0.287, 0.287, 0.287, 0.287, 0.287],
    }
    expected_results_df = pd.DataFrame(expected_results)

    assert res.equals(expected_results_df)