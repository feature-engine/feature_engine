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


@pytest.mark.parametrize("_distribution", [3, "poisson", ["normal", "binary"], 2.22])
def test_when_not_permitted_distribution(_distribution):
    with pytest.raises(ValueError):
        ProbeFeatureSelection(distribution=_distribution)


@pytest.mark.parametrize("_n_probes", [5, 7, 11])
def test_when_not_permitted_n_probes_with_all_distribution(_n_probes):
    with pytest.raises(ValueError):
        ProbeFeatureSelection(
            distribution="all",
            n_probes=_n_probes,
        )


@pytest.mark.parametrize("_n_probes", "tree", [False, 2], 101.1)
def test_when_not_permitted_n_probes(_n_probes):
    with pytest.raises(ValueError):
        ProbeFeatureSelection(n_probes=_n_probes)


def test_fit_attributes(df_test):

    X, y = df_test

    sel = ProbeFeatureSelection(
        estimator=RandomForestClassifier(),
        distribution="normal",
        n_probes=2,
        scoring="recall",
        cv=3,
        random_state=3,
        confirm_variables=False,
    )

    sel.fit(X, y)

    # expected results
    expected_probe_features = {
        "gaussian_probe_0": [5.366,  1.31,  0.289, -5.59, -0.832],
        "gaussian_probe_1": [0.104,  3.396, -7.67, -0.807, -5.729],
    }

    expected_probe_features_df = pd.DataFrame(expected_probe_features)

    expected_feature_importances = pd.Series(
        data=[0.03, 0, 0, 0, 0.26, 0, 0.22, 0.33, 0.02, 0.12, 0, 0, 0, 0],
        index=[
            'var_0',
            'var_1',
            'var_2',
            'var_3',
            'var_4',
            'var_5',
            'var_6',
            'var_7',
            'var_8',
            'var_9',
            'var_10',
            'var_11',
            'gaussian_probe_0',
            'gaussian_probe_1',
        ]
    )

    assert sel.probe_features_.head().round(3).equals(expected_probe_features_df)
    assert sel.features_to_drop_ == ['var_2', 'var_10']
    assert sel.feature_importances_.round(2).equals(expected_feature_importances)