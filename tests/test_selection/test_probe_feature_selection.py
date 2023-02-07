import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils.validation import column_or_1d

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
        ProbeFeatureSelection(
            estimator=DecisionTreeRegressor(),
            distribution=_distribution,
        )


@pytest.mark.parametrize("_n_probes", [5, 7, 11])
def test_when_not_permitted_n_probes_with_all_distribution(_n_probes):
    with pytest.raises(ValueError):
        ProbeFeatureSelection(
            estimator=RandomForestClassifier(),
            distribution="all",
            n_probes=_n_probes,
        )


@pytest.mark.parametrize("_n_probes", ["tree", [False, 2], 101.1])
def test_when_not_permitted_n_probes(_n_probes):
    with pytest.raises(ValueError):
        ProbeFeatureSelection(
            estimator=DecisionTreeRegressor(),
            n_probes=_n_probes,
        )


def test_fit_attributes(df_test):

    X, y = df_test
    y = column_or_1d(y, warn=True)

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
        "gaussian_probe_0": [5.366, 1.31, 0.289, -5.59, -0.832],
        "gaussian_probe_1": [0.104, 3.396, -7.67, -0.807, -5.729],
    }

    expected_probe_features_df = pd.DataFrame(expected_probe_features)

    expected_feature_importances = pd.Series(
        data=[0.03, 0, 0, 0, 0.26, 0, 0.22, 0.33, 0.02, 0.12, 0, 0, 0, 0],
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
            "gaussian_probe_0",
            "gaussian_probe_1",
        ],
    )

    assert sel.probe_features_.head().round(3).equals(expected_probe_features_df)
    assert sel.features_to_drop_ == ["var_2", "var_10"]
    assert sel.feature_importances_.round(2).equals(expected_feature_importances)


def test_transformer_with_normal_distribution(load_diabetes_dataset):
    X, y = load_diabetes_dataset
    X.columns = [f"var_{col}" for col in X.columns]
    # y = column_or_1d(y, warn=True)

    sel = ProbeFeatureSelection(
        estimator=DecisionTreeRegressor(),
        scoring="neg_mean_squared_error",
        distribution="normal",
        cv=5,
        n_probes=3,
        random_state=84,
        confirm_variables=False,
    )

    sel.fit(X, y.values.ravel())
    results = sel.transform(X)

    # expected results
    expected_results = {
        "var_2": [0.062, -0.051, 0.044, -0.012, -0.036],
        "var_3": [0.022, -0.026, -0.006, -0.037, 0.022],
        "var_8": [0.02, -0.068, 0.003, 0.023, -0.032],
    }
    expected_results_df = pd.DataFrame(expected_results)

    assert results.head().round(3).equals(expected_results_df)


def test_transformer_with_binary_distribution(load_diabetes_dataset):
    X, y = load_diabetes_dataset

    sel = ProbeFeatureSelection(
        estimator=DecisionTreeRegressor(),
        scoring="neg_mean_absolute_error",
        distribution="binary",
        cv=5,
        n_probes=3,
        random_state=84,
        confirm_variables=False,
    )

    sel.fit(X, y.values.ravel())
    results = sel.transform(X)

    # expected results
    expected_results = {
        "var_0": [0.038, -0.002, 0.085, -0.089, 0.005],
        "var_1": [0.051, -0.045, 0.051, -0.045, -0.045],
        "var_2": [0.062, -0.051, 0.044, -0.012, -0.036],
        "var_3": [0.022, -0.026, -0.006, -0.037, 0.022],
        "var_4": [-0.044, -0.008, -0.046, 0.012, 0.004],
        "var_5": [-0.035, -0.019, -0.034, 0.025, 0.016],
        "var_6": [-0.043, 0.074, -0.032, -0.036, 0.008],
        "var_7": [-0.003, -0.039, -0.003, 0.034, -0.003],
        "var_8": [0.02, -0.068, 0.003, 0.023, -0.032],
        "var_9": [-0.018, -0.092, -0.026, -0.009, -0.047],
    }
    expected_results_df = pd.DataFrame(expected_results)

    assert results.head().round(3).equals(expected_results_df)


def test_transformer_with_uniform_distribution(df_test):
    X, y = df_test
    # y = column_or_1d(y, warn=True)

    sel = ProbeFeatureSelection(
        estimator=LogisticRegression(),
        scoring="roc_auc",
        distribution="uniform",
        cv=5,
        n_probes=2,
        random_state=100,
        confirm_variables=False,
    )

    sel.fit(X, y.ravel())
    results = sel.transform(X)

    # expected results
    expected_results = {
        "var_0": [1.471, 1.819, 1.625, 1.939, 1.579],
        "var_1": [-2.376, 1.969, 1.499, 0.075, 0.372],
        "var_2": [-0.247, -0.127, 0.334, 1.627, 0.338],
        "var_3": [1.21, 0.035, -2.234, 0.943, 0.952],
        "var_4": [-3.248, -2.91, -3.399, -4.783, -3.199],
        "var_5": [0.092, -0.187, -0.314, -0.468, 0.729],
        "var_6": [3.687, 3.319, 3.862, 5.423, 3.636],
        "var_7": [-2.23, -1.447, -2.241, -3.535, -2.054],
        "var_8": [2.07, 2.421, 2.264, 2.792, 2.187],
        "var_9": [3.528, 3.304, 3.717, 5.132, 3.513],
        "var_10": [2.071, 1.185, -0.066, 0.714, 0.399],
        "var_11": [-1.989, -1.31, -0.853, 0.485, -0.187],
    }
    expected_results_df = pd.DataFrame(expected_results)

    assert results.head().round(3).equals(expected_results_df)


def test_transformer_with_all_distribution(df_test):
    X, y = df_test

    sel = ProbeFeatureSelection(
        estimator=DecisionTreeClassifier(),
        scoring="precision",
        distribution="all",
        cv=2,
        n_probes=3,
        random_state=1,
        confirm_variables=False,
    )

    sel.fit(X, y)
    results = sel.transform(X)

    # expected results
    expected_results = {
        "var_0": [1.471, 1.819, 1.625, 1.939, 1.579],
        "var_1": [-2.376, 1.969, 1.499, 0.075, 0.372],
        "var_2": [-0.247, -0.127, 0.334, 1.627, 0.338],
        "var_3": [1.21, 0.035, -2.234, 0.943, 0.952],
        "var_4": [-3.248, -2.91, -3.399, -4.783, -3.199],
        "var_5": [0.092, -0.187, -0.314, -0.468, 0.729],
        "var_6": [3.687, 3.319, 3.862, 5.423, 3.636],
        "var_7": [-2.23, -1.447, -2.241, -3.535, -2.054],
        "var_8": [2.07, 2.421, 2.264, 2.792, 2.187],
        "var_9": [3.528, 3.304, 3.717, 5.132, 3.513],
        "var_10": [2.071, 1.185, -0.066, 0.714, 0.399],
        "var_11": [-1.989, -1.31, -0.853, 0.485, -0.187],
    }
    expected_results_df = pd.DataFrame(expected_results)

    assert results.head().round(3).equals(expected_results_df)
