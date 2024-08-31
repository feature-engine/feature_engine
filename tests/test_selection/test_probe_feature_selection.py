import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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


@pytest.mark.parametrize("collective", [True, False])
def test_collective_param(collective):
    tr = ProbeFeatureSelection(
        estimator=DecisionTreeRegressor(),
        collective=collective,
    )
    assert tr.collective is collective


@pytest.mark.parametrize("collective", [10, "string", 0.1])
def test_collective_raises_error(collective):
    msg = f"collective takes values True or False. Got {collective} instead."
    with pytest.raises(ValueError, match=msg):
        ProbeFeatureSelection(
            estimator=DecisionTreeRegressor(),
            collective=collective,
        )


@pytest.mark.parametrize("_distribution", [3, "poisson", ["normal", "binary"], 2.22])
def test_raises_error_when_not_permitted_distribution(_distribution):
    with pytest.raises(ValueError):
        ProbeFeatureSelection(
            estimator=DecisionTreeRegressor(),
            distribution=_distribution,
        )


@pytest.mark.parametrize("_n_probes", [5, 7, 11])
def test_raises_error_when_not_permitted_n_probes_with_all_distribution(_n_probes):
    with pytest.raises(ValueError):
        ProbeFeatureSelection(
            estimator=RandomForestClassifier(),
            distribution="all",
            n_probes=_n_probes,
        )


@pytest.mark.parametrize("_n_probes", ["tree", [False, 2], 101.1])
def test_raises_error_when_not_permitted_n_probes(_n_probes):
    with pytest.raises(ValueError):
        ProbeFeatureSelection(
            estimator=DecisionTreeRegressor(),
            n_probes=_n_probes,
        )


def test_fit_transform_functionality(df_test):
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
    X_tr = sel.fit_transform(X, y)

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
    pd.testing.assert_frame_equal(
        sel.probe_features_.head().round(3),
        expected_probe_features_df,
        check_dtype=False,
    )
    assert sel.feature_importances_.round(2).equals(expected_feature_importances)
    assert sel.features_to_drop_ == ["var_2", "var_10"]
    pd.testing.assert_frame_equal(
        X_tr, X.drop(columns=["var_2", "var_10"]), check_dtype=False
    )


def test_generate_probe_features_all():
    sel = ProbeFeatureSelection(
        estimator=DecisionTreeClassifier(),
        n_probes=6,
        distribution="all",
        random_state=1,
    )

    n_obs = 5
    probe_features = sel._generate_probe_features(n_obs).round(3)

    # expected results
    expected_results = {
        "gaussian_probe_0": [4.873, -1.835, -1.585, -3.219, 2.596],
        "binary_probe_0": [0, 1, 0, 0, 1],
        "uniform_probe_0": [0.443, 0.23, 0.534, 0.914, 0.457],
        "gaussian_probe_1": [-6.905, 2.032, -0.321, 2.176, 2.805],
        "binary_probe_1": [1, 1, 1, 0, 1],
        "uniform_probe_1": [0.876, 0.895, 0.085, 0.039, 0.17],
    }
    expected_results_df = pd.DataFrame(expected_results)

    pd.testing.assert_frame_equal(
        probe_features, expected_results_df, check_dtype=False
    )


def test_generate_probe_features_normal():
    sel = ProbeFeatureSelection(
        estimator=DecisionTreeClassifier(),
        n_probes=2,
        distribution="normal",
        random_state=1,
    )

    n_obs = 3
    probe_features = sel._generate_probe_features(n_obs).round(3)

    # expected results
    expected_results = {
        "gaussian_probe_0": [4.873, -1.835, -1.585],
        "gaussian_probe_1": [-3.219, 2.596, -6.905],
    }
    expected_results_df = pd.DataFrame(expected_results)
    pd.testing.assert_frame_equal(
        probe_features, expected_results_df, check_dtype=False
    )


def test_generate_probe_features_binary():
    sel = ProbeFeatureSelection(
        estimator=DecisionTreeClassifier(),
        n_probes=3,
        distribution="binary",
        random_state=1,
    )

    n_obs = 2
    probe_features = sel._generate_probe_features(n_obs)

    # expected results
    expected_results = {
        "binary_probe_0": [1, 1],
        "binary_probe_1": [0, 0],
        "binary_probe_2": [1, 1],
    }
    expected_results_df = pd.DataFrame(expected_results)
    pd.testing.assert_frame_equal(
        probe_features, expected_results_df, check_dtype=False
    )


def test_generate_probe_features_uniform():
    sel = ProbeFeatureSelection(
        estimator=DecisionTreeClassifier(),
        n_probes=1,
        distribution="uniform",
        random_state=1,
    )

    n_obs = 3
    probe_features = sel._generate_probe_features(n_obs).round(3)

    # expected results
    expected_results = {"uniform_probe_0": [0.417, 0.72, 0.0]}
    expected_results_df = pd.DataFrame(expected_results)
    pd.testing.assert_frame_equal(
        probe_features, expected_results_df, check_dtype=False
    )


def test_get_features_to_drop():
    # 1 probe
    sel = ProbeFeatureSelection(estimator=LogisticRegression(), n_probes=1)
    sel.feature_importances_ = pd.Series(
        [11, 12, 9, 10], index=["var1", "var2", "var3", "probe"]
    )
    sel.probe_features_ = pd.DataFrame({"probe": [1, 1, 1, 1, 1]})
    sel.variables_ = ["var1", "var2", "var3"]
    assert sel._get_features_to_drop() == ["var3"]

    # 2 probes
    sel = ProbeFeatureSelection(estimator=LogisticRegression(), n_probes=2)
    sel.feature_importances_ = pd.Series(
        [11, 12, 10, 8.7, 10, 8],
        index=["var1", "var2", "var3", "var4", "probe1", "probe2"],
    )
    sel.probe_features_ = pd.DataFrame(
        {"probe1": [1, 1, 1, 1, 1], "probe2": [1, 1, 1, 1, 1]}
    )
    sel.variables_ = ["var1", "var2", "var3", "var4"]
    assert sel._get_features_to_drop() == ["var4"]


def test_cv_generator(df_test):
    X, y = df_test
    cv = StratifiedKFold(n_splits=3)

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

    # splitter passed as such
    sel = ProbeFeatureSelection(
        estimator=RandomForestClassifier(),
        distribution="normal",
        n_probes=2,
        scoring="recall",
        cv=cv,
        random_state=3,
        confirm_variables=False,
    )
    X_tr = sel.fit_transform(X, y)

    pd.testing.assert_frame_equal(
        sel.probe_features_.head().round(3),
        expected_probe_features_df,
        check_dtype=False,
    )
    assert sel.feature_importances_.round(2).equals(expected_feature_importances)
    assert sel.features_to_drop_ == ["var_2", "var_10"]
    pd.testing.assert_frame_equal(
        X_tr, X.drop(columns=["var_2", "var_10"]), check_dtype=False
    )

    # splitter passed as splits
    sel = ProbeFeatureSelection(
        estimator=RandomForestClassifier(),
        distribution="normal",
        n_probes=2,
        scoring="recall",
        cv=cv.split(X, y),
        random_state=3,
        confirm_variables=False,
    )
    X_tr = sel.fit_transform(X, y)

    pd.testing.assert_frame_equal(
        sel.probe_features_.head().round(3),
        expected_probe_features_df,
        check_dtype=False,
    )
    assert sel.feature_importances_.round(2).equals(expected_feature_importances)
    assert sel.features_to_drop_ == ["var_2", "var_10"]
    pd.testing.assert_frame_equal(
        X_tr, X.drop(columns=["var_2", "var_10"]), check_dtype=False
    )


def test_feature_importance_std(df_test):
    X, y = df_test

    sel = ProbeFeatureSelection(
        estimator=RandomForestClassifier(),
        distribution="normal",
        n_probes=2,
        scoring="recall",
        cv=3,
        random_state=3,
        confirm_variables=False,
    ).fit(X, y)

    # expected results
    expected_std = pd.Series(
        data=[
            0.0088,
            0.0002,
            0.0005,
            0.0007,
            0.0343,
            0.0013,
            0.0089,
            0.0551,
            0.0049,
            0.0123,
            0.0005,
            0.0005,
            0.0005,
            0.0006,
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
            "gaussian_probe_0",
            "gaussian_probe_1",
        ],
    )

    assert sel.feature_importances_std_.round(4).equals(expected_std)


def test_single_feature_importance_generation(df_test):
    X, y = df_test

    sel = ProbeFeatureSelection(
        estimator=RandomForestClassifier(n_estimators=3, random_state=3),
        distribution="normal",
        collective=False,
        n_probes=2,
        scoring="recall",
        cv=3,
        random_state=3,
        confirm_variables=False,
    ).fit(X, y)

    # expected results
    expected_ = pd.Series(
        data=[
            0.5867,
            0.5342,
            0.5042,
            0.4941,
            0.9456,
            0.5081,
            0.9294,
            0.9859,
            0.4799,
            0.8972,
            0.4476,
            0.5544,
            0.4799,
            0.5323,
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
            "gaussian_probe_0",
            "gaussian_probe_1",
        ],
    )

    assert sel.feature_importances_.round(4).equals(expected_)


def test_probe_feature_selector_with_groups(df_test_with_groups):
    X, y, groups = df_test_with_groups
    cv = GroupKFold(n_splits=3)
    cv_indices = cv.split(X=X, y=y, groups=groups)

    estimator = RandomForestRegressor(n_estimators=3, random_state=3)
    distribution = "normal"
    n_probes = 2
    scoring = "neg_mean_absolute_error"
    random_state = 3
    confirm_variables = True

    sel_expected = ProbeFeatureSelection(
        estimator=estimator,
        distribution=distribution,
        n_probes=n_probes,
        scoring=scoring,
        cv=cv_indices,
        random_state=random_state,
        confirm_variables=confirm_variables,
    )
    X_tr_expected = sel_expected.fit_transform(X, y)

    sel = ProbeFeatureSelection(
        estimator=estimator,
        distribution=distribution,
        n_probes=n_probes,
        scoring=scoring,
        cv=cv,
        groups=groups,
        random_state=random_state,
        confirm_variables=confirm_variables,
    )
    X_tr = sel.fit_transform(X, y)

    pd.testing.assert_frame_equal(X_tr_expected, X_tr)
