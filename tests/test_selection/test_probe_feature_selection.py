import re

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from feature_engine.selection import ProbeFeatureSelection
from tests.backend_helpers import frame_to_dict, make_series

VARIABLES = [f"var_{i}" for i in range(12)]

# importance of the variables and probes of the random forest in test_fit_transform
FOREST_IMPORTANCE = {
    "var_0": 0.03,
    "var_1": 0,
    "var_2": 0,
    "var_3": 0,
    "var_4": 0.26,
    "var_5": 0,
    "var_6": 0.22,
    "var_7": 0.33,
    "var_8": 0.02,
    "var_9": 0.12,
    "var_10": 0,
    "var_11": 0,
    "gaussian_probe_0": 0,
    "gaussian_probe_1": 0,
}

# the first rows of the probes drawn with random_state=3
GAUSSIAN_PROBES = {
    "gaussian_probe_0": [5.366, 1.31, 0.289, -5.59, -0.832],
    "gaussian_probe_1": [0.104, 3.396, -7.67, -0.807, -5.729],
}


def _split_target(make_df, data):
    X = make_df({k: v for k, v in data.items() if k != "target"})
    return X, make_series(make_df, data["target"])


def _forest_selector(cv=3):
    # the estimator is not seeded: fit() seeds numpy's global random generator.
    return ProbeFeatureSelection(
        estimator=RandomForestClassifier(),
        distribution="normal",
        n_probes=2,
        scoring="recall",
        cv=cv,
        random_state=3,
    )


# init parameters
@pytest.mark.parametrize("collective", [10, "string", 0.1, None, [True]])
def test_error_if_collective_not_bool(collective):
    msg = f"collective takes values True or False. Got {collective} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        ProbeFeatureSelection(estimator=DecisionTreeRegressor(), collective=collective)


@pytest.mark.parametrize(
    "distribution", [3, 2.22, None, "distribution", ["salud", "binary"], ("normal",)]
)
def test_error_if_distribution_not_permitted(distribution):
    msg = (
        "distribution takes values 'normal', 'binary', 'uniform', "
        f"'discrete_uniform', 'poisson', or 'all'. Got {distribution} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ProbeFeatureSelection(
            estimator=DecisionTreeRegressor(), distribution=distribution
        )


@pytest.mark.parametrize("n_probes", ["tree", [False, 2], 101.1, None])
def test_error_if_n_probes_not_int(n_probes):
    msg = f"n_probes must be an integer. Got {n_probes} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        ProbeFeatureSelection(estimator=DecisionTreeRegressor(), n_probes=n_probes)


@pytest.mark.parametrize(
    "n_categories", [0.1, "string", 0, -1, None, [10], [10, 1], {10}]
)
def test_error_if_n_categories_not_positive_int(n_categories):
    msg = f"n_categories must be a positive integer. Got {n_categories} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        ProbeFeatureSelection(
            estimator=DecisionTreeRegressor(), n_categories=n_categories
        )


@pytest.mark.parametrize("threshold", [1, "string", None, ["mean"]])
def test_error_if_threshold_not_permitted(threshold):
    msg = (
        "threshold takes values 'mean', 'max' or 'mean_plus_std'. "
        f"Got {threshold} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ProbeFeatureSelection(estimator=DecisionTreeRegressor(), threshold=threshold)


@pytest.mark.parametrize(
    "estimator, collective, scoring, n_probes, distribution, n_categories, "
    "threshold, cv, groups, random_state, confirm_variables",
    [
        (
            RandomForestClassifier(),
            True,
            "precision",
            3,
            "all",
            3,
            "mean",
            3,
            None,
            4,
            False,
        ),
        (
            Lasso(),
            False,
            "neg_mean_squared_error",
            7,
            "binary",
            7,
            "max",
            StratifiedKFold(),
            [1, 2],
            None,
            True,
        ),
        (
            DecisionTreeRegressor(),
            True,
            "r2",
            1,
            ["binary", "uniform"],
            10,
            "mean_plus_std",
            GroupKFold(),
            None,
            84,
            False,
        ),
    ],
)
def test_init_param_assignment(
    estimator,
    collective,
    scoring,
    n_probes,
    distribution,
    n_categories,
    threshold,
    cv,
    groups,
    random_state,
    confirm_variables,
):
    sel = ProbeFeatureSelection(
        estimator=estimator,
        collective=collective,
        scoring=scoring,
        n_probes=n_probes,
        distribution=distribution,
        n_categories=n_categories,
        threshold=threshold,
        cv=cv,
        groups=groups,
        random_state=random_state,
        confirm_variables=confirm_variables,
    )
    assert sel.estimator is estimator
    assert sel.collective is collective
    assert sel.scoring == scoring
    assert sel.n_probes == n_probes
    assert sel.distribution == distribution
    assert sel.n_categories == n_categories
    assert sel.threshold == threshold
    assert sel.cv is cv
    assert sel.groups == groups
    assert sel.random_state == random_state
    assert sel.confirm_variables is confirm_variables


# fit and transform
def test_fit_transform(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)

    sel = _forest_selector()
    Xt = sel.fit_transform(X, y)

    assert isinstance(sel.probe_features_, make_df)
    probes = frame_to_dict(sel.probe_features_)
    assert {k: [round(x, 3) for x in v[:5]] for k, v in probes.items()} == (
        GAUSSIAN_PROBES
    )
    assert len(probes["gaussian_probe_0"]) == 1000
    assert dict(sel.feature_importances_) == pytest.approx(
        FOREST_IMPORTANCE, abs=5e-3
    )
    assert sel.variables_ == VARIABLES
    assert sel.features_to_drop_ == ["var_2", "var_10"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        var: data_classification[var]
        for var in VARIABLES
        if var not in ["var_2", "var_10"]
    }


def test_feature_importances_are_series_for_pandas_and_dict_otherwise(
    make_df, data_classification
):
    X, y = _split_target(make_df, data_classification)
    sel = _forest_selector().fit(X, y)

    container = pd.Series if make_df is pd.DataFrame else dict
    assert isinstance(sel.feature_importances_, container)
    assert isinstance(sel.feature_importances_std_, container)


@pytest.mark.parametrize("cv_type", ["splitter", "splits"])
def test_cv_as_splitter_or_splits(make_df, data_classification, cv_type):
    X, y = _split_target(make_df, data_classification)
    cv = StratifiedKFold(n_splits=3)
    if cv_type == "splits":
        cv = cv.split(np.zeros(1000), data_classification["target"])

    sel = _forest_selector(cv=cv).fit(X, y)

    assert dict(sel.feature_importances_) == pytest.approx(
        FOREST_IMPORTANCE, abs=5e-3
    )
    assert sel.features_to_drop_ == ["var_2", "var_10"]


def test_feature_importance_std(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = _forest_selector().fit(X, y)

    assert dict(sel.feature_importances_std_) == pytest.approx(
        {
            "var_0": 0.0088,
            "var_1": 0.0002,
            "var_2": 0.0005,
            "var_3": 0.0007,
            "var_4": 0.0343,
            "var_5": 0.0013,
            "var_6": 0.0089,
            "var_7": 0.0551,
            "var_8": 0.0049,
            "var_9": 0.0123,
            "var_10": 0.0005,
            "var_11": 0.0005,
            "gaussian_probe_0": 0.0005,
            "gaussian_probe_1": 0.0006,
        },
        abs=5e-5,
    )


def test_single_feature_models(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)

    sel = ProbeFeatureSelection(
        estimator=RandomForestClassifier(n_estimators=3, random_state=3),
        distribution="normal",
        collective=False,
        n_probes=2,
        scoring="recall",
        cv=3,
        random_state=3,
    ).fit(X, y)

    assert dict(sel.feature_importances_) == pytest.approx(
        {
            "var_0": 0.5867,
            "var_1": 0.5342,
            "var_2": 0.5042,
            "var_3": 0.4941,
            "var_4": 0.9456,
            "var_5": 0.5081,
            "var_6": 0.9294,
            "var_7": 0.9859,
            "var_8": 0.4799,
            "var_9": 0.8972,
            "var_10": 0.4476,
            "var_11": 0.5544,
            "gaussian_probe_0": 0.4799,
            "gaussian_probe_1": 0.5323,
        },
        abs=5e-5,
    )
    assert sel.features_to_drop_ == ["var_2", "var_3", "var_8", "var_10"]


def test_single_feature_models_with_all_distributions(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)

    sel = ProbeFeatureSelection(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=0),
        collective=False,
        scoring="accuracy",
        distribution="all",
        threshold="mean_plus_std",
        cv=3,
        random_state=1,
    )
    Xt = sel.fit_transform(X, y)

    assert dict(sel.feature_importances_) == pytest.approx(
        {
            "var_0": 0.589,
            "var_1": 0.516,
            "var_2": 0.502,
            "var_3": 0.475,
            "var_4": 0.962,
            "var_5": 0.486,
            "var_6": 0.961,
            "var_7": 0.992,
            "var_8": 0.519,
            "var_9": 0.945,
            "var_10": 0.486,
            "var_11": 0.518,
            "gaussian_probe_0": 0.49,
            "binary_probe_0": 0.513,
            "uniform_probe_0": 0.508,
            "discrete_uniform_probe_0": 0.463,
            "poisson_probe_0": 0.499,
        },
        abs=5e-4,
    )
    kept = ["var_0", "var_4", "var_6", "var_7", "var_9"]
    assert sel.features_to_drop_ == [var for var in VARIABLES if var not in kept]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {var: data_classification[var] for var in kept}


def test_examines_only_the_indicated_variables(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)

    sel = ProbeFeatureSelection(
        estimator=LogisticRegression(),
        variables=["var_0", "var_2", "var_4", "var_7", "var_10"],
        distribution=["binary", "uniform"],
        n_probes=2,
        threshold="max",
        cv=3,
        random_state=1,
    )
    Xt = sel.fit_transform(X, y)

    assert list(sel.probe_features_.columns) == [
        "binary_probe_0",
        "binary_probe_1",
        "uniform_probe_0",
        "uniform_probe_1",
    ]
    assert dict(sel.feature_importances_) == pytest.approx(
        {
            "var_0": 2.0089,
            "var_2": 0.097,
            "var_4": 0.3998,
            "var_7": 2.6832,
            "var_10": 0.1582,
            "binary_probe_0": 0.268,
            "binary_probe_1": 0.3385,
            "uniform_probe_0": 0.0804,
            "uniform_probe_1": 0.2257,
        },
        abs=5e-5,
    )
    assert sel.features_to_drop_ == ["var_2", "var_10"]
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == [v for v in VARIABLES if v not in ["var_2", "var_10"]]


def test_non_numerical_variables_are_not_examined(make_df, data_classification):
    data = {**data_classification, "city": ["London", "Paris"] * 500}
    X, y = _split_target(make_df, data)

    sel = _forest_selector()
    Xt = sel.fit_transform(X, y)

    assert sel.variables_ == VARIABLES
    assert sel.features_to_drop_ == ["var_2", "var_10"]
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == [
        v for v in VARIABLES + ["city"] if v not in ["var_2", "var_10"]
    ]


@pytest.mark.parametrize("to_target", [list, np.array])
def test_target_as_list_or_array(make_df, data_classification, to_target):
    X, _ = _split_target(make_df, data_classification)
    y = to_target(data_classification["target"])

    sel = _forest_selector().fit(X, y)

    assert dict(sel.feature_importances_) == pytest.approx(
        FOREST_IMPORTANCE, abs=5e-3
    )
    assert sel.features_to_drop_ == ["var_2", "var_10"]


def test_groups_give_the_same_result_as_the_group_splits(
    make_df, data_classification
):
    X, y = _split_target(make_df, data_classification)
    groups = np.repeat(np.arange(10), 100)
    params = dict(
        estimator=RandomForestRegressor(n_estimators=3, random_state=3),
        n_probes=2,
        scoring="neg_mean_absolute_error",
        random_state=3,
    )
    splits = GroupKFold(n_splits=3).split(np.zeros(1000), groups=groups)
    sel_splits = ProbeFeatureSelection(cv=splits, **params).fit(X, y)

    sel = ProbeFeatureSelection(cv=GroupKFold(n_splits=3), groups=groups, **params)
    Xt = sel.fit_transform(X, y)

    assert dict(sel.feature_importances_) == dict(sel_splits.feature_importances_)
    assert sel.features_to_drop_ == sel_splits.features_to_drop_
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == frame_to_dict(sel_splits.transform(X))


def test_integer_column_names():
    # pandas only: polars does not allow integer column names.
    X = pd.DataFrame({0: [0.0, 0.0, 1.0, 1.0] * 25, 1: [0.0, 1.0] * 50})
    y = pd.Series([0, 1] * 50)

    sel = ProbeFeatureSelection(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=0),
        collective=False,
        cv=2,
        random_state=2,
    )
    Xt = sel.fit_transform(X, y)

    assert sel.variables_ == [0, 1]
    pd.testing.assert_series_equal(
        sel.feature_importances_,
        pd.Series([0.5, 1.0, 0.5124], index=[0, 1, "gaussian_probe_0"]),
    )
    assert sel.features_to_drop_ == [0]
    pd.testing.assert_frame_equal(Xt, X[[1]])


def test_fit_does_not_change_the_index_of_the_input(data_classification):
    # pandas only: the probes are aligned with X by position, not by index.
    X = pd.DataFrame({var: data_classification[var] for var in VARIABLES})
    X.index = np.arange(1000)[::-1] * 3 + 10
    y = pd.Series(data_classification["target"], index=X.index)
    X_original = X.copy()

    sel = _forest_selector()
    Xt = sel.fit_transform(X, y)

    pd.testing.assert_frame_equal(X, X_original)
    assert sel.probe_features_.index.equals(pd.RangeIndex(1000))
    assert sel.features_to_drop_ == ["var_2", "var_10"]
    pd.testing.assert_frame_equal(Xt, X.drop(columns=["var_2", "var_10"]))


def _round_probes(probes):
    return {name: np.round(values, 3).tolist() for name, values in probes.items()}


def test_generate_some_probe_features():
    sel = ProbeFeatureSelection(
        estimator=DecisionTreeClassifier(),
        n_probes=2,
        distribution=["normal", "binary", "uniform"],
        random_state=1,
    )

    assert _round_probes(sel._generate_probe_features(5)) == {
        "gaussian_probe_0": [4.873, -1.835, -1.585, -3.219, 2.596],
        "gaussian_probe_1": [-6.905, 5.234, -2.284, 0.957, -0.748],
        "binary_probe_0": [1, 0, 0, 0, 1],
        "binary_probe_1": [1, 1, 1, 1, 0],
        "uniform_probe_0": [0.198, 0.801, 0.968, 0.313, 0.692],
        "uniform_probe_1": [0.876, 0.895, 0.085, 0.039, 0.170],
    }


def test_generate_all_probe_features():
    sel = ProbeFeatureSelection(
        estimator=DecisionTreeClassifier(),
        n_probes=1,
        distribution="all",
        random_state=1,
    )

    assert _round_probes(sel._generate_probe_features(5)) == {
        "gaussian_probe_0": [4.873, -1.835, -1.585, -3.219, 2.596],
        "binary_probe_0": [0, 1, 0, 0, 1],
        "uniform_probe_0": [0.443, 0.230, 0.534, 0.914, 0.457],
        "discrete_uniform_probe_0": [1, 7, 0, 6, 9],
        "poisson_probe_0": [19, 15, 2, 14, 10],
    }


@pytest.mark.parametrize(
    "distribution, n_probes, n_obs, expected",
    [
        (
            "normal",
            2,
            3,
            {
                "gaussian_probe_0": [4.873, -1.835, -1.585],
                "gaussian_probe_1": [-3.219, 2.596, -6.905],
            },
        ),
        (
            "binary",
            3,
            2,
            {
                "binary_probe_0": [1, 1],
                "binary_probe_1": [0, 0],
                "binary_probe_2": [1, 1],
            },
        ),
        ("uniform", 1, 3, {"uniform_probe_0": [0.417, 0.72, 0.0]}),
        (
            ["discrete_uniform", "poisson"],
            2,
            3,
            {
                "discrete_uniform_probe_0": [5, 8, 9],
                "discrete_uniform_probe_1": [5, 0, 0],
                "poisson_probe_0": [9, 14, 10],
                "poisson_probe_1": [7, 15, 13],
            },
        ),
    ],
)
def test_generate_probe_features_per_distribution(
    distribution, n_probes, n_obs, expected
):
    sel = ProbeFeatureSelection(
        estimator=DecisionTreeClassifier(),
        n_probes=n_probes,
        distribution=distribution,
        random_state=1,
    )

    assert _round_probes(sel._generate_probe_features(n_obs)) == expected


@pytest.mark.parametrize(
    "n_categories, max_discrete, max_poisson", [(5, 4, 12), (10, 9, 19), (3, 2, 8)]
)
def test_generate_features_n_categories(n_categories, max_discrete, max_poisson):
    sel = ProbeFeatureSelection(
        estimator=DecisionTreeClassifier(),
        n_probes=1,
        distribution=["discrete_uniform", "poisson"],
        n_categories=n_categories,
        random_state=1,
    )

    probes = sel._generate_probe_features(100)
    assert probes["discrete_uniform_probe_0"].max() == max_discrete
    assert probes["poisson_probe_0"].max() == max_poisson


@pytest.mark.parametrize("container", [dict, pd.Series])
@pytest.mark.parametrize("threshold", ["mean", "max", "mean_plus_std"])
def test_get_features_to_drop_with_one_probe(container, threshold):
    sel = ProbeFeatureSelection(estimator=LogisticRegression(), threshold=threshold)
    sel.feature_importances_ = container(
        {"var1": 11, "var2": 12, "var3": 9, "probe": 10}
    )
    sel.variables_ = ["var1", "var2", "var3"]
    assert sel._get_features_to_drop(["probe"]) == ["var3"]


@pytest.mark.parametrize("container", [dict, pd.Series])
@pytest.mark.parametrize(
    "threshold, features_to_drop",
    [
        ("mean", ["var4"]),
        ("max", ["var3", "var4"]),
        ("mean_plus_std", ["var1", "var3", "var4"]),
    ],
)
def test_get_features_to_drop_with_many_probes(
    container, threshold, features_to_drop
):
    sel = ProbeFeatureSelection(estimator=LogisticRegression(), threshold=threshold)
    sel.feature_importances_ = container(
        {"var1": 11, "var2": 20, "var3": 9.9, "var4": 8.7, "probe1": 10, "probe2": 8}
    )
    sel.variables_ = ["var1", "var2", "var3", "var4"]
    assert sel._get_features_to_drop(["probe1", "probe2"]) == features_to_drop
