from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold

from feature_engine.selection.base_selection_functions import (
    _select_all_variables,
    _select_numerical_variables,
    find_correlated_features,
    find_feature_importance,
    get_feature_importances,
    single_feature_performance,
)
from tests.backend_helpers import make_series

DATA_VARTYPES = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
    "date_range": [datetime(2020, 2, 24, 0, minute) for minute in range(4)],
    "date_obj0": ["2020-02-24", "2020-02-25", "2020-02-26", "2020-02-27"],
}

# a is uncorrelated, b is correlated with c and d, and c and d only with b.
DATA_CORR = {
    "a": [1, -1, 0, 0, 0, 0, 2],
    "b": [0, 0, 1, -1, 1, -1, 0],
    "c": [0, 0, 1, -1, 0, 0, 1],
    "d": [0, 0, 0, 0, 1, -1, 1],
}

EXPECTED_MEAN = {
    "var_0": 0.5813469607144305,
    "var_1": 0.5325152703164752,
    "var_2": 0.5023573007759755,
    "var_3": 0.47596844810700234,
    "var_4": 0.9696712897767115,
    "var_5": 0.5078009005719849,
    "var_6": 0.966096275433625,
    "var_7": 0.9918595739378872,
    "var_8": 0.521667767752105,
    "var_9": 0.9476311088509884,
    "var_10": 0.4871054926777818,
    "var_11": 0.5180029642379039,
}

EXPECTED_STD = {
    "var_0": 0.0035430274728173775,
    "var_1": 0.0046697767238672565,
    "var_2": 0.023714708852568194,
    "var_3": 0.04219857610624132,
    "var_4": 0.010364079344188424,
    "var_5": 0.03203946151605523,
    "var_6": 0.0063709642968091335,
    "var_7": 0.0014579159677989356,
    "var_8": 0.027570153897628277,
    "var_9": 0.014363240810578251,
    "var_10": 0.020283618255582142,
    "var_11": 0.02707242215734807,
}


EXPECTED_IMPORTANCE_MEAN = {
    "var_0": 0.008110472647428566,
    "var_1": 0.004425867029009318,
    "var_2": 0.0014110527658847542,
    "var_3": 0.0,
    "var_4": 0.09519163119147249,
    "var_5": 0.005151538162222261,
    "var_6": 0.06819196935501609,
    "var_7": 0.7958920351591532,
    "var_8": 0.005514122712728161,
    "var_9": 0.006699116878609683,
    "var_10": 0.0006441114577744834,
    "var_11": 0.008768082640701015,
}

EXPECTED_IMPORTANCE_STD = {
    "var_0": 0.0044896289532760465,
    "var_1": 0.004400023300047043,
    "var_2": 0.0012779435856766451,
    "var_3": 0.0,
    "var_4": 0.15831114958148926,
    "var_5": 0.005860881861755391,
    "var_6": 0.0666073858478959,
    "var_7": 0.12339701768095801,
    "var_8": 0.0037045342320885986,
    "var_9": 0.0016309720229483885,
    "var_10": 0.0011156337706026607,
    "var_11": 0.005458498614789974,
}


def _split_target(make_df, data):
    X = make_df({k: v for k, v in data.items() if k != "target"})
    return X, make_series(make_df, data["target"])


def _pearson(x, y):
    return np.corrcoef(x, y)[0, 1]


@pytest.fixture(scope="module")
def data_with_groups():
    rng = np.random.default_rng(1)
    data = {f"var_{i}": rng.normal(size=100).tolist() for i in range(1, 6)}
    data["target"] = rng.integers(0, 100, size=100).tolist()
    groups = np.repeat(np.arange(1, 11), 10)
    rng.shuffle(groups)
    return data, groups.tolist()


@pytest.mark.parametrize(
    "variables, confirm_variables, exclude_datetime, expected",
    [
        (None, False, False, list(DATA_VARTYPES)),
        (None, False, True, ["Name", "City", "Age", "Marks"]),
        (["Name", "Age"], False, True, ["Name", "Age"]),
        (["Name", "Age", "Hola"], True, True, ["Name", "Age"]),
    ],
)
def test_select_all_variables(
    make_df, variables, confirm_variables, exclude_datetime, expected
):
    variables_ = _select_all_variables(
        make_df(DATA_VARTYPES), variables, confirm_variables, exclude_datetime
    )
    assert variables_ == expected


@pytest.mark.parametrize(
    "variables, confirm_variables, expected",
    [
        (None, False, ["Age", "Marks"]),
        (["Marks"], False, ["Marks"]),
        (["Marks", "Hola"], True, ["Marks"]),
    ],
)
def test_select_numerical_variables(make_df, variables, confirm_variables, expected):
    variables_ = _select_numerical_variables(
        make_df(DATA_VARTYPES), variables, confirm_variables
    )
    assert variables_ == expected


@pytest.mark.parametrize(
    "variables, expected",
    [
        (["a", "b", "c", "d"], ([{"b", "c", "d"}], ["c", "d"], {"b": {"c", "d"}})),
        (["a", "c", "b", "d"], ([{"c", "b"}], ["b"], {"c": {"b"}})),
    ],
)
def test_find_correlated_features(make_df, variables, expected):
    X = make_df(
        {
            "a": [1, -1, 0, 0, 0, 0],
            "b": [0, 0, 1, -1, 1, -1],
            "c": [0, 0, 1, -1, 0, 0],
            "d": [0, 0, 0, 0, 1, -1],
        }
    )
    assert find_correlated_features(X, variables, "pearson", 0.7) == expected


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall", _pearson])
def test_find_correlated_features_methods(make_df, method):
    X = make_df(DATA_CORR)
    groups, drop, dict_ = find_correlated_features(X, list(DATA_CORR), method, 0.5)
    assert groups == [{"b", "c", "d"}]
    assert drop == ["c", "d"]
    assert dict_ == {"b": {"c", "d"}}


@pytest.mark.parametrize(
    "method, expected",
    [
        ("pearson", ([{"a", "c"}, {"b", "d"}], ["c", "d"], {"a": {"c"}, "b": {"d"}})),
        ("spearman", ([{"b", "d"}], ["d"], {"b": {"d"}})),
        ("kendall", ([{"b", "d"}], ["d"], {"b": {"d"}})),
        (_pearson, ([{"a", "c"}, {"b", "d"}], ["c", "d"], {"a": {"c"}, "b": {"d"}})),
    ],
)
def test_find_correlated_features_skips_missing_values(make_df, method, expected):
    # each pair of variables is compared on the rows where neither is missing.
    data = {
        "a": [None, -1, 0, 0, 0, 0, 2],
        "b": [0, 0, 1, -1, 1, -1, None],
        "c": [0, 0, None, -1, 0, 0, 1],
        "d": [0, 0, 0, 0, 1, -1, 1],
    }
    X = make_df(data)
    assert find_correlated_features(X, list(data), method, 0.6) == expected


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_find_correlated_features_ignores_constant_variables(make_df, method):
    X = make_df({**DATA_CORR, "e": [1] * 7})
    groups, drop, dict_ = find_correlated_features(
        X, ["e", *DATA_CORR], method, 0.5
    )
    assert groups == [{"b", "c", "d"}]
    assert drop == ["c", "d"]
    assert dict_ == {"b": {"c", "d"}}


def test_find_correlated_features_with_integer_column_names():
    X = pd.DataFrame({i: values for i, values in enumerate(DATA_CORR.values())})
    groups, drop, dict_ = find_correlated_features(X, [0, 1, 2, 3], "pearson", 0.5)
    assert groups == [{1, 2, 3}]
    assert drop == [2, 3]
    assert dict_ == {1: {2, 3}}


def test_single_feature_performance(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    rf = RandomForestClassifier(n_estimators=5, random_state=1)

    mean_, std_ = single_feature_performance(
        X=X,
        y=y,
        variables=list(EXPECTED_MEAN),
        estimator=rf,
        cv=3,
        scoring="roc_auc",
    )
    assert mean_ == pytest.approx(EXPECTED_MEAN)
    assert std_ == pytest.approx(EXPECTED_STD)


@pytest.mark.parametrize("cv_type", ["splitter", "generator"])
def test_single_feature_performance_with_cv_splitter(
    make_df, data_classification, cv_type
):
    X, y = _split_target(make_df, data_classification)
    rf = RandomForestClassifier(n_estimators=5, random_state=1)
    cv = StratifiedKFold(n_splits=3)
    if cv_type == "generator":
        cv = cv.split(X, y)

    mean_, _ = single_feature_performance(
        X=X, y=y, variables=list(EXPECTED_MEAN), estimator=rf, cv=cv, scoring="roc_auc"
    )
    assert mean_ == pytest.approx(EXPECTED_MEAN)


@pytest.mark.parametrize("target_type", [list, np.array])
def test_single_feature_performance_with_list_and_array_target(
    make_df, data_classification, target_type
):
    X, _ = _split_target(make_df, data_classification)
    y = target_type(data_classification["target"])
    rf = RandomForestClassifier(n_estimators=5, random_state=1)

    mean_, _ = single_feature_performance(
        X=X, y=y, variables=["var_0", "var_4"], estimator=rf, cv=3, scoring="roc_auc"
    )
    assert mean_ == pytest.approx(
        {"var_0": EXPECTED_MEAN["var_0"], "var_4": EXPECTED_MEAN["var_4"]}
    )


def test_single_feature_performance_with_groups(make_df, data_with_groups):
    data, groups = data_with_groups
    X, y = _split_target(make_df, data)
    rf = RandomForestClassifier(n_estimators=5, random_state=1)
    variables = ["var_1", "var_2", "var_3", "var_4", "var_5"]
    cv = GroupKFold(n_splits=3)

    expected_mean_, expected_std_ = single_feature_performance(
        X=X,
        y=y,
        variables=variables,
        estimator=rf,
        cv=cv.split(X=X, y=y, groups=groups),
        scoring="neg_mean_absolute_error",
    )
    mean_, std_ = single_feature_performance(
        X=X,
        y=y,
        variables=variables,
        estimator=rf,
        cv=cv,
        scoring="neg_mean_absolute_error",
        groups=groups,
    )
    assert mean_ == expected_mean_
    assert std_ == expected_std_


@pytest.mark.parametrize("cv_type", ["splitter", "generator"])
def test_find_feature_importance(make_df, data_classification, cv_type):
    X, y = _split_target(make_df, data_classification)
    rf = RandomForestClassifier(n_estimators=3, random_state=3)
    cv = StratifiedKFold(n_splits=3)
    if cv_type == "generator":
        cv = cv.split(X, y)

    mean_, std_ = find_feature_importance(
        X=X, y=y, estimator=rf, cv=cv, scoring="recall"
    )

    importance_type = pd.Series if make_df is pd.DataFrame else dict
    assert isinstance(mean_, importance_type)
    assert isinstance(std_, importance_type)
    assert dict(mean_) == pytest.approx(EXPECTED_IMPORTANCE_MEAN)
    assert dict(std_) == pytest.approx(EXPECTED_IMPORTANCE_STD)


def test_find_feature_importance_with_groups(make_df, data_with_groups):
    data, groups = data_with_groups
    X, y = _split_target(make_df, data)
    rf = RandomForestClassifier(n_estimators=3, random_state=1)
    cv = GroupKFold(n_splits=3)

    expected_mean_, expected_std_ = find_feature_importance(
        X=X,
        y=y,
        estimator=rf,
        cv=cv.split(X=X, y=y, groups=groups),
        scoring="neg_mean_absolute_error",
    )
    mean_, std_ = find_feature_importance(
        X=X,
        y=y,
        estimator=rf,
        cv=cv,
        scoring="neg_mean_absolute_error",
        groups=groups,
    )
    assert dict(mean_) == dict(expected_mean_)
    assert dict(std_) == dict(expected_std_)


def test_find_feature_importance_returns_series_indexed_by_columns():
    X = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6], "b": [0.5, 0, 1, 1, 3, 2]})
    X.columns.name = "features"
    y = pd.Series([1.0, 2, 3, 4, 5, 6])

    mean_, std_ = find_feature_importance(
        X=X, y=y, estimator=Lasso(alpha=0.01), cv=2, scoring="r2"
    )
    assert list(mean_.index) == ["a", "b"]
    assert mean_.index.name == "features"
    assert std_.index.name == "features"


@pytest.mark.parametrize(
    "estimator, expected",
    [
        (LogisticRegression(), [0.728 ** (1 / 3), 1.0]),
        (Lasso(), [0.5, 2.0]),
    ],
)
def test_get_feature_importances(estimator, expected):
    if isinstance(estimator, Lasso):
        estimator.coef_ = np.array([-0.5, 2.0])
    else:
        estimator.coef_ = np.array([[0.6, 0.0], [0.0, 1.0], [0.8, 0.0]])
    assert get_feature_importances(estimator) == pytest.approx(expected)
