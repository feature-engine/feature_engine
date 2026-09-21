import re

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from feature_engine.selection.base_recursive_selector import BaseRecursiveSelector
from tests.backend_helpers import make_series

VARIABLES = ["var_0", "var_4", "var_7"]


def _split_target(make_df, data):
    X = make_df({k: v for k, v in data.items() if k != "target"})
    return X, make_series(make_df, data["target"])


# init parameters
@pytest.mark.parametrize("threshold", [None, [0.1], "a_string", {"a": 1}])
def test_error_if_threshold_not_number(threshold):
    msg = f"threshold must be an integer or a float. Got {threshold} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        BaseRecursiveSelector(RandomForestClassifier(), threshold=threshold)


@pytest.mark.parametrize(
    "estimator, scoring, cv, groups, threshold, confirm_variables",
    [
        (RandomForestClassifier(), "roc_auc", 3, None, 0.01, False),
        (LogisticRegression(), "accuracy", StratifiedKFold(), [1, 2], 1, True),
        (KNeighborsClassifier(), "r2", GroupKFold(), None, -0.5, False),
    ],
)
def test_init_param_assignment(
    estimator, scoring, cv, groups, threshold, confirm_variables
):
    selector = BaseRecursiveSelector(
        estimator,
        scoring=scoring,
        cv=cv,
        groups=groups,
        threshold=threshold,
        confirm_variables=confirm_variables,
    )
    assert selector.estimator is estimator
    assert selector.scoring == scoring
    assert selector.cv is cv
    assert selector.groups == groups
    assert selector.threshold == threshold
    assert selector.confirm_variables is confirm_variables


# fit
@pytest.mark.parametrize("cv_type", ["int", "splitter", "generator"])
def test_fit_with_feature_importances(make_df, data_classification, cv_type):
    X, y = _split_target(make_df, data_classification)
    cv = {"int": 3, "splitter": StratifiedKFold(n_splits=3)}.get(cv_type)
    if cv_type == "generator":
        cv = StratifiedKFold(n_splits=3).split(X, y)

    selector = BaseRecursiveSelector(
        RandomForestClassifier(n_estimators=5, random_state=1),
        scoring="roc_auc",
        cv=cv,
        variables=VARIABLES,
    )
    selector.fit(X, y)

    assert selector.variables_ == VARIABLES
    assert selector.feature_names_in_ == [f"var_{i}" for i in range(12)]
    assert selector.n_features_in_ == 12
    assert selector.initial_model_performance_ == pytest.approx(0.9947395643178775)
    assert dict(selector.feature_importances_) == pytest.approx(
        {
            "var_0": 0.05016049596989658,
            "var_4": 0.5704427408209541,
            "var_7": 0.37939676320914945,
        }
    )
    assert dict(selector.feature_importances_std_) == pytest.approx(
        {
            "var_0": 0.02070511883333705,
            "var_4": 0.031350384984021235,
            "var_7": 0.03942210400299374,
        }
    )


def test_fit_with_coefficients(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    selector = BaseRecursiveSelector(
        LogisticRegression(), scoring="roc_auc", cv=3, variables=VARIABLES
    )
    selector.fit(X, y)

    assert selector.initial_model_performance_ == pytest.approx(0.996732746280939)
    assert dict(selector.feature_importances_) == pytest.approx(
        {
            "var_0": 2.0421118575507067,
            "var_4": 0.36861802531867144,
            "var_7": 2.68269483714549,
        }
    )
    assert dict(selector.feature_importances_std_) == pytest.approx(
        {
            "var_0": 0.08627973281236784,
            "var_4": 0.12490483002792199,
            "var_7": 0.13862536091514688,
        }
    )


def test_fit_with_permutation_importance(make_df, data_classification):
    # KNN has no coef_ or feature_importances_, so importance comes from
    # permutation_importance.
    X, y = _split_target(make_df, data_classification)
    selector = BaseRecursiveSelector(
        KNeighborsClassifier(), scoring="accuracy", cv=3, variables=VARIABLES
    )
    selector.fit(X, y)

    assert selector.initial_model_performance_ == pytest.approx(0.991997986009962)
    assert dict(selector.feature_importances_) == pytest.approx(
        {
            "var_0": 0.0050000000000000044,
            "var_4": 0.0753333333333333,
            "var_7": 0.42766666666666664,
        }
    )
    assert dict(selector.feature_importances_std_) == pytest.approx(
        {
            "var_0": 0.0010000000000000009,
            "var_4": 0.0005773502691896263,
            "var_7": 0.0037859388972001223,
        }
    )


def test_feature_importances_type(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    selector = BaseRecursiveSelector(
        LogisticRegression(), scoring="roc_auc", cv=3, variables=VARIABLES
    )
    selector.fit(X, y)

    importance_type = pd.Series if make_df is pd.DataFrame else dict
    assert isinstance(selector.feature_importances_, importance_type)
    assert isinstance(selector.feature_importances_std_, importance_type)
    assert list(selector.feature_importances_.keys()) == VARIABLES


def test_fit_returns_narwhals_frame_and_target(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    selector = BaseRecursiveSelector(
        LogisticRegression(), scoring="roc_auc", cv=3, variables=VARIABLES
    )
    nw_X, y_ = selector.fit(X, y)

    assert isinstance(nw_X, nw.DataFrame)
    assert nw_X.to_native() is X
    assert isinstance(y_, type(y))


def test_fit_finds_numerical_variables(make_df, data_classification):
    X, y = _split_target(
        make_df,
        {
            "var_0": data_classification["var_0"],
            "cat": ["a", "b"] * 500,
            "var_4": data_classification["var_4"],
            "target": data_classification["target"],
        },
    )
    selector = BaseRecursiveSelector(LogisticRegression(), scoring="roc_auc", cv=3)
    selector.fit(X, y)

    assert selector.variables_ == ["var_0", "var_4"]
    assert selector.feature_names_in_ == ["var_0", "cat", "var_4"]
    assert list(selector.feature_importances_.keys()) == ["var_0", "var_4"]


def test_fit_with_confirm_variables(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    selector = BaseRecursiveSelector(
        LogisticRegression(),
        scoring="roc_auc",
        cv=3,
        variables=["var_0", "var_4", "Hola"],
        confirm_variables=True,
    )
    selector.fit(X, y)

    assert selector.variables_ == ["var_0", "var_4"]
    assert list(selector.feature_importances_.keys()) == ["var_0", "var_4"]


@pytest.mark.parametrize("target_type", [list, np.array])
def test_fit_with_list_and_array_target(make_df, data_classification, target_type):
    X, _ = _split_target(make_df, data_classification)
    y = target_type(data_classification["target"])
    selector = BaseRecursiveSelector(
        LogisticRegression(), scoring="roc_auc", cv=3, variables=VARIABLES
    )
    selector.fit(X, y)

    assert selector.initial_model_performance_ == pytest.approx(0.996732746280939)


def test_error_if_only_one_variable(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    selector = BaseRecursiveSelector(
        LogisticRegression(), scoring="roc_auc", cv=3, variables=["var_0"]
    )
    msg = (
        "The selector needs at least 2 or more variables to select from. "
        "Got only 1 variable: ['var_0']."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        selector.fit(X, y)


def test_feature_importances_are_pandas_series_indexed_by_variables(
    data_classification,
):
    X, y = _split_target(pd.DataFrame, data_classification)
    selector = BaseRecursiveSelector(
        LogisticRegression(), scoring="roc_auc", cv=3, variables=VARIABLES
    )
    selector.fit(X, y)

    pd.testing.assert_series_equal(
        selector.feature_importances_,
        pd.Series(
            [2.0421118575507067, 0.36861802531867144, 2.68269483714549],
            index=VARIABLES,
        ),
    )


def test_fit_with_integer_column_names(data_classification):
    X, y = _split_target(pd.DataFrame, data_classification)
    X.columns = list(range(12))
    selector = BaseRecursiveSelector(
        LogisticRegression(), scoring="roc_auc", cv=3, variables=[0, 4, 7]
    )
    selector.fit(X, y)

    assert selector.variables_ == [0, 4, 7]
    assert selector.feature_names_in_ == list(range(12))
    assert dict(selector.feature_importances_) == pytest.approx(
        {0: 2.0421118575507067, 4: 0.36861802531867144, 7: 2.68269483714549}
    )
