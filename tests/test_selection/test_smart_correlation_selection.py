import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold

from feature_engine.selection import SmartCorrelatedSelection
from tests.backend_helpers import frame_to_dict, make_series

# at threshold 0.506: var_b is correlated with var_c and var_d, and var_e with var_f.
VAR_CAR = {
    "var_a": [1, -1, 0, 0, 0, 0, 0, 0, 0],
    "var_b": [0, 0, 1, -1, 2, -2, 0, 0, 1],
    "var_c": [0, 0, 10, -10, 0, 0, 0, 0, 9],
    "var_d": [0, 0, 0, 0, 1, -1, 0, 0, 1],
    "var_e": [0, 0, 0, 0, 0, -1, 2, 3, 4],
    "var_f": [0, 0, 0, 0, 0, -1, 20, 30, 30],
}

WITH_NAN = {
    "var_a": [1, -1, 0, 0, 0, 0, 0, 0],
    "var_b": [None, 0, 1, -1, 2, -2, 0, 0],
    "var_c": [0, 0, 10, -10, 0, 0, 0, 0],
    "var_d": [0, 0, 0, 0, 1, -1, 0, 0],
    "var_e": [None, 0, 0, 0, 0, -1, 2, 3],
    "var_f": [0, 0, 0, 0, 0, -1, 20, 30],
}

# pearson, spearman and kendall find different groups at threshold 0.9.
CORR_METHODS = {
    "var_a": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "var_b": [1.2, 2.1, None, 4.3, 4.9, 6.2, 7.1, 7.8, 9.4, 30],
    "var_c": [2.0, 1, 3, 4, 6, 5, 8, 7, 10, 9],
    "var_d": [10.0, 9, 8, 7, 6, 5, 4, None, 2, 1],
    "var_e": [0.0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
}
CORR_METHODS_TARGET = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1]


@pytest.fixture(scope="module")
def data_single_group():
    # var_1, var_2 and var_4 are correlated, var_1 and var_2 above 0.8.
    X, y = make_classification(
        n_samples=1000,
        n_features=6,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )
    data = {f"var_{i}": X[:, i].tolist() for i in range(6)}
    data["target"] = y.tolist()
    return data


@pytest.fixture(scope="module")
def data_duplicated():
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )
    return {
        "var_0": X[:, 0].tolist(),
        "var_1": X[:, 1].tolist(),
        "var_0_duplicated": X[:, 0].tolist(),
        "var_1_duplicated": X[:, 1].tolist(),
        "target": y.tolist(),
    }


def _split_target(make_df, data):
    X = make_df({k: v for k, v in data.items() if k != "target"})
    return X, make_series(make_df, data["target"])


# init parameters
@pytest.mark.parametrize("threshold", [3, "0.1", -0, 2, 0, 1, -0.1, 1.5, None, [0.5]])
def test_error_if_threshold_not_float_between_0_and_1(threshold):
    msg = f"`threshold` must be a float between 0 and 1. Got {threshold} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        SmartCorrelatedSelection(threshold=threshold)


@pytest.mark.parametrize("missing_values", [2, "hola", False, None, ["raise"]])
def test_error_if_missing_values_not_permitted(missing_values):
    msg = (
        "missing_values takes only values 'raise' or 'ignore'. "
        f"Got {missing_values} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SmartCorrelatedSelection(missing_values=missing_values)


@pytest.mark.parametrize(
    "selection_method", [3, "hola", ["cardinality"], None, ("variance",)]
)
def test_error_if_selection_method_not_permitted(selection_method):
    msg = (
        "selection_method takes only values 'missing_values', 'cardinality', "
        "'variance', 'model_performance' or 'corr_with_target'. "
        f"Got {selection_method} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SmartCorrelatedSelection(selection_method=selection_method)


def test_error_if_model_performance_and_estimator_is_none():
    msg = (
        "Please provide an estimator, e.g., "
        "RandomForestClassifier or select another "
        "selection_method."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SmartCorrelatedSelection(selection_method="model_performance", estimator=None)


def test_error_if_selection_method_missing_values_and_missing_values_raise():
    msg = (
        "When `selection_method = 'missing_values'`, you need to set "
        "`missing_values` to `'ignore'`. Got raise instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SmartCorrelatedSelection(
            missing_values="raise", selection_method="missing_values"
        )


@pytest.mark.parametrize("confirm_variables", [2, "hola", [True], None])
def test_error_if_confirm_variables_not_bool(confirm_variables):
    msg = (
        "confirm_variables takes only values True and False. "
        f"Got {confirm_variables} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SmartCorrelatedSelection(confirm_variables=confirm_variables)


@pytest.mark.parametrize(
    "method, threshold, missing_values, selection_method, estimator, scoring, cv, "
    "groups, confirm_variables",
    [
        ("pearson", 0.8, "ignore", "missing_values", None, "roc_auc", 3, None, False),
        ("kendall", 0.5, "raise", "cardinality", None, "accuracy", 5, [1, 2], True),
        ("spearman", 0.4, "raise", "variance", None, "r2", KFold(), None, False),
        (
            np.corrcoef,
            0.9,
            "ignore",
            "model_performance",
            LogisticRegression(),
            "roc_auc",
            GroupKFold(),
            [1, 1, 2],
            False,
        ),
        ("pearson", 0.7, "raise", "corr_with_target", None, "roc_auc", 3, None, True),
    ],
)
def test_init_param_assignment(
    method,
    threshold,
    missing_values,
    selection_method,
    estimator,
    scoring,
    cv,
    groups,
    confirm_variables,
):
    sel = SmartCorrelatedSelection(
        method=method,
        threshold=threshold,
        missing_values=missing_values,
        selection_method=selection_method,
        estimator=estimator,
        scoring=scoring,
        cv=cv,
        groups=groups,
        confirm_variables=confirm_variables,
    )
    assert sel.method is method
    assert sel.threshold == threshold
    assert sel.missing_values == missing_values
    assert sel.selection_method == selection_method
    assert sel.estimator is estimator
    assert sel.scoring == scoring
    assert sel.cv is cv
    assert sel.groups == groups
    assert sel.confirm_variables is confirm_variables


# fit and transform
def test_selection_method_missing_values(make_df):
    # missing values: var_b and var_e 1, all other variables 0.
    X = make_df(WITH_NAN)
    sel = SmartCorrelatedSelection(threshold=0.4, selection_method="missing_values")
    Xt = sel.fit_transform(X)

    assert sel.variables_ == ["var_a", "var_b", "var_c", "var_d", "var_e", "var_f"]
    assert sel.correlated_feature_sets_ == [{"var_b", "var_c"}, {"var_e", "var_f"}]
    assert sel.correlated_feature_dict_ == {"var_c": {"var_b"}, "var_f": {"var_e"}}
    assert sel.features_to_drop_ == ["var_b", "var_e"]
    assert sel.feature_names_in_ == list(WITH_NAN)
    assert sel.n_features_in_ == 6
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        k: v for k, v in WITH_NAN.items() if k not in ["var_b", "var_e"]
    }


def test_selection_method_variance(make_df):
    # std: var_f 13.73, var_c 5.83, var_e 1.69, var_b 1.17, var_d 0.60, var_a 0.50.
    X = make_df(VAR_CAR)
    sel = SmartCorrelatedSelection(threshold=0.506, selection_method="variance")
    Xt = sel.fit_transform(X)

    assert sel.correlated_feature_sets_ == [{"var_e", "var_f"}, {"var_b", "var_c"}]
    assert sel.correlated_feature_dict_ == {"var_f": {"var_e"}, "var_c": {"var_b"}}
    assert sel.features_to_drop_ == ["var_e", "var_b"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        k: v for k, v in VAR_CAR.items() if k not in ["var_e", "var_b"]
    }


def test_selection_method_cardinality(make_df):
    # cardinality: var_b 5, var_e 5, var_c 4, var_f 4, var_a 3, var_d 3.
    X = make_df(VAR_CAR)
    sel = SmartCorrelatedSelection(threshold=0.506, selection_method="cardinality")
    Xt = sel.fit_transform(X)

    assert sel.correlated_feature_sets_ == [
        {"var_b", "var_c", "var_d"},
        {"var_e", "var_f"},
    ]
    assert sel.correlated_feature_dict_ == {
        "var_b": {"var_c", "var_d"},
        "var_e": {"var_f"},
    }
    assert sel.features_to_drop_ == ["var_c", "var_d", "var_f"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_a": VAR_CAR["var_a"],
        "var_b": VAR_CAR["var_b"],
        "var_e": VAR_CAR["var_e"],
    }


def test_cardinality_does_not_count_missing_values(make_df):
    # var_x has 4 values and var_y 5: missing values are not a category.
    X = make_df({"var_x": [1, 2, 3, 4, None, None], "var_y": [1, 2, 3, 4, 5, 5]})
    sel = SmartCorrelatedSelection(selection_method="cardinality").fit(X)

    assert sel.correlated_feature_dict_ == {"var_y": {"var_x"}}
    assert sel.features_to_drop_ == ["var_x"]


def test_selection_method_corr_with_target(make_df, data_single_group):
    X, y = _split_target(make_df, data_single_group)
    sel = SmartCorrelatedSelection(
        missing_values="raise", selection_method="corr_with_target"
    )
    Xt = sel.fit_transform(X, y)

    assert sel.correlated_feature_sets_ == [{"var_1", "var_2", "var_4"}]
    assert sel.correlated_feature_dict_ == {"var_2": {"var_1", "var_4"}}
    assert sel.features_to_drop_ == ["var_4", "var_1"]
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_0", "var_2", "var_3", "var_5"]


@pytest.mark.parametrize(
    "method, correlated_dict, features_to_drop",
    [
        ("pearson", {"var_a": {"var_c", "var_d"}}, ["var_d", "var_c"]),
        (
            "spearman",
            {"var_a": {"var_b", "var_c", "var_d"}},
            ["var_d", "var_b", "var_c"],
        ),
        ("kendall", {"var_d": {"var_a", "var_b"}}, ["var_a", "var_b"]),
    ],
)
def test_corr_with_target_uses_correlation_method(
    make_df, method, correlated_dict, features_to_drop
):
    # absolute correlation with the target:
    # pearson: var_a 0.782, var_d 0.763, var_c 0.711, var_b 0.468, var_e 0.408
    # spearman: var_a 0.782, var_d 0.779, var_b 0.730, var_c 0.711, var_e 0.408
    # kendall: var_d 0.671, var_a 0.669, var_b 0.629, var_c 0.609, var_e 0.408
    X = make_df(CORR_METHODS)
    y = make_series(make_df, CORR_METHODS_TARGET)
    sel = SmartCorrelatedSelection(
        method=method, threshold=0.9, selection_method="corr_with_target"
    )
    Xt = sel.fit_transform(X, y)

    assert sel.correlated_feature_dict_ == correlated_dict
    assert sel.features_to_drop_ == features_to_drop
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == [f for f in CORR_METHODS if f not in features_to_drop]


def test_corr_with_target_ties_keep_first_feature(make_df, data_duplicated):
    X, y = _split_target(make_df, data_duplicated)
    sel = SmartCorrelatedSelection(selection_method="corr_with_target")
    Xt = sel.fit_transform(X, y)

    assert sel.correlated_feature_sets_ == [
        {"var_1", "var_1_duplicated"},
        {"var_0", "var_0_duplicated"},
    ]
    assert sel.correlated_feature_dict_ == {
        "var_1": {"var_1_duplicated"},
        "var_0": {"var_0_duplicated"},
    }
    assert sel.features_to_drop_ == ["var_1_duplicated", "var_0_duplicated"]
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_0", "var_1"]


def test_model_performance_single_group(make_df, data_single_group):
    X, y = _split_target(make_df, data_single_group)
    sel = SmartCorrelatedSelection(
        missing_values="raise",
        selection_method="model_performance",
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        scoring="roc_auc",
        cv=3,
    )
    Xt = sel.fit_transform(X, y)

    assert sel.correlated_feature_sets_ == [{"var_1", "var_2"}]
    assert sel.correlated_feature_dict_ == {"var_2": {"var_1"}}
    assert sel.features_to_drop_ == ["var_1"]
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_0", "var_2", "var_3", "var_4", "var_5"]


def test_model_performance_two_groups(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = SmartCorrelatedSelection(
        missing_values="raise",
        selection_method="model_performance",
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        scoring="roc_auc",
        cv=3,
    )
    Xt = sel.fit_transform(X, y)

    assert sel.correlated_feature_sets_ == [
        {"var_0", "var_8"},
        {"var_4", "var_6", "var_7", "var_9"},
    ]
    assert sel.correlated_feature_dict_ == {
        "var_0": {"var_8"},
        "var_7": {"var_4", "var_6", "var_9"},
    }
    assert sel.features_to_drop_ == ["var_8", "var_4", "var_6", "var_9"]
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == [
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_5",
        "var_7",
        "var_10",
        "var_11",
    ]


def test_model_performance_ties_keep_first_feature_alphabetically(
    make_df, data_duplicated
):
    # duplicated features train equally good models.
    X, y = _split_target(make_df, data_duplicated)
    sel = SmartCorrelatedSelection(
        selection_method="model_performance",
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        cv=3,
    )
    Xt = sel.fit_transform(X, y)

    assert sel.correlated_feature_dict_ == {
        "var_0": {"var_0_duplicated"},
        "var_1": {"var_1_duplicated"},
    }
    assert sel.features_to_drop_ == ["var_0_duplicated", "var_1_duplicated"]
    assert list(Xt.columns) == ["var_0", "var_1"]


def test_model_performance_with_cv_generator(make_df, data_single_group):
    X, y = _split_target(make_df, data_single_group)
    sel = SmartCorrelatedSelection(
        selection_method="model_performance",
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        cv=KFold(3).split(np.zeros(len(y)), y),
    )
    Xt = sel.fit_transform(X, y)

    assert sel.features_to_drop_ == ["var_1"]
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_0", "var_2", "var_3", "var_4", "var_5"]


def test_model_performance_with_groups(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    groups = np.repeat(np.arange(10), 100)
    params = dict(
        selection_method="model_performance",
        estimator=LogisticRegression(),
        scoring="roc_auc",
    )
    sel = SmartCorrelatedSelection(cv=GroupKFold(3), groups=groups, **params)
    sel.fit(X, y)
    splits = GroupKFold(3).split(np.zeros(len(y)), y, groups)
    sel_splits = SmartCorrelatedSelection(cv=splits, **params).fit(X, y)

    assert sel.correlated_feature_dict_ == {
        "var_0": {"var_8"},
        "var_7": {"var_4", "var_6", "var_9"},
    }
    assert sel.features_to_drop_ == sel_splits.features_to_drop_


@pytest.mark.parametrize("target_type", [list, np.array])
@pytest.mark.parametrize("selection_method", ["corr_with_target", "model_performance"])
def test_target_as_list_or_array(
    make_df, data_single_group, target_type, selection_method
):
    X, _ = _split_target(make_df, data_single_group)
    y = target_type(data_single_group["target"])
    sel = SmartCorrelatedSelection(
        selection_method=selection_method,
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        cv=3,
    )
    sel.fit(X, y)

    expected = {
        "corr_with_target": {"var_2": {"var_1", "var_4"}},
        "model_performance": {"var_2": {"var_1"}},
    }
    assert sel.correlated_feature_dict_ == expected[selection_method]


@pytest.mark.parametrize("selection_method", ["model_performance", "corr_with_target"])
def test_error_if_y_is_none(make_df, data_single_group, selection_method):
    X, _ = _split_target(make_df, data_single_group)
    sel = SmartCorrelatedSelection(
        selection_method=selection_method, estimator=LogisticRegression()
    )
    msg = (
        f"When `selection_method = '{selection_method}'` y is needed to fit "
        "the transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        sel.fit(X)


@pytest.mark.parametrize("selection_method", ["variance", "corr_with_target"])
def test_error_if_missing_values_raise_and_nan(make_df, selection_method):
    X = make_df(WITH_NAN)
    y = make_series(make_df, [0, 1, 0, 1, 0, 1, 0, 1])
    sel = SmartCorrelatedSelection(
        selection_method=selection_method, missing_values="raise"
    )
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        sel.fit(X, y)


def test_error_if_missing_values_raise_and_inf(make_df):
    X = make_df({"var_a": [1.0, float("inf"), 3.0], "var_b": [1.0, 2.0, 3.0]})
    sel = SmartCorrelatedSelection(selection_method="variance", missing_values="raise")
    msg = (
        "Some of the variables to transform contain inf values. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        sel.fit(X)


def test_callable_method(make_df, data_classification):
    def abs_pearson(a, b):
        return abs(np.corrcoef(a, b)[0, 1])

    X, y = _split_target(make_df, data_classification)
    sel = SmartCorrelatedSelection(method=abs_pearson, selection_method="variance")
    Xt = sel.fit_transform(X, y)

    assert sel.correlated_feature_dict_ == {
        "var_7": {"var_4", "var_6", "var_9"},
        "var_8": {"var_0"},
    }
    assert sel.features_to_drop_ == ["var_6", "var_4", "var_9", "var_0"]
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == [
        "var_1",
        "var_2",
        "var_3",
        "var_5",
        "var_7",
        "var_8",
        "var_10",
        "var_11",
    ]


@pytest.mark.parametrize("confirm_variables", [False, True])
def test_variables_ignore_other_columns(
    make_df, data_classification, confirm_variables
):
    data = {k: v for k, v in data_classification.items() if k != "target"}
    data["cat"] = ["a", "b"] * 500
    variables = ["var_4", "var_6", "var_7", "var_0"]
    if confirm_variables is True:
        variables = variables + ["not_in_df"]
    X = make_df(data)
    sel = SmartCorrelatedSelection(
        variables=variables,
        selection_method="variance",
        confirm_variables=confirm_variables,
    )
    Xt = sel.fit_transform(X)

    assert sel.variables_ == ["var_4", "var_6", "var_7", "var_0"]
    assert sel.features_to_drop_ == ["var_6", "var_4"]
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == [c for c in data if c not in ["var_6", "var_4"]]


def test_fit_does_not_modify_input(make_df, data_single_group):
    X, y = _split_target(make_df, data_single_group)
    SmartCorrelatedSelection(selection_method="corr_with_target").fit(X, y)
    X_original, _ = _split_target(make_df, data_single_group)
    assert frame_to_dict(X) == frame_to_dict(X_original)


def test_error_if_method_not_permitted():
    # the error comes from pandas.
    X = pd.DataFrame(VAR_CAR)
    sel = SmartCorrelatedSelection(method="not_valid")
    msg = (
        "method must be either 'pearson', 'spearman', 'kendall', or a callable, "
        "'not_valid' was supplied"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        sel.fit(X)


def test_error_if_index_of_X_and_y_differ(data_single_group):
    X = pd.DataFrame(
        {k: v for k, v in data_single_group.items() if k != "target"},
        index=range(10, 1010),
    )
    y = pd.Series(data_single_group["target"])
    sel = SmartCorrelatedSelection(selection_method="corr_with_target")
    with pytest.raises(
        ValueError, match=re.escape("The indexes of X and y do not match.")
    ):
        sel.fit(X, y)


@pytest.mark.parametrize(
    "selection_method",
    ["missing_values", "cardinality", "variance", "corr_with_target"],
)
def test_integer_column_names(selection_method):
    X = pd.DataFrame({i: VAR_CAR[f"var_{c}"] for i, c in enumerate("abcdef")})
    y = pd.Series([0, 1, 0, 1, 0, 1, 1, 1, 1])
    sel = SmartCorrelatedSelection(threshold=0.506, selection_method=selection_method)
    Xt = sel.fit_transform(X, y)

    expected = {
        "missing_values": [2, 3, 5],
        "cardinality": [2, 3, 5],
        "variance": [4, 1],
        "corr_with_target": [2, 3, 4],
    }
    assert sel.features_to_drop_ == expected[selection_method]
    pd.testing.assert_frame_equal(Xt, X.drop(columns=expected[selection_method]))


def test_model_performance_with_integer_column_names(data_single_group):
    X = pd.DataFrame({i: data_single_group[f"var_{i}"] for i in range(6)})
    y = pd.Series(data_single_group["target"])
    sel = SmartCorrelatedSelection(
        selection_method="model_performance",
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        cv=3,
    )
    Xt = sel.fit_transform(X, y)

    assert sel.correlated_feature_dict_ == {2: {1}}
    pd.testing.assert_frame_equal(Xt, X[[0, 2, 3, 4, 5]])


def test_transform_keeps_pandas_index(data_single_group):
    X = pd.DataFrame(
        {k: v for k, v in data_single_group.items() if k != "target"},
        index=range(10, 1010),
    )
    y = pd.Series(data_single_group["target"], index=X.index)
    Xt = SmartCorrelatedSelection(selection_method="corr_with_target").fit_transform(
        X, y
    )
    pd.testing.assert_frame_equal(Xt, X[["var_0", "var_2", "var_3", "var_5"]])
