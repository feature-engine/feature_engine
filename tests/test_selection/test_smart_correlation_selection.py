import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold

from feature_engine.selection import SmartCorrelatedSelection
from tests.estimator_checks.init_params_allowed_values_checks import (
    check_error_param_confirm_variables,
    check_error_param_missing_values,
)


@pytest.fixture(scope="module")
def df_single():
    # create array with 4 correlated features and 2 independent ones
    X, y = make_classification(
        n_samples=1000,
        n_features=6,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )

    # trasform array into pandas df
    colnames = ["var_" + str(i) for i in range(6)]
    X = pd.DataFrame(X, columns=colnames)

    return X, y


@pytest.fixture(scope="module")
def df_var_car():
    # create dataframe with known variance and cardinality:

    # at threshold 0.506

    # a=> no correlated
    # b => correlated with c and d
    # c => correlated with b
    # d => correlated with b
    # e => correlated with f
    # f => correlated with e

    X = pd.DataFrame(
        {
            "var_a": [1, -1, 0, 0, 0, 0, 0, 0, 0],
            "var_b": [0, 0, 1, -1, 2, -2, 0, 0, 1],
            "var_c": [0, 0, 10, -10, 0, 0, 0, 0, 9],
            "var_d": [0, 0, 0, 0, 1, -1, 0, 0, 1],
            "var_e": [0, 0, 0, 0, 0, -1, 2, 3, 4],
            "var_f": [0, 0, 0, 0, 0, -1, 20, 30, 30],
        }
    )

    return X


@pytest.fixture(scope="module")
def df_nan():
    X = pd.DataFrame(
        {
            "var_a": [1, -1, 0, 0, 0, 0, 0, 0],
            "var_b": [np.nan, 0, 1, -1, 2, -2, 0, 0],
            "var_c": [0, 0, 10, -10, 0, 0, 0, 0],
            "var_d": [0, 0, 0, 0, 1, -1, 0, 0],
            "var_e": [np.nan, 0, 0, 0, 0, -1, 2, 3],
            "var_f": [0, 0, 0, 0, 0, -1, 20, 30],
        }
    )
    return X


_input_params = [
    (None, "pearson", 0.8, "ignore", "missing_values", False),
    ("var1", "kendall", 0.5, "raise", "cardinality", True),
    (["var1", "var2"], "spearman", 0.4, "raise", "variance", False),
]


@pytest.mark.parametrize(
    "_variables, _method, _threshold, _missing_values, _sel_method, _confirm_vars",
    _input_params,
)
def test_input_params_assignment(
    _variables, _method, _threshold, _missing_values, _sel_method, _confirm_vars
):
    sel = SmartCorrelatedSelection(
        variables=_variables,
        method=_method,
        threshold=_threshold,
        missing_values=_missing_values,
        selection_method=_sel_method,
        confirm_variables=_confirm_vars,
    )

    assert sel.variables == _variables
    assert sel.method == _method
    assert sel.threshold == _threshold
    assert sel.missing_values == _missing_values
    assert sel.selection_method == _sel_method
    assert sel.confirm_variables == _confirm_vars


@pytest.mark.parametrize("_threshold", [3, "0.1", -0, 2, 0, 3, 1])
def test_raises_error_when_threshold_not_permitted(_threshold):
    msg = f"`threshold` must be a float between 0 and 1. Got {_threshold} instead."
    with pytest.raises(ValueError) as record:
        SmartCorrelatedSelection(threshold=_threshold)
    assert record.value.args[0] == msg


@pytest.mark.parametrize("_method", [3, "hola", ["cardinality"]])
def test_raises_error_when_selection_method_not_permitted(_method):
    msg = (
        "selection_method takes only values 'missing_values', 'cardinality', "
        f"'variance' or 'model_performance'. Got {_method} instead."
    )
    with pytest.raises(ValueError) as record:
        SmartCorrelatedSelection(selection_method=_method)
    assert record.value.args[0] == msg


def test_raises_error_when_selection_method_performance_and_estimator_none():
    msg = (
        "Please provide an estimator, e.g., "
        "RandomForestClassifier or select another "
        "selection_method."
    )
    with pytest.raises(ValueError) as record:
        SmartCorrelatedSelection(selection_method="model_performance", estimator=None)
    assert record.value.args[0] == msg


def test_error_param_missing_values():
    check_error_param_missing_values(SmartCorrelatedSelection())


def test_error_param_confirm_variables():
    check_error_param_confirm_variables(SmartCorrelatedSelection())


def test_model_performance_single_corr_group(df_single):
    X, y = df_single

    transformer = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="model_performance",
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        scoring="roc_auc",
        cv=3,
    )

    Xt = transformer.fit_transform(X, y)

    # expected result
    df = X[["var_0", "var_2", "var_3", "var_4", "var_5"]].copy()

    # test init params
    assert transformer.scoring == "roc_auc"
    assert transformer.cv == 3

    # test fit attrs
    assert transformer.correlated_feature_sets_ == [{"var_1", "var_2"}]
    assert transformer.features_to_drop_ == ["var_1"]
    assert transformer.correlated_feature_dict_ == {"var_2": {"var_1"}}
    # test transform output
    pd.testing.assert_frame_equal(Xt, df)


def test_model_performance_2_correlated_groups(df_test):
    X, y = df_test

    transformer = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="model_performance",
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        scoring="roc_auc",
        cv=3,
    )

    Xt = transformer.fit_transform(X, y)

    # expected result
    df = X[
        ["var_0", "var_1", "var_2", "var_3", "var_5", "var_7", "var_10", "var_11"]
    ].copy()

    # test fit attrs
    assert transformer.correlated_feature_sets_ == [
        {"var_0", "var_8"},
        {"var_4", "var_6", "var_7", "var_9"},
    ]
    assert transformer.features_to_drop_ == [
        "var_8",
        "var_4",
        "var_6",
        "var_9",
    ]
    assert transformer.correlated_feature_dict_ == {
        "var_0": {"var_8"},
        "var_7": {"var_4", "var_6", "var_9"},
    }
    # test transform output
    pd.testing.assert_frame_equal(Xt, df)


def test_cv_generator(df_single):
    X, y = df_single
    cv = KFold(3)

    transformer = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="model_performance",
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        scoring="roc_auc",
        cv=cv.split(X, y),
    )

    Xt = transformer.fit_transform(X, y)

    df = X[["var_0", "var_2", "var_3", "var_4", "var_5"]].copy()
    pd.testing.assert_frame_equal(Xt, df)


def test_error_if_select_model_performance_and_y_is_none(df_single):
    X, y = df_single

    transformer = SmartCorrelatedSelection(
        selection_method="model_performance",
        estimator=RandomForestClassifier(n_estimators=10, random_state=1),
        scoring="roc_auc",
    )
    msg = (
        "When `selection_method = 'model_performance'` y is needed to fit "
        "the transformer."
    )
    with pytest.raises(ValueError) as record:
        transformer.fit(X)
    assert record.value.args[0] == msg


def test_selection_method_variance(df_var_car):
    X = df_var_car

    # std of each variable:
    # var_f  13.727507
    # var_c  5.830952
    # var_e  1.691482
    # var_b  1.166667
    # var_d  0.600925
    # var_a  0.500000

    transformer = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.506,
        missing_values="raise",
        selection_method="variance",
        estimator=None,
    )

    Xt = transformer.fit_transform(X)

    assert transformer.features_to_drop_ == ["var_e", "var_b"]
    assert transformer.correlated_feature_dict_ == {
        "var_f": {"var_e"},
        "var_c": {"var_b"},
    }
    # test transform output
    pd.testing.assert_frame_equal(Xt, X.drop(["var_e", "var_b"], axis=1))


def test_selection_method_cardinality(df_var_car):
    X = df_var_car

    # cardinality of variables:
    # var_b   5
    # var_e   5
    # var_c   4
    # var_f   4
    # var_a   3
    # var_d   3

    transformer = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.506,
        missing_values="raise",
        selection_method="cardinality",
        estimator=None,
    )

    Xt = transformer.fit_transform(X)

    assert transformer.features_to_drop_ == ["var_c", "var_d", "var_f"]
    assert transformer.correlated_feature_dict_ == {
        "var_b": {"var_c", "var_d"},
        "var_e": {"var_f"},
    }
    # test transform output
    pd.testing.assert_frame_equal(Xt, X.drop(["var_c", "var_d", "var_f"], axis=1))


def test_selection_method_missing_values(df_nan):
    X = df_nan

    # expected order of the variables:
    # var_a    0
    # var_c    0
    # var_d    0
    # var_f    0
    # var_b    1
    # var_e    1

    transformer = SmartCorrelatedSelection(
        variables=None,
        method="pearson",
        threshold=0.4,
        missing_values="ignore",
        selection_method="missing_values",
        estimator=None,
    )

    Xt = transformer.fit_transform(X)

    assert transformer.features_to_drop_ == ["var_b", "var_e"]
    assert transformer.correlated_feature_dict_ == {
        "var_c": {"var_b"},
        "var_f": {"var_e"},
    }
    # test transform output
    pd.testing.assert_frame_equal(Xt, X.drop(["var_b", "var_e"], axis=1))


def test_error_when_selection_method_missing_values_and_missing_values_raise(df_na):
    msg = (
        "When `selection_method = 'missing_values'`, you need to set "
        "`missing_values` to `'ignore'`. Got raise instead."
    )
    with pytest.raises(ValueError) as record:
        SmartCorrelatedSelection(
            missing_values="raise",
            selection_method="missing_values",
        )
    assert record.value.args[0] == msg


def test_raises_error_when_method_not_permitted(df_var_car):
    X = df_var_car
    transformer = SmartCorrelatedSelection(method="not_valid")

    with pytest.raises(ValueError) as errmsg:
        transformer.fit(X)

    exceptionmsg = errmsg.value.args[0]

    assert (
        exceptionmsg
        == "method must be either 'pearson', 'spearman', 'kendall', or a callable,"
        + " 'not_valid' was supplied"
    )


def test_raises_missing_data_error(df_nan):
    df = df_nan
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    sel = SmartCorrelatedSelection(selection_method="variance", missing_values="raise")
    with pytest.raises(ValueError) as record:
        sel.fit(df)
    assert record.value.args[0] == msg


def test_callable_method(df_test, random_uniform_method):
    X, _ = df_test

    transformer = SmartCorrelatedSelection(
        method=random_uniform_method,
    )

    Xt = transformer.fit_transform(X)

    # test no empty dataframe
    assert not Xt.empty

    # test fit attrs
    assert len(transformer.correlated_feature_sets_) > 0
    assert len(transformer.features_to_drop_) > 0
    assert len(transformer.variables_) > 0
    assert transformer.n_features_in_ == len(X.columns)


def test_smart_correlation_selection_with_groups(df_test_with_groups):
    X, y, groups = df_test_with_groups
    cv = GroupKFold(n_splits=3)
    cv_indices = cv.split(X=X, y=y, groups=groups)

    estimator = RandomForestRegressor(n_estimators=3, random_state=1)
    scoring = "neg_mean_absolute_error"
    selection_method = "variance"

    transformer_expected = SmartCorrelatedSelection(
        estimator=estimator,
        scoring=scoring,
        selection_method=selection_method,
        cv=cv_indices,
    )

    X_tr_expected = transformer_expected.fit_transform(X, y)

    transformer = SmartCorrelatedSelection(
        estimator=estimator,
        scoring=scoring,
        selection_method=selection_method,
        cv=cv,
        groups=groups,
    )

    X_tr = transformer.fit_transform(X, y)

    pd.testing.assert_frame_equal(X_tr_expected, X_tr)
