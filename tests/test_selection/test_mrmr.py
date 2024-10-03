import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.model_selection import GridSearchCV

from feature_engine.encoding import OrdinalEncoder
from feature_engine.selection import MRMR


@pytest.mark.parametrize("method", ["MIQ", "MID", "FCQ", "FCD", "RFCQ"])
def test_method_param(method):
    tr = MRMR(method=method)
    assert tr.method == method


@pytest.mark.parametrize("method", [10, "string", False, [0, 1]])
def test_method_raises_error(method):
    msg = (
        "method must be one of 'MIQ', 'MID', 'FCQ', 'FCD', 'RFCQ'. "
        f"Got {method} instead."
    )
    with pytest.raises(ValueError) as record:
        MRMR(method=method)
    assert str(record.value) == msg


@pytest.mark.parametrize("threshold", [0, 0.5, 10, -1, None])
def test_threshold_param(threshold):
    tr = MRMR(threshold=threshold)
    if threshold is not None:
        assert tr.threshold == threshold
    else:
        assert tr.threshold is None


@pytest.mark.parametrize("threshold", ["string", {"string"}, [0, 1]])
def test_threshold_raises_error(threshold):
    msg = "threshold can only take integer or float. " f"Got {threshold} instead."
    with pytest.raises(ValueError) as record:
        MRMR(threshold=threshold)
    assert str(record.value) == msg


@pytest.mark.parametrize("scoring", ["roc_auc", "accuracy", "precision"])
def test_raises_error_when_metric_not_suitable_for_regression(scoring):
    msg = (
        f"The metric {scoring} is not suitable for regression. Set the "
        "parameter regression to False or choose a different performance "
        "metric."
    )
    with pytest.raises(ValueError) as record:
        MRMR(method="RFCQ", regression=True, scoring=scoring)
    assert str(record.value) == msg


@pytest.mark.parametrize(
    "method",
    [
        "MIQ",
        "MID",
        "FCQ",
        "FCD",
    ],
)
@pytest.mark.parametrize("scoring", ["roc_auc", "accuracy", "precision"])
def test_metric_does_not_raise_error_when_not_RF_regression(method, scoring):
    tr = MRMR(method=method, regression=True, scoring=scoring)
    assert tr.method == method
    assert tr.scoring == scoring


@pytest.mark.parametrize("scoring", ["mse", "mae", "r2"])
def test_raises_error_when_metric_not_suitable_for_classif(scoring):
    msg = (
        f"The metric {scoring} is not suitable for classification. Set the"
        "parameter regression to True or choose a different performance "
        "metric."
    )
    with pytest.raises(ValueError) as record:
        MRMR(method="RFCQ", regression=False, scoring=scoring)
    assert str(record.value) == msg


@pytest.mark.parametrize(
    "method",
    [
        "MIQ",
        "MID",
        "FCQ",
        "FCD",
    ],
)
@pytest.mark.parametrize("scoring", ["mse", "mae", "r2"])
def test_metric_does_not_raise_error_when_not_RF_classif(method, scoring):
    tr = MRMR(method=method, regression=False, scoring=scoring)
    assert tr.method == method
    assert tr.scoring == scoring


def test_mrmr_mid_and_miq_classif(df_test):
    X, y = df_test
    relevance = mutual_info_classif(X, y, random_state=42)

    redundance = []

    for feature in X.columns:
        red = np.mean(
            mutual_info_regression(X.drop(feature, axis=1), X[feature], random_state=42)
        )
        redundance.append(red)

    mrmr_d = relevance - redundance
    mrmr_q = relevance / redundance

    sel = MRMR(method="MID", regression=False, random_state=42)
    sel.fit(X, y)

    assert (sel.relevance_ == relevance).all()
    assert np.allclose(np.array(redundance), sel.redundance_)
    assert (sel.mrmr_ == mrmr_d).all()

    sel = MRMR(method="MIQ", regression=False, random_state=42)
    sel.fit(X, y)

    assert (sel.relevance_ == relevance).all()
    assert np.allclose(np.array(redundance), sel.redundance_)
    assert (sel.mrmr_ == mrmr_q).all()
    assert sel.features_to_drop_ == [
        "var_0",
        "var_2",
        "var_3",
        "var_8",
        "var_10",
        "var_11",
    ]

    Xtr = sel.transform(X)
    pd.testing.assert_frame_equal(
        X[["var_1", "var_4", "var_5", "var_6", "var_7", "var_9"]], Xtr
    )


def test_mrmr_fcd_and_fcq_classif(df_test):
    X, y = df_test
    relevance = f_classif(X, y)[0]

    redundance = []
    for feature in X.columns:
        f = f_regression(X.drop(feature, axis=1), X[feature])
        red = np.mean(f[0])
        redundance.append(red)

    mrmr_d = relevance - np.array(redundance)
    mrmr_q = relevance / np.array(redundance)

    sel = MRMR(method="FCD", regression=False, random_state=42)
    sel.fit(X, y)

    assert (sel.relevance_ == relevance).all()
    assert np.allclose(np.array(redundance), sel.redundance_)
    assert np.allclose(np.array(sel.mrmr_), mrmr_d)

    sel = MRMR(method="FCQ", regression=False, random_state=42)
    sel.fit(X, y)

    assert (sel.relevance_ == relevance).all()
    assert np.allclose(np.array(redundance), sel.redundance_)
    assert np.allclose(np.array(sel.mrmr_), mrmr_q)
    assert sel.features_to_drop_ == [
        "var_0",
        "var_3",
        "var_4",
        "var_6",
        "var_8",
        "var_9",
        "var_10",
    ]

    Xtr = sel.transform(X)
    pd.testing.assert_frame_equal(
        X[["var_1", "var_2", "var_5", "var_7", "var_11"]], Xtr
    )


@pytest.fixture(scope="module")
def df_test_regression():
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_redundant=4,
        n_clusters_per_class=1,
        weights=[0.50],
        class_sep=2,
        random_state=1,
    )

    # transform arrays into pandas df and series
    colnames = ["var_" + str(i) for i in range(12)]
    X = pd.DataFrame(X, columns=colnames)
    y = pd.Series(0.5 * X["var_1"] - 0.3 * X["var_3"] + X["var_8"])
    return X, y


def test_mrmr_mid_and_miq_regression(df_test_regression):
    X, y = df_test_regression
    relevance = mutual_info_regression(X, y, random_state=42)

    redundance = []

    for feature in X.columns:
        red = np.mean(
            mutual_info_regression(X.drop(feature, axis=1), X[feature], random_state=42)
        )
        redundance.append(red)

    mrmr_d = relevance - redundance
    mrmr_q = relevance / redundance

    sel = MRMR(method="MID", regression=True, random_state=42)
    sel.fit(X, y)

    assert (sel.relevance_ == relevance).all()
    assert np.allclose(np.array(redundance), sel.redundance_)
    assert (sel.mrmr_ == mrmr_d).all()

    sel = MRMR(method="MIQ", regression=True, random_state=42, threshold=2)
    sel.fit(X, y)

    assert (sel.relevance_ == relevance).all()
    assert np.allclose(np.array(redundance), sel.redundance_)
    assert (sel.mrmr_ == mrmr_q).all()
    assert sel.features_to_drop_ == [
        "var_0",
        "var_2",
        "var_4",
        "var_6",
        "var_7",
        "var_8",
        "var_9",
        "var_10",
        "var_11",
    ]

    Xtr = sel.transform(X)
    pd.testing.assert_frame_equal(X[["var_1", "var_3", "var_5"]], Xtr)


def test_mrmr_fcd_and_fcq_regression(df_test_regression):
    X, y = df_test_regression
    relevance = f_regression(X, y)[0]

    redundance = []
    for feature in X.columns:
        f = f_regression(X.drop(feature, axis=1), X[feature])
        red = np.mean(f[0])
        redundance.append(red)

    mrmr_d = relevance - np.array(redundance)
    mrmr_q = relevance / np.array(redundance)

    sel = MRMR(method="FCD", regression=True, random_state=42, threshold=2)
    sel.fit(X, y)

    assert np.allclose(np.array(relevance), sel.relevance_)
    assert np.allclose(np.array(redundance), sel.redundance_)
    assert np.allclose(np.array(sel.mrmr_), mrmr_d)

    sel = MRMR(method="FCQ", regression=True, random_state=42, threshold=20)
    sel.fit(X, y)

    assert np.allclose(np.array(relevance), sel.relevance_)
    assert np.allclose(np.array(redundance), sel.redundance_)
    assert np.allclose(np.array(sel.mrmr_), mrmr_q)
    assert sel.features_to_drop_ == [
        "var_0",
        "var_2",
        "var_4",
        "var_5",
        "var_6",
        "var_7",
        "var_8",
        "var_9",
        "var_10",
        "var_11",
    ]

    Xtr = sel.transform(X)
    pd.testing.assert_frame_equal(X[["var_1", "var_3"]], Xtr)


def test_mrmr_random_forest(df_test, df_test_regression):
    # classification
    X, y = df_test

    param_grid = {"max_depth": [1, 2, 3, 4]}
    model = GridSearchCV(
        RandomForestClassifier(random_state=42),
        cv=3,
        scoring="roc_auc",
        param_grid=param_grid,
    )
    model.fit(X, y)
    relevance = model.best_estimator_.feature_importances_

    redundance = []
    for feature in X.columns:
        f = f_regression(X.drop(feature, axis=1), X[feature])
        red = np.mean(f[0])
        redundance.append(red)

    mrmr_q = relevance / np.array(redundance)

    sel = MRMR(method="RFCQ", regression=False, random_state=42)
    sel.fit(X, y)

    assert np.allclose(np.array(relevance), sel.relevance_)
    assert np.allclose(np.array(redundance), sel.redundance_)
    assert np.allclose(np.array(sel.mrmr_), mrmr_q)

    # regression
    X, y = df_test_regression

    model = GridSearchCV(
        RandomForestRegressor(random_state=42),
        cv=3,
        scoring="r2",
        param_grid=param_grid,
    )
    model.fit(X, y)
    relevance = model.best_estimator_.feature_importances_

    redundance = []
    for feature in X.columns:
        f = f_regression(X.drop(feature, axis=1), X[feature])
        red = np.mean(f[0])
        redundance.append(red)

    mrmr_q = relevance / np.array(redundance)

    sel = MRMR(method="RFCQ", regression=True, scoring="r2", random_state=42)
    sel.fit(X, y)

    assert np.allclose(np.array(relevance), sel.relevance_)
    assert np.allclose(np.array(redundance), sel.redundance_)
    assert np.allclose(np.array(sel.mrmr_), mrmr_q)


def test_can_work_on_variable_groups(df_test):
    X, y = df_test
    varlist = ["var_" + str(i) for i in range(5)]

    relevance = f_classif(X[varlist], y)[0]

    redundance = []
    for feature in varlist:
        f = f_regression(X[varlist].drop(feature, axis=1), X[feature])
        red = np.mean(f[0])
        redundance.append(red)

    mrmr_d = relevance - np.array(redundance)

    sel = MRMR(variables=varlist, method="FCD", regression=False, random_state=42)
    sel.fit(X, y)

    assert (sel.relevance_ == relevance).all()
    assert np.allclose(np.array(redundance), sel.redundance_)
    assert np.allclose(np.array(sel.mrmr_), mrmr_d)


@pytest.mark.parametrize(
    "discrete", [[True, True, False, False], np.array([True, True, False, False])]
)
def test_discrete_features_among_predictors(df_test_num_cat, discrete):
    # the first 2 features are discrete/categorical
    X, y = df_test_num_cat
    X = OrdinalEncoder(encoding_method="arbitrary").fit_transform(X)

    sel = MRMR(
        method="MID", discrete_features=discrete, regression=False, random_state=42
    )
    redundance = sel._calculate_redundance(X)

    mi = mutual_info_classif(
        X=X.drop(["var_A"], axis=1),
        y=X["var_A"],
        discrete_features=[True, False, False],
        random_state=42,
    )
    assert np.mean(mi) == redundance[0]

    mi = mutual_info_classif(
        X=X.drop(["var_B"], axis=1),
        y=X["var_B"],
        discrete_features=[True, False, False],
        random_state=42,
    )
    assert np.mean(mi) == redundance[1]

    mi = mutual_info_regression(
        X=X.drop(["var_C"], axis=1),
        y=X["var_C"],
        discrete_features=[True, True, False],
        random_state=42,
    )
    assert np.mean(mi) == redundance[2]

    mi = mutual_info_regression(
        X=X.drop(["var_D"], axis=1),
        y=X["var_D"],
        discrete_features=[True, True, False],
        random_state=42,
    )
    assert np.mean(mi) == redundance[3]
