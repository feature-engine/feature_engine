import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import (
    make_classification,
    load_breast_cancer,
    fetch_california_housing,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.model_selection import GridSearchCV

from feature_engine.selection import MRMR


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


@pytest.mark.parametrize("max_features", [1, 10, 20, None])
def test_max_features_param(max_features):
    tr = MRMR(max_features=max_features)
    if max_features is not None:
        assert tr.max_features == max_features
    else:
        assert tr.max_features is None


@pytest.mark.parametrize("max_features", ["string", -1, [0, 1]])
def test_max_features_raises_error(max_features):
    msg = (
        "max_features must be an integer with the number of features to "
        f"select. Got {max_features} instead."
    )
    with pytest.raises(ValueError) as record:
        MRMR(max_features=max_features)
    assert str(record.value) == msg


@pytest.mark.parametrize(
    ("vars", "max_feat"),
    [(["var1", "var2"], 2), (["var1", "var2"], 3), (["var1", "var2", "var3"], 4)],
)
def test_max_features_raises_error_when_more_than_vars(vars, max_feat):
    msg = (
        f"The number of variables to examine is {len(vars)}, which is "
        "less than or equal to the number of features to select indicated "
        f"in `max_features`, which is {max_feat}. Please check the "
        "values entered in the parameters `variables` and `max_features`."
    )
    with pytest.raises(ValueError) as record:
        MRMR(variables=vars, max_features=max_feat)
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
    # scoring is only used when method='RFCQ'. Otherwise it should be ignored.
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
    # scoring is only used when method='RFCQ'. Otherwise it should be ignored.
    tr = MRMR(method=method, regression=False, scoring=scoring)
    assert tr.method == method
    assert tr.scoring == scoring


@pytest.mark.parametrize("method", ["MID", "MIQ"])
def test_calculate_relevance_mi(df_test, df_test_regression, method):
    X, y = df_test
    expected_relevance = mutual_info_classif(X, y, random_state=42)

    sel = MRMR(method=method, regression=False, random_state=42)
    relevance = sel._calculate_relevance(X, y)

    assert np.array_equal(expected_relevance, relevance)

    X, y = df_test_regression
    expected_relevance = mutual_info_regression(X, y, random_state=42)

    sel = MRMR(method=method, regression=True, random_state=42)
    relevance = sel._calculate_relevance(X, y)

    assert np.array_equal(expected_relevance, relevance)


@pytest.mark.parametrize("method", ["FCD", "FCQ"])
def test_calculate_relevance_f(df_test, df_test_regression, method):
    X, y = df_test
    expected_relevance = f_classif(X, y)[0]

    sel = MRMR(method=method, regression=False, random_state=42)
    relevance = sel._calculate_relevance(X, y)

    assert np.array_equal(expected_relevance, relevance)

    X, y = df_test_regression
    expected_relevance = f_regression(X, y)[0]

    sel = MRMR(method=method, regression=True, random_state=42)
    relevance = sel._calculate_relevance(X, y)

    assert np.array_equal(expected_relevance, relevance)


def test_calculate_relevance_RF(df_test, df_test_regression):
    param_grid = {"max_depth": [1, 2, 3, 4]}

    X, y = df_test
    model = RandomForestClassifier(random_state=42)
    model = GridSearchCV(model, cv=3, scoring="roc_auc", param_grid=param_grid)
    model.fit(X, y)
    expected_relevance = model.best_estimator_.feature_importances_

    sel = MRMR(
        method="RFCQ", regression=False, random_state=42, cv=3, scoring="roc_auc"
    )
    relevance = sel._calculate_relevance(X, y)

    assert np.array_equal(expected_relevance, relevance)

    X, y = df_test_regression
    model = RandomForestRegressor(random_state=42)
    model = GridSearchCV(model, cv=3, scoring="r2", param_grid=param_grid)
    model.fit(X, y)
    expected_relevance = model.best_estimator_.feature_importances_

    sel = MRMR(
        method="RFCQ",
        regression=True,
        random_state=42,
        cv=3,
        scoring="r2",
        param_grid=param_grid,
    )
    relevance = sel._calculate_relevance(X, y)

    assert np.array_equal(expected_relevance, relevance)


@pytest.mark.parametrize(
    ("method1", "method2"), [("MID", "FCD"), ("MIQ", "FCQ"), ("MIQ", "RFCQ")]
)
def test_calculate_redundance(df_test_regression, method1, method2):
    X, y = df_test_regression

    expected_redundance = mutual_info_regression(X, y, random_state=42)
    sel = MRMR(method=method1, random_state=42)
    redundance = sel._calculate_redundance(X, y)
    assert np.array_equal(expected_redundance, redundance)

    expected_redundance = np.absolute(X.corrwith(y).values)
    sel = MRMR(method=method2)
    redundance = sel._calculate_redundance(X, y)
    assert np.array_equal(expected_redundance, redundance)


@pytest.mark.parametrize("method", ["MID", "MIQ", "FCD", "FCQ", "RFCQ"])
def test_calculate_mrmr(method):
    relevance = np.array([1, 2, 3, 4, 5])
    redundance = np.array([1, 2, 3, 4, 5])

    sel = MRMR(method=method)
    mrmr = sel._calculate_mrmr(relevance, redundance)

    if method in ["MID", "FCD"]:
        expected_mrmr = relevance - redundance
    else:
        expected_mrmr = relevance / redundance
    assert np.array_equal(expected_mrmr, mrmr)


def test_mrmr_mid_and_miq_classif():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    selected = [
        "worst perimeter",
        "fractal dimension error",
        "worst texture",
        "worst smoothness",
        "mean concave points",
        "perimeter error",
        "worst symmetry",
        "worst concave points",
        "area error",
        "mean perimeter",
    ]
    to_drop = [f for f in X.columns if f not in selected]

    sel = MRMR(method="MIQ", max_features=10, regression=False, random_state=42)
    sel.fit(X, y)
    Xtr = sel.transform(X)

    assert sel.features_to_drop_ == to_drop
    pd.testing.assert_frame_equal(X.drop(to_drop, axis=1), Xtr)

    selected = [
        "worst perimeter",
        "worst smoothness",
        "worst texture",
        "mean concave points",
        "perimeter error",
        "worst concavity",
        "worst symmetry",
        "area error",
        "symmetry error",
        "worst concave points",
    ]
    to_drop = [f for f in X.columns if f not in selected]

    sel = MRMR(method="MID", max_features=10, regression=False, random_state=42)
    sel.fit(X, y)
    Xtr = sel.transform(X)

    assert sel.features_to_drop_ == to_drop
    pd.testing.assert_frame_equal(X.drop(to_drop, axis=1), Xtr)


def test_mrmr_fcd_and_fcq_regression():
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)

    selected = ["MedInc", "Latitude", "HouseAge", "AveRooms", "AveOccup"]
    to_drop = [f for f in X.columns if f not in selected]

    sel = MRMR(method="FCQ", max_features=5, regression=True)
    Xtr = sel.fit_transform(X, y)

    assert sel.features_to_drop_ == to_drop
    pd.testing.assert_frame_equal(X.drop(to_drop, axis=1), Xtr)

    selected = ["MedInc", "AveRooms", "Latitude", "HouseAge"]
    to_drop = [f for f in X.columns if f not in selected]

    sel = MRMR(method="FCD", max_features=4, regression=True)
    Xtr = sel.fit_transform(X, y)

    assert sel.features_to_drop_ == to_drop
    pd.testing.assert_frame_equal(X.drop(to_drop, axis=1), Xtr)
