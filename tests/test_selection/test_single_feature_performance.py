import re
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import SelectBySingleFeaturePerformance
from tests.backend_helpers import frame_to_dict, make_series

# performance of RandomForestClassifier(n_estimators=5, random_state=1), cv=3 and
# roc-auc, on the data_classification fixture.
CLF_PERFORMANCE = {
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

CLF_PERFORMANCE_STD = {
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

# performance of LinearRegression(), cv=3 and r2, on the diabetes dataset.
R2_PERFORMANCE = {
    "age": 0.029231969375784466,
    "sex": -0.003738551760264386,
    "bmi": 0.33662080998769284,
    "bp": 0.19218913007834937,
    "s1": 0.037115559827549806,
    "s2": 0.01785422825693254,
    "s3": 0.15153886177526887,
    "s4": 0.1772160996650173,
    "s5": 0.31494478799681097,
    "s6": 0.13876602125792703,
}

R2_PERFORMANCE_STD = {
    "age": 0.017870583127141664,
    "sex": 0.005465336770744777,
    "bmi": 0.04257342727445452,
    "bp": 0.027318947204928765,
    "s1": 0.031397211603399186,
    "s2": 0.03224477055466244,
    "s3": 0.020243573053986393,
    "s4": 0.04782262499458294,
    "s5": 0.02473650354444323,
    "s6": 0.029051175300521623,
}


@pytest.fixture(scope="module")
def data_diabetes():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    data = X.to_dict(orient="list")
    data["target"] = y.tolist()
    return data


def _split_target(make_df, data):
    X = make_df({k: v for k, v in data.items() if k != "target"})
    return X, make_series(make_df, data["target"])


def _rf_classifier():
    return RandomForestClassifier(n_estimators=5, random_state=1)


# init parameters
@pytest.mark.parametrize("threshold", ["hola", [0.6], (0.6,), [], {"a": 1}])
def test_error_if_threshold_not_number(threshold):
    msg = f"`threshold` can only be integer, float or None. Got {threshold} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        SelectBySingleFeaturePerformance(_rf_classifier(), threshold=threshold)


@pytest.mark.parametrize("threshold", [0, 0.4, -1, 1.5])
def test_error_if_roc_auc_threshold_not_between_05_and_1(threshold):
    msg = (
        "`threshold` for roc-auc score should be between 0.5 and 1. "
        f"Got {threshold} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SelectBySingleFeaturePerformance(
            _rf_classifier(), scoring="roc_auc", threshold=threshold
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.5, 4])
def test_error_if_r2_threshold_not_between_0_and_1(threshold):
    msg = (
        "`threshold` for r2 score should be between 0 and 1. "
        f"Got {threshold} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        SelectBySingleFeaturePerformance(
            LinearRegression(), scoring="r2", threshold=threshold
        )


@pytest.mark.parametrize(
    "scoring, cv, groups, threshold, confirm_variables",
    [
        ("roc_auc", 3, None, None, False),
        ("roc_auc", 5, None, 0.5, True),
        ("r2", 2, None, 0, False),
        ("neg_mean_squared_error", GroupKFold(3), [1, 1, 2, 2, 3, 3], -10, True),
    ],
)
def test_init_param_assignment(scoring, cv, groups, threshold, confirm_variables):
    estimator = LinearRegression()
    sel = SelectBySingleFeaturePerformance(
        estimator,
        scoring=scoring,
        cv=cv,
        groups=groups,
        threshold=threshold,
        confirm_variables=confirm_variables,
    )
    assert sel.estimator is estimator
    assert sel.scoring == scoring
    assert sel.cv is cv
    assert sel.groups is groups
    assert sel.threshold == threshold
    assert sel.confirm_variables is confirm_variables


# fit and transform
def test_default_threshold_is_mean_performance(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = SelectBySingleFeaturePerformance(_rf_classifier())
    Xt = sel.fit_transform(X, y)

    # the mean performance is 0.667
    dropped = ["var_0", "var_1", "var_2", "var_3", "var_5", "var_8", "var_10"]
    assert sel.features_to_drop_ == dropped + ["var_11"]
    assert isinstance(sel.feature_performance_, dict)
    assert isinstance(sel.feature_performance_std_, dict)
    assert sel.feature_performance_ == pytest.approx(CLF_PERFORMANCE)
    assert sel.feature_performance_std_ == pytest.approx(CLF_PERFORMANCE_STD)
    assert sel.variables_ == list(CLF_PERFORMANCE)
    assert sel.feature_names_in_ == list(CLF_PERFORMANCE)
    assert sel.n_features_in_ == 12
    assert sel.get_support() == [
        False, False, False, False, True, False, True, True, False, True, False, False
    ]  # fmt: skip

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        var: data_classification[var] for var in ["var_4", "var_6", "var_7", "var_9"]
    }


def test_regression_with_r2_threshold(make_df, data_diabetes):
    X, y = _split_target(make_df, data_diabetes)
    sel = SelectBySingleFeaturePerformance(
        LinearRegression(), scoring="r2", cv=3, threshold=0.01
    )
    Xt = sel.fit_transform(X, y)

    assert sel.features_to_drop_ == ["sex"]
    assert sel.feature_performance_ == pytest.approx(R2_PERFORMANCE)
    assert sel.feature_performance_std_ == pytest.approx(R2_PERFORMANCE_STD)
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    assert frame_to_dict(Xt)["bmi"] == data_diabetes["bmi"]


def test_threshold_zero_is_not_replaced_by_mean_performance(make_df, data_diabetes):
    # the mean r2 is 0.139, which would drop age, sex, s1, s2 and s6.
    X, y = _split_target(make_df, data_diabetes)
    sel = SelectBySingleFeaturePerformance(
        LinearRegression(), scoring="r2", cv=3, threshold=0
    )
    sel.fit(X, y)
    assert sel.features_to_drop_ == ["sex"]


def test_regression_with_negative_mse_threshold(make_df, data_diabetes):
    X, y = _split_target(make_df, data_diabetes)
    sel = SelectBySingleFeaturePerformance(
        DecisionTreeRegressor(random_state=0),
        scoring="neg_mean_squared_error",
        cv=2,
        threshold=-6000,
    )
    Xt = sel.fit_transform(X, y)

    assert sel.features_to_drop_ == ["age", "bmi", "bp", "s1", "s2", "s3", "s5", "s6"]
    assert sel.feature_performance_ == pytest.approx(
        {
            "age": -7657.154138192973,
            "sex": -5966.662211695372,
            "bmi": -6613.779604700854,
            "bp": -6488.926775995113,
            "s1": -9415.586278197177,
            "s2": -11760.999622926094,
            "s3": -6592.584431571728,
            "s4": -5270.563893676307,
            "s5": -7547.199359602815,
            "s6": -6287.557824391035,
        }
    )
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "sex": data_diabetes["sex"],
        "s4": data_diabetes["s4"],
    }


def test_warning_if_all_features_dropped(make_df, data_diabetes):
    X, y = _split_target(make_df, data_diabetes)
    sel = SelectBySingleFeaturePerformance(
        DecisionTreeRegressor(random_state=0),
        scoring="neg_mean_squared_error",
        cv=2,
        threshold=10,
    )
    msg = "All features will be dropped, try changing the threshold."
    with pytest.warns(UserWarning, match=re.escape(msg)):
        sel.fit(X, y)
    assert sel.features_to_drop_ == list(R2_PERFORMANCE)


@pytest.mark.parametrize(
    "variables, confirm_variables",
    [
        (["var_0", "var_4", "var_7"], False),
        (["var_0", "var_4", "var_7", "var_20"], True),
    ],
)
def test_variables_subset(make_df, data_classification, variables, confirm_variables):
    X, y = _split_target(make_df, data_classification)
    sel = SelectBySingleFeaturePerformance(
        _rf_classifier(), variables=variables, confirm_variables=confirm_variables
    )
    Xt = sel.fit_transform(X, y)

    assert sel.variables_ == ["var_0", "var_4", "var_7"]
    assert sel.features_to_drop_ == ["var_0"]
    assert sel.feature_performance_ == pytest.approx(
        {var: CLF_PERFORMANCE[var] for var in ["var_0", "var_4", "var_7"]}
    )
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == [f"var_{i}" for i in range(1, 12)]


def test_non_numerical_variables_are_not_evaluated(make_df, data_classification):
    data = {
        **data_classification,
        "cat_1": ["a"] * 1000,
        "cat_2": ["b"] * 1000,
    }
    X, y = _split_target(make_df, data)
    sel = SelectBySingleFeaturePerformance(_rf_classifier(), threshold=0.5)
    Xt = sel.fit_transform(X, y)

    assert sel.variables_ == list(CLF_PERFORMANCE)
    assert sel.features_to_drop_ == ["var_3", "var_10"]
    assert sel.feature_performance_ == pytest.approx(CLF_PERFORMANCE)
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt)["cat_1"] == ["a"] * 1000
    assert list(Xt.columns) == [
        "var_0", "var_1", "var_2", "var_4", "var_5", "var_6", "var_7", "var_8",
        "var_9", "var_11", "cat_1", "cat_2",
    ]  # fmt: skip


@pytest.mark.parametrize("variable, dropped", [("var_0", []), ("var_3", ["var_3"])])
def test_single_variable_with_threshold(
    make_df, data_classification, variable, dropped
):
    X, y = _split_target(make_df, data_classification)
    sel = SelectBySingleFeaturePerformance(
        _rf_classifier(), variables=[variable], threshold=0.5
    )
    Xt = sel.fit_transform(X, y)

    assert sel.features_to_drop_ == dropped
    assert sel.feature_performance_ == pytest.approx(
        {variable: CLF_PERFORMANCE[variable]}
    )
    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == [var for var in CLF_PERFORMANCE if var not in dropped]


def test_error_if_single_variable_and_threshold_is_none(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = SelectBySingleFeaturePerformance(_rf_classifier(), variables=["var_1"])
    msg = (
        "When evaluating a single feature you need to manually set a value for the "
        "threshold. The transformer is evaluating the performance of ['var_1'] and "
        "the threshold was left to None when initializing the transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        sel.fit(X, y)


def test_mean_performance_skips_features_that_could_not_be_evaluated(make_df):
    # linear regression can't be trained with missing values, so the performance
    # of var_c is NaN, and the threshold is the mean of var_a and var_b.
    rng = np.random.default_rng(0)
    var_a = rng.normal(size=30)
    data = {
        "var_a": var_a.tolist(),
        "var_b": rng.normal(size=30).tolist(),
        "var_c": [None] + rng.normal(size=29).tolist(),
    }
    y = make_series(make_df, (2 * var_a + rng.normal(size=30)).tolist())
    sel = SelectBySingleFeaturePerformance(LinearRegression(), scoring="r2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel.fit(make_df(data), y)

    assert sel.feature_performance_ == pytest.approx(
        {"var_a": 0.5912882991411328, "var_b": -0.4485453118122386, "var_c": np.nan},
        nan_ok=True,
    )
    assert sel.features_to_drop_ == ["var_b"]


@pytest.mark.parametrize("target_type", [list, np.array])
def test_target_as_list_and_array(make_df, data_classification, target_type):
    X, _ = _split_target(make_df, data_classification)
    y = target_type(data_classification["target"])
    sel = SelectBySingleFeaturePerformance(
        _rf_classifier(), variables=["var_0", "var_4"]
    )
    sel.fit(X, y)

    assert sel.features_to_drop_ == ["var_0"]
    assert sel.feature_performance_ == pytest.approx(
        {"var_0": CLF_PERFORMANCE["var_0"], "var_4": CLF_PERFORMANCE["var_4"]}
    )


def test_cv_splitter(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = SelectBySingleFeaturePerformance(
        _rf_classifier(), cv=StratifiedKFold(n_splits=3)
    )
    sel.fit(X, y)

    # the folds of cv=3 for a classifier are the ones of StratifiedKFold(3).
    assert sel.feature_performance_ == pytest.approx(CLF_PERFORMANCE)


def test_groups_give_same_result_as_cv_generator(make_df):
    rng = np.random.default_rng(1)
    data = {f"var_{i}": rng.normal(size=100).tolist() for i in range(1, 6)}
    data["target"] = rng.integers(0, 100, size=100).tolist()
    groups = np.repeat(np.arange(1, 11), 10)
    rng.shuffle(groups)
    X, y = _split_target(make_df, data)
    estimator = RandomForestRegressor(n_estimators=3, random_state=3)
    cv = GroupKFold(n_splits=3)

    sel_groups = SelectBySingleFeaturePerformance(
        estimator, scoring="neg_mean_absolute_error", cv=cv, groups=groups
    ).fit(X, y)
    sel_generator = SelectBySingleFeaturePerformance(
        estimator,
        scoring="neg_mean_absolute_error",
        cv=cv.split(X=X, y=y, groups=groups),
    ).fit(X, y)

    assert sel_groups.features_to_drop_ == sel_generator.features_to_drop_
    assert sel_groups.feature_performance_ == sel_generator.feature_performance_
    assert sel_groups.feature_performance_std_ == sel_generator.feature_performance_std_


def test_transform_returns_training_column_order(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = SelectBySingleFeaturePerformance(_rf_classifier()).fit(X, y)
    reordered = {var: data_classification[var] for var in reversed(CLF_PERFORMANCE)}
    Xt = sel.transform(make_df(reordered))

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_4", "var_6", "var_7", "var_9"]


def test_error_if_transform_before_fit(make_df, data_classification):
    X, _ = _split_target(make_df, data_classification)
    msg = (
        "This SelectBySingleFeaturePerformance instance is not fitted yet. Call 'fit' "
        "with appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        SelectBySingleFeaturePerformance(_rf_classifier()).transform(X)


def test_integer_column_names(data_diabetes):
    # polars does not allow integer column names.
    X = pd.DataFrame({i: data_diabetes[var] for i, var in enumerate(R2_PERFORMANCE)})
    y = pd.Series(data_diabetes["target"])
    sel = SelectBySingleFeaturePerformance(
        LinearRegression(), scoring="r2", cv=3, threshold=0.01
    )
    Xt = sel.fit_transform(X, y)

    assert sel.variables_ == list(range(10))
    assert sel.features_to_drop_ == [1]
    assert sel.feature_performance_ == pytest.approx(
        dict(enumerate(R2_PERFORMANCE.values()))
    )
    pd.testing.assert_frame_equal(Xt, X[[0, 2, 3, 4, 5, 6, 7, 8, 9]])


def test_keeps_pandas_index(data_diabetes):
    X = pd.DataFrame(
        {var: data_diabetes[var] for var in R2_PERFORMANCE},
        index=range(100, 542),
    )
    y = pd.Series(data_diabetes["target"], index=range(100, 542))
    sel = SelectBySingleFeaturePerformance(
        LinearRegression(), scoring="r2", cv=3, threshold=0.01
    )
    Xt = sel.fit_transform(X, y)

    pd.testing.assert_frame_equal(Xt, X.drop(columns=["sex"]))
