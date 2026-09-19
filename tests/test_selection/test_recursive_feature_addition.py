import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import RecursiveFeatureAddition
from tests.backend_helpers import frame_to_dict, make_series


@pytest.fixture(scope="module")
def data_diabetes():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    data = {col: X[col].tolist() for col in X.columns}
    data["target"] = y.tolist()
    return data


@pytest.fixture(scope="module")
def data_with_groups():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 5))
    data = {f"var_{i}": X[:, i].tolist() for i in range(5)}
    data["target"] = (3 * X[:, 0] + X[:, 1] + rng.normal(size=100)).tolist()
    groups = np.repeat(np.arange(10), 10)
    rng.shuffle(groups)
    return data, groups


def _split_target(make_df, data):
    X = make_df({k: v for k, v in data.items() if k != "target"})
    return X, make_series(make_df, data["target"])


# init parameters
@pytest.mark.parametrize("threshold", [None, [0.1], "a_string", {"a": 1}])
def test_error_if_threshold_not_number(threshold):
    msg = f"threshold must be an integer or a float. Got {threshold} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        RecursiveFeatureAddition(RandomForestClassifier(), threshold=threshold)


@pytest.mark.parametrize("confirm_variables", [None, 1, "True", [True]])
def test_error_if_confirm_variables_not_bool(confirm_variables):
    msg = (
        "confirm_variables takes only values True and False. "
        f"Got {confirm_variables} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        RecursiveFeatureAddition(
            RandomForestClassifier(), confirm_variables=confirm_variables
        )


@pytest.mark.parametrize(
    "estimator, scoring, cv, groups, threshold, confirm_variables",
    [
        (RandomForestClassifier(), "roc_auc", 3, None, 0.01, False),
        (Lasso(), "neg_mean_squared_error", KFold(), [1, 2], 1, True),
        (DecisionTreeRegressor(), "r2", StratifiedKFold(), None, -0.5, False),
    ],
)
def test_init_param_assignment(
    estimator, scoring, cv, groups, threshold, confirm_variables
):
    selector = RecursiveFeatureAddition(
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


# fit and transform
@pytest.mark.parametrize(
    "estimator, cv, threshold, scoring, performance, drifts, dropped, retained",
    [
        (
            RandomForestClassifier(n_estimators=5, random_state=1),
            3,
            0.001,
            "roc_auc",
            0.9954973573949477,
            {
                "var_4": 0,
                "var_7": 0.024094900525623353,
                "var_6": -0.0010340785943196984,
                "var_9": -0.0010221260221260353,
                "var_0": -1.2604530676751935e-05,
                "var_8": -0.001076745655058886,
                "var_10": -0.0011244472840857833,
                "var_11": -0.001028283407801478,
                "var_1": -1.832727736339468e-05,
                "var_2": -9.015137027179598e-05,
                "var_3": -0.001076636995311686,
                "var_5": -7.218629206573457e-05,
            },
            [
                "var_0",
                "var_1",
                "var_2",
                "var_3",
                "var_5",
                "var_6",
                "var_8",
                "var_9",
                "var_10",
                "var_11",
            ],
            ["var_4", "var_7"],
        ),
        (
            LogisticRegression(random_state=10),
            2,
            0.0001,
            "accuracy",
            0.99,
            {
                "var_7": 0,
                "var_8": 0.001,
                "var_0": 0.002,
                "var_6": 0,
                "var_4": 0,
                "var_11": -0.001,
                "var_1": -0.001,
                "var_5": -0.003,
                "var_3": -0.002,
                "var_10": 0,
                "var_9": 0,
                "var_2": 0,
            },
            [
                "var_1",
                "var_2",
                "var_3",
                "var_4",
                "var_5",
                "var_6",
                "var_9",
                "var_10",
                "var_11",
            ],
            ["var_0", "var_7", "var_8"],
        ),
    ],
)
def test_classification(
    make_df,
    data_classification,
    estimator,
    cv,
    threshold,
    scoring,
    performance,
    drifts,
    dropped,
    retained,
):
    X, y = _split_target(make_df, data_classification)
    selector = RecursiveFeatureAddition(
        estimator=estimator, cv=cv, threshold=threshold, scoring=scoring
    )
    Xt = selector.fit_transform(X, y)

    assert selector.initial_model_performance_ == pytest.approx(performance)
    assert selector.performance_drifts_ == pytest.approx(drifts)
    assert list(selector.performance_drifts_) == list(drifts)
    assert selector.features_to_drop_ == dropped
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {f: data_classification[f] for f in retained}


@pytest.mark.parametrize(
    "estimator, cv, threshold, scoring, drifts, dropped, retained",
    [
        (
            Lasso(alpha=0.001, random_state=10),
            3,
            0.1,
            "r2",
            {
                "s5": 0,
                "s1": 0.005876977602335132,
                "bmi": 0.1367220291124882,
                "s2": -0.002593999790115431,
                "bp": 0.017686140335452738,
                "sex": -0.004478777508460652,
                "s4": -0.003491685019777313,
                "s3": 0.008817074774180478,
                "s6": 0.002042365258727974,
                "age": -0.011362436906615925,
            },
            ["age", "sex", "bp", "s1", "s2", "s3", "s4", "s6"],
            ["bmi", "s5"],
        ),
        (
            DecisionTreeRegressor(random_state=10),
            2,
            100,
            "neg_mean_squared_error",
            {
                "bmi": 0,
                "s5": -660.1106667923586,
                "s2": -943.6298975615891,
                "s4": 99.31498680241202,
                "bp": -222.29449032177035,
                "s6": -716.6378161136254,
                "s3": -1701.2939247109098,
                "age": -1693.8544450729005,
                "s1": -781.6593093262954,
                "sex": 106.92720965309127,
            },
            ["age", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
            ["sex", "bmi"],
        ),
    ],
)
def test_regression(
    make_df, data_diabetes, estimator, cv, threshold, scoring, drifts, dropped, retained
):
    X, y = _split_target(make_df, data_diabetes)
    selector = RecursiveFeatureAddition(
        estimator=estimator, cv=cv, threshold=threshold, scoring=scoring
    )
    Xt = selector.fit_transform(X, y)

    assert selector.performance_drifts_ == pytest.approx(drifts)
    assert list(selector.performance_drifts_) == list(drifts)
    assert selector.features_to_drop_ == dropped
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {f: data_diabetes[f] for f in retained}


def test_attributes_with_linear_regression(make_df, data_diabetes):
    X, y = _split_target(make_df, data_diabetes)
    selector = RecursiveFeatureAddition(LinearRegression(), scoring="r2", cv=3)
    selector.fit(X, y)

    assert selector.initial_model_performance_ == pytest.approx(0.48870212980353145)
    # the importance is sorted from the most to the least important feature.
    assert dict(selector.feature_importances_) == pytest.approx(
        {
            "s1": 750.0238715216746,
            "s5": 741.4713367752565,
            "bmi": 522.3301645404875,
            "s2": 436.67158399913274,
            "bp": 322.0918016880965,
            "sex": 238.6195264502995,
            "s4": 182.174833735295,
            "s3": 113.96599187843395,
            "s6": 64.76841724774016,
            "age": 41.41804062408145,
        }
    )
    assert list(selector.feature_importances_.keys()) == [
        "s1",
        "s5",
        "bmi",
        "s2",
        "bp",
        "sex",
        "s4",
        "s3",
        "s6",
        "age",
    ]
    assert dict(selector.feature_importances_std_) == pytest.approx(
        {
            "age": 18.21715207624677,
            "sex": 68.35471931174253,
            "bmi": 86.03069812358049,
            "bp": 57.11038282196886,
            "s1": 329.3758185655728,
            "s2": 299.7569984100255,
            "s3": 72.80549636690168,
            "s4": 47.925822175574034,
            "s5": 117.8299487852095,
            "s6": 42.75477362271488,
        }
    )
    assert selector.performance_drifts_ == pytest.approx(
        {
            "s1": 0,
            "s5": 0.28371458794131676,
            "bmi": 0.13777147993887456,
            "s2": 0.0023327265047611845,
            "bp": 0.018759914615172624,
            "sex": 0.0027996354657458533,
            "s4": 0.0026951494400216935,
            "s3": 0.002683934134630417,
            "s6": 0.00030406740886079753,
            "age": -0.007387230783454712,
        }
    )
    assert selector.performance_drifts_std_ == pytest.approx(
        {
            "s1": 0,
            "s5": 0.02933691070157033,
            "bmi": 0.017524267327502716,
            "s2": 0.020525965661877258,
            "bp": 0.01732640124454753,
            "sex": 0.008676750772593741,
            "s4": 0.024234566449074697,
            "s3": 0.02339185113959813,
            "s6": 0.016865740401721358,
            "age": 0.020420816112180495,
        }
    )
    assert selector.features_to_drop_ == ["age", "sex", "s2", "s3", "s4", "s6"]


def test_feature_importances_type(make_df, data_diabetes):
    X, y = _split_target(make_df, data_diabetes)
    selector = RecursiveFeatureAddition(LinearRegression(), scoring="r2", cv=3)
    selector.fit(X, y)

    importance_type = pd.Series if make_df is pd.DataFrame else dict
    assert isinstance(selector.feature_importances_, importance_type)
    assert isinstance(selector.feature_importances_std_, importance_type)


def test_ranking_of_features_with_equal_importance(make_df):
    # Lasso sets most coefficients to 0. With more than 16 features the sort is not
    # stable, so both backends must rank these ties as pandas' sort_values does.
    X, y = make_classification(
        n_samples=500, n_features=25, n_informative=4, random_state=3
    )
    X = make_df({f"var_{i}": X[:, i] for i in range(25)})
    selector = RecursiveFeatureAddition(Lasso(alpha=0.05), scoring="r2", cv=3)
    selector.fit(X, make_series(make_df, y))

    assert list(selector.feature_importances_.keys()) == [
        "var_22",
        "var_13",
        "var_16",
        "var_9",
        "var_19",
        "var_11",
        "var_0",
        "var_14",
        "var_23",
        "var_21",
        "var_20",
        "var_18",
        "var_17",
        "var_15",
        "var_12",
        "var_1",
        "var_10",
        "var_8",
        "var_7",
        "var_6",
        "var_5",
        "var_4",
        "var_3",
        "var_2",
        "var_24",
    ]
    assert list(selector.performance_drifts_) == list(
        selector.feature_importances_.keys()
    )


def test_fit_with_permutation_importance(make_df, data_classification):
    # KNN has no coef_ or feature_importances_, so the importance comes from
    # permutation_importance.
    X, y = _split_target(make_df, data_classification)
    selector = RecursiveFeatureAddition(
        KNeighborsClassifier(), scoring="accuracy", cv=3
    )
    Xt = selector.fit_transform(X, y)

    assert list(selector.feature_importances_.keys())[:3] == [
        "var_7",
        "var_6",
        "var_4",
    ]
    assert selector.feature_importances_["var_7"] == pytest.approx(0.2313333333333333)
    assert selector.features_to_drop_ == [
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_6",
        "var_8",
        "var_9",
        "var_10",
        "var_11",
    ]
    assert frame_to_dict(Xt) == {"var_7": data_classification["var_7"]}


def test_variables_and_confirm_variables(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    selector = RecursiveFeatureAddition(
        LogisticRegression(),
        cv=3,
        variables=["var_0", "var_4", "var_7", "var_9", "Hola"],
        confirm_variables=True,
    )
    Xt = selector.fit_transform(X, y)

    assert selector.variables_ == ["var_0", "var_4", "var_7", "var_9"]
    assert selector.performance_drifts_ == pytest.approx(
        {
            "var_7": 0,
            "var_0": 0.0007903910012343474,
            "var_4": 0.0004547772620060453,
            "var_9": 0.0005748100627617214,
        }
    )
    assert selector.features_to_drop_ == ["var_0", "var_4", "var_9"]
    assert list(Xt.columns) == [
        "var_1",
        "var_2",
        "var_3",
        "var_5",
        "var_6",
        "var_7",
        "var_8",
        "var_10",
        "var_11",
    ]


def test_non_numerical_variables_are_ignored_and_kept(make_df, data_classification):
    data = {
        "var_0": data_classification["var_0"],
        "cat": ["a", "b"] * 500,
        "var_4": data_classification["var_4"],
        "var_7": data_classification["var_7"],
        "target": data_classification["target"],
    }
    X, y = _split_target(make_df, data)
    selector = RecursiveFeatureAddition(LogisticRegression(), cv=3)
    Xt = selector.fit_transform(X, y)

    assert selector.variables_ == ["var_0", "var_4", "var_7"]
    assert list(selector.performance_drifts_) == ["var_7", "var_0", "var_4"]
    assert selector.features_to_drop_ == ["var_0", "var_4"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"cat": data["cat"], "var_7": data["var_7"]}


def test_fit_with_missing_values(make_df, data_classification):
    # The selector doesn't check for NaN: estimators that handle it can be used.
    data = dict(data_classification)
    for var in ["var_0", "var_4", "var_7"]:
        data[var] = [None if i % 20 == 0 else v for i, v in enumerate(data[var])]
    X, y = _split_target(make_df, data)
    selector = RecursiveFeatureAddition(
        HistGradientBoostingClassifier(max_iter=10, random_state=0),
        scoring="accuracy",
        cv=3,
    )
    Xt = selector.fit_transform(X, y)

    assert selector.features_to_drop_ == [
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_8",
        "var_9",
        "var_10",
        "var_11",
    ]
    assert frame_to_dict(Xt) == {f: data[f] for f in ["var_6", "var_7"]}


def test_transform_returns_features_in_training_order(make_df, data_diabetes):
    X, y = _split_target(make_df, data_diabetes)
    selector = RecursiveFeatureAddition(LinearRegression(), scoring="r2", cv=3)
    selector.fit(X, y)
    Xt = selector.transform(X[list(X.columns)[::-1]])

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {f: data_diabetes[f] for f in ["bmi", "bp", "s1", "s5"]}


@pytest.mark.parametrize("cv_type", ["splitter", "generator"])
def test_cv_splitter_and_generator(make_df, data_diabetes, cv_type):
    X, y = _split_target(make_df, data_diabetes)
    cv = KFold(n_splits=3)
    if cv_type == "generator":
        cv = cv.split(X, y)
    selector = RecursiveFeatureAddition(LinearRegression(), scoring="r2", cv=cv)
    selector.fit(X, y)

    assert selector.performance_drifts_["s5"] == pytest.approx(0.28371458794131676)
    assert selector.features_to_drop_ == ["age", "sex", "s2", "s3", "s4", "s6"]


def test_cv_with_groups(make_df, data_with_groups):
    data, groups = data_with_groups
    X, y = _split_target(make_df, data)
    selector = RecursiveFeatureAddition(
        LinearRegression(),
        scoring="r2",
        cv=GroupKFold(n_splits=3),
        groups=groups,
        threshold=0.05,
    )
    selector.fit(X, y)

    assert selector.performance_drifts_ == pytest.approx(
        {
            "var_0": 0,
            "var_1": 0.07658061969570773,
            "var_4": 0.0015218369914395957,
            "var_2": -0.009235024059350394,
            "var_3": -0.002472987677324623,
        }
    )
    assert selector.features_to_drop_ == ["var_2", "var_3", "var_4"]

    indices = GroupKFold(n_splits=3).split(X, y, groups)
    selector_indices = RecursiveFeatureAddition(
        LinearRegression(), scoring="r2", cv=indices, threshold=0.05
    )
    selector_indices.fit(X, y)

    assert selector_indices.performance_drifts_ == pytest.approx(
        selector.performance_drifts_
    )
    assert selector_indices.features_to_drop_ == ["var_2", "var_3", "var_4"]


@pytest.mark.parametrize("target_type", [list, np.array])
def test_list_and_array_target(make_df, data_diabetes, target_type):
    X, _ = _split_target(make_df, data_diabetes)
    y = target_type(data_diabetes["target"])
    selector = RecursiveFeatureAddition(LinearRegression(), scoring="r2", cv=3)
    selector.fit(X, y)

    assert selector.performance_drifts_["s5"] == pytest.approx(0.28371458794131676)
    assert selector.features_to_drop_ == ["age", "sex", "s2", "s3", "s4", "s6"]


def test_error_if_transform_before_fit(make_df, data_diabetes):
    X, _ = _split_target(make_df, data_diabetes)
    msg = (
        "This RecursiveFeatureAddition instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        RecursiveFeatureAddition(LinearRegression()).transform(X)


def test_feature_importances_are_pandas_series(data_diabetes):
    X, y = _split_target(pd.DataFrame, data_diabetes)
    selector = RecursiveFeatureAddition(LinearRegression(), scoring="r2", cv=3)
    selector.fit(X, y)

    pd.testing.assert_series_equal(
        selector.feature_importances_,
        pd.Series(
            [
                750.0238715216746,
                741.4713367752565,
                522.3301645404875,
                436.67158399913274,
                322.0918016880965,
                238.6195264502995,
                182.174833735295,
                113.96599187843395,
                64.76841724774016,
                41.41804062408145,
            ],
            index=["s1", "s5", "bmi", "s2", "bp", "sex", "s4", "s3", "s6", "age"],
        ),
    )


def test_integer_column_names(data_diabetes):
    X, y = _split_target(pd.DataFrame, data_diabetes)
    X.columns = list(range(10))
    selector = RecursiveFeatureAddition(LinearRegression(), scoring="r2", cv=3)
    Xt = selector.fit_transform(X, y)

    assert selector.features_to_drop_ == [0, 1, 5, 6, 7, 9]
    assert list(selector.performance_drifts_) == [4, 8, 2, 5, 3, 1, 7, 6, 9, 0]
    pd.testing.assert_frame_equal(Xt, X[[2, 3, 4, 8]])
