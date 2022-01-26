import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor

from feature_engine.selection import RecursiveFeatureAddition


_input_params = [
    (RandomForestClassifier(), 'roc_auc', 3, 0.1, None),
    (LinearRegression(), "neg_mean_squared_error", KFold(), 0.01, ['var_a', 'var_b']),
    (DecisionTreeRegressor(), 'r2', StratifiedKFold(), 0.5, ['var_a']),
    (RandomForestClassifier(), 'accuracy', 5, 0.002, 'var_a'),
]


@pytest.mark.parametrize(
    "_estimator, _scoring, _cv, _threshold, _variables", _input_params
)
def test_input_params_assignment(
    _estimator, _scoring, _cv, _threshold, _variables
):
    sel = RecursiveFeatureAddition(
        estimator=_estimator,
        scoring=_scoring,
        cv=_cv,
        threshold=_threshold,
        variables=_variables,
    )

    assert sel.estimator==_estimator
    assert sel.scoring==_scoring
    assert sel.cv==_cv
    assert sel.threshold==_threshold
    assert sel.variables==_variables


def test_raises_error_when_no_estimator_passed():
    with pytest.raises(TypeError):
        RecursiveFeatureAddition()


_thresholds = [None, [0.1], "a_string"]

@pytest.mark.parametrize("_thresholds", _thresholds)
def test_raises_threshold_error(_thresholds):
    with pytest.raises(ValueError):
        RecursiveFeatureAddition(RandomForestClassifier(), threshold=_thresholds)

_not_a_df = [
    "not_a_df",
    [1, 2, 3, "some_data"],
    pd.Series([-2, 1.5, 8.94], name="not_a_df"),
]


@pytest.mark.parametrize("_not_a_df", _not_a_df)
def test_raises_error_when_fitting_not_a_df(_not_a_df):
    transformer = RecursiveFeatureAddition(RandomForestClassifier())
    # trying to fit not a df
    with pytest.raises(TypeError):
        transformer.fit(_not_a_df)


_variables = ["var_1", ["var_2"],["var_1", "var_2", "var_3", "var_11"], None]


@pytest.mark.parametrize("_variables", _variables)
def test_variables_params(_variables, df_test):
    X, y = df_test
    sel = RecursiveFeatureAddition(
        RandomForestClassifier(), variables=_variables
    ).fit(X, y)

    if _variables is not None:
        assert sel.variables == _variables
        if isinstance(_variables, list):
            assert sel.variables_ == _variables
        else:
            assert sel.variables_ == [_variables]
    else:
        assert sel.variables is None
        assert sel.variables_ == ["var_" + str(i) for i in range(12)]

    # test selector excludes non-numerical variables automatically
    X['cat_var'] =  ['A']*1000
    sel = RecursiveFeatureAddition(
        RandomForestClassifier(), variables=None
    ).fit(X, y)
    assert sel.variables is None
    assert sel.variables_ == ["var_" + str(i) for i in range(12)]


def test_raises_error_when_user_passes_categorical_var(df_test):
    X, y = df_test

    # add categorical variable
    X['cat_var'] = ['A'] * 1000

    with pytest.raises(TypeError):
        RecursiveFeatureAddition(
            RandomForestClassifier(), variables=["var_1", "var_2", "cat_var"]
        ).fit(X, y)

    with pytest.raises(TypeError):
        RecursiveFeatureAddition(
            RandomForestClassifier(), variables="cat_var").fit(X, y)


def test_classification_threshold_parameters(df_test):
    X, y = df_test

    sel = RecursiveFeatureAddition(
        RandomForestClassifier(random_state=1), threshold=0.001
    )

    sel.fit(X, y)

    # expected result
    Xtransformed = X[["var_7", "var_10"]].copy()

    # # expected ordered features by importance, from most important
    # # to least important
    # ordered_features = [
    #     "var_7",
    #     "var_4",
    #     "var_6",
    #     "var_9",
    #     "var_0",
    #     "var_8",
    #     "var_1",
    #     "var_10",
    #     "var_5",
    #     "var_11",
    #     "var_2",
    #     "var_3",
    # ]

    # test fit attrs
    assert np.round(sel.initial_model_performance_, 3) == 0.997
    # assert sel.feature_importances_ ==
    assert sel.features_to_drop_ == [
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_6",
        "var_8",
        "var_9",
        "var_11",
    ]
    assert len(sel.performance_drifts_.keys()) == len(X.columns)
    assert all([var in sel.performance_drifts_.keys() for var in X.columns])
    assert sel.n_features_in_ == len(X.columns)

    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_3_and_r2(load_diabetes_dataset):
    #  test for regression using cv=3, and the r2 as metric.
    X, y = load_diabetes_dataset

    kfold = KFold(n_splits=3, shuffle=True, random_state=10)
    sel = RecursiveFeatureAddition(estimator=LinearRegression(), scoring="r2", cv=kfold, threshold=0.001)
    sel.fit(X, y)

    # expected output
    Xtransformed = X[[1, 2, 3, 6, 8]].copy()

    # expected ordered features by importance, from most important
    # to least important
    ordered_features = [4, 8, 2, 5, 3, 1, 7, 6, 9, 0]

    # test init params
    # assert sel.cv == 3
    assert sel.variables is None
    assert sel.scoring == "r2"
    assert sel.threshold == 0.001
    # fit params
    assert sel.variables_ == list(X.columns)
    assert np.round(sel.initial_model_performance_, 2) == 0.49
    print(sel.performance_drifts_)
    assert sel.features_to_drop_ == [0, 4, 5, 7, 9]
    assert len(sel.performance_drifts_.keys()) == len(ordered_features)
    assert all([var in sel.performance_drifts_.keys() for var in ordered_features])

    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_regression_cv_2_and_mse(load_diabetes_dataset):
    #  test for regression using cv=2, and the neg_mean_squared_error as metric.
    # add suitable threshold for regression mse
    X, y = load_diabetes_dataset

    kfold = KFold(n_splits=2, shuffle=True, random_state=10)
    sel = RecursiveFeatureAddition(
        estimator=DecisionTreeRegressor(random_state=0),
        scoring="neg_mean_squared_error",
        cv=kfold,
        threshold=10,
    )
    # fit transformer
    sel.fit(X, y)

    # expected output
    Xtransformed = X[[1, 2, 7]].copy()

    # expected ordred features by importance, from most important
    # to least important
    ordered_features = [2, 8, 5, 7, 3, 9, 6, 4, 0, 1]

    # test init params
    assert sel.cv == 2
    assert sel.variables is None
    assert sel.scoring == "neg_mean_squared_error"
    assert sel.threshold == 10
    # fit params
    assert sel.variables_ == list(X.columns)
    assert np.round(sel.initial_model_performance_, 0) == -5836.0
    assert sel.features_to_drop_ == [0, 3, 4, 5, 6, 8, 9]
    assert list(sel.performance_drifts_.keys()) == ordered_features
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_non_fitted_error(df_test):
    # when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        sel = RecursiveFeatureAddition(RandomForestClassifier(random_state=1))
        sel.transform(df_test)





def test_automatic_variable_selection(df_test):
    X, y = df_test

    # add 2 additional categorical variables, these should not be evaluated by
    # the selector
    X["cat_1"] = "cat1"
    X["cat_2"] = "cat2"

    sel = RecursiveFeatureAddition(
        RandomForestClassifier(random_state=1), threshold=0.001
    )
    sel.fit(X, y)

    # expected result
    Xtransformed = X[["var_7", "var_10", "cat_1", "cat_2"]].copy()

    # expected ordered features by importance, from most important
    # to least important
    ordered_features = [
        "var_7",
        "var_4",
        "var_6",
        "var_9",
        "var_0",
        "var_8",
        "var_1",
        "var_10",
        "var_5",
        "var_11",
        "var_2",
        "var_3",
    ]

    # test init params
    assert sel.variables is None
    assert sel.threshold == 0.001
    assert sel.cv == 3
    assert sel.scoring == "roc_auc"
    # test fit attrs
    assert sel.variables_ == [
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
    ]
    assert np.round(sel.initial_model_performance_, 3) == 0.997
    assert sel.features_to_drop_ == [
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_6",
        "var_8",
        "var_9",
        "var_11",
    ]
    assert list(sel.performance_drifts_.keys()) == ordered_features
    # test transform output
    pd.testing.assert_frame_equal(sel.transform(X), Xtransformed)


def test_KFold_generators(df_test):

    X, y = df_test

    # Kfold
    sel = RecursiveFeatureAddition(
        RandomForestClassifier(random_state=1),
        threshold=0.001,
        cv=KFold(n_splits=3),
    )
    sel.fit(X, y)
    Xtransformed = sel.transform(X)

    # test fit attrs
    assert sel.initial_model_performance_ > 0.995
    assert isinstance(sel.features_to_drop_, list)
    assert all([x for x in sel.features_to_drop_ if x in X.columns])
    assert len(sel.features_to_drop_) < X.shape[1]
    assert not Xtransformed.empty
    assert all([x for x in Xtransformed.columns if x not in sel.features_to_drop_])
    assert isinstance(sel.performance_drifts_, dict)
    assert all([x for x in X.columns if x in sel.performance_drifts_.keys()])
    assert all(
        [
            isinstance(sel.performance_drifts_[var], (int, float))
            for var in sel.performance_drifts_.keys()
        ]
    )

    # Stratfied
    sel = RecursiveFeatureAddition(
        RandomForestClassifier(random_state=1),
        threshold=0.001,
        cv=StratifiedKFold(n_splits=3),
    )
    sel.fit(X, y)
    Xtransformed = sel.transform(X)

    # test fit attrs
    assert sel.initial_model_performance_ > 0.995
    assert isinstance(sel.features_to_drop_, list)
    assert all([x for x in sel.features_to_drop_ if x in X.columns])
    assert len(sel.features_to_drop_) < X.shape[1]
    assert not Xtransformed.empty
    assert all([x for x in Xtransformed.columns if x not in sel.features_to_drop_])
    assert isinstance(sel.performance_drifts_, dict)
    assert all([x for x in X.columns if x in sel.performance_drifts_.keys()])
    assert all(
        [
            isinstance(sel.performance_drifts_[var], (int, float))
            for var in sel.performance_drifts_.keys()
        ]
    )

    # None
    sel = RecursiveFeatureAddition(
        RandomForestClassifier(random_state=1),
        threshold=0.001,
        cv=None,
    )
    sel.fit(X, y)
    Xtransformed = sel.transform(X)

    # test fit attrs
    assert sel.initial_model_performance_ > 0.995
    assert isinstance(sel.features_to_drop_, list)
    assert all([x for x in sel.features_to_drop_ if x in X.columns])
    assert len(sel.features_to_drop_) < X.shape[1]
    assert not Xtransformed.empty
    assert all([x for x in Xtransformed.columns if x not in sel.features_to_drop_])
    assert isinstance(sel.performance_drifts_, dict)
    assert all([x for x in X.columns if x in sel.performance_drifts_.keys()])
    assert all(
        [
            isinstance(sel.performance_drifts_[var], (int, float))
            for var in sel.performance_drifts_.keys()
        ]
    )
