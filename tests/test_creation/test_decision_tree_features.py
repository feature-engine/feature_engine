import warnings

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from feature_engine.creation import DecisionTreeFeatures
from tests.estimator_checks.fit_functionality_checks import check_return_empty

DATA = {
    "Name": [
        "tom",
        "nick",
        "krish",
        "megan",
        "peter",
        "jordan",
        "fred",
        "sam",
        "alexa",
        "brittany",
    ],
    "Age": [20, 44, 19, 33, 51, 40, 41, 37, 30, 54],
    "Height": [164, 150, 178, 158, 188, 190, 168, 174, 176, 171],
    "Marks": [1.0, 0.8, 0.6, 0.1, 0.3, 0.4, 0.8, 0.6, 0.5, 0.2],
}
REGRESSION_Y = [4.1, 5.8, 3.9, 6.2, 4.3, 4.5, 7.2, 4.4, 4.1, 6.7]
BINARY_Y = [1, 1, 1, 0, 0, 1, 0, 1, 0, 0]
MULTICLASS_Y = [1, 1, 2, 2, 0, 1, 0, 1, 0, 0]

COMBOS = [
    "Age",
    "Height",
    "Marks",
    ["Age", "Height"],
    ["Age", "Marks"],
    ["Height", "Marks"],
    ["Age", "Height", "Marks"],
]


def _select(X, combo):
    cols = combo if isinstance(combo, list) else [combo]
    return nw.from_native(X, eager_only=True).select(cols).to_native()


def _expected_tree_predictions(
    X,
    y,
    scoring,
    random_state,
    regression=True,
    binary=False,
    precision=None,
    param_grid=None,
):
    # Fits a fresh GridSearchCV per combo on the same backend as X, so this
    # works as the reference for both pandas and polars input alike.
    if param_grid is None:
        param_grid = {"max_depth": [1, 2, 3, 4]}
    if regression is True:
        est = DecisionTreeRegressor(random_state=random_state)
    else:
        est = DecisionTreeClassifier(random_state=random_state)
    tree = GridSearchCV(est, cv=3, scoring=scoring, param_grid=param_grid)

    expected = {}
    for combo in COMBOS:
        X_sub = _select(X, combo)
        tree.fit(X_sub, y)
        if regression is True:
            preds = tree.predict(X_sub)
        elif binary is True:
            preds = tree.predict_proba(X_sub)[:, 1]
        else:
            preds = tree.predict(X_sub)
        if precision is not None:
            preds = np.round(preds, precision)
        expected[f"tree({combo})"] = list(preds)
    return expected


def assert_df_equal(X, expected: dict) -> None:
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(expected.keys())
    for col, values in expected.items():
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            assert result[col] == pytest.approx(values, abs=1e-6)
        else:
            assert result[col] == values


@pytest.mark.parametrize("precision", ["string", 0.1, -1, np.nan])
def test_error_if_precision_gets_not_permitted_value(precision):
    msg = "precision must be None or a positive integer. " f"Got {precision} instead."
    with pytest.raises(ValueError, match=msg):
        DecisionTreeFeatures(precision=precision)


@pytest.mark.parametrize("regression", ["string", 0.1, -1, np.nan])
def test_error_if_regression_gets_not_permitted_value(regression):
    msg = f"regression must be a boolean value. Got {regression} instead."
    with pytest.raises(ValueError, match=msg):
        DecisionTreeFeatures(regression=regression)


@pytest.mark.parametrize("drop", ["string", 0.1, -1, np.nan])
def test_error_if_drop_original_gets_not_permitted_value(drop):
    msg = (
        "drop_original takes only boolean values True and False. "
        f"Got {drop} instead."
    )
    with pytest.raises(ValueError, match=msg):
        DecisionTreeFeatures(drop_original=drop)


@pytest.mark.parametrize(
    "input_features, expected",
    [
        (1, ["vara", "varb", "varc"]),
        (
            2,
            [
                "vara",
                "varb",
                "varc",
                ["vara", "varb"],
                ["vara", "varc"],
                ["varb", "varc"],
            ],
        ),
        (
            3,
            [
                "vara",
                "varb",
                "varc",
                ["vara", "varb"],
                ["vara", "varc"],
                ["varb", "varc"],
                ["vara", "varb", "varc"],
            ],
        ),
        (
            4,
            [
                "vara",
                "varb",
                "varc",
                ["vara", "varb"],
                ["vara", "varc"],
                ["varb", "varc"],
                ["vara", "varb", "varc"],
            ],
        ),
        (
            100,
            [
                "vara",
                "varb",
                "varc",
                ["vara", "varb"],
                ["vara", "varc"],
                ["varb", "varc"],
                ["vara", "varb", "varc"],
            ],
        ),
    ],
)
def test_create_variable_combinations_when_int(input_features, expected):
    vars = ["vara", "varb", "varc"]
    transformer = DecisionTreeFeatures()
    combos = transformer._create_variable_combinations(
        variables=vars, how_to_combine=input_features
    )
    assert combos == expected


@pytest.mark.parametrize(
    "vars, expected",
    [
        (
            ["vara", "varb", "varc"],
            [
                "vara",
                "varb",
                "varc",
                ["vara", "varb"],
                ["vara", "varc"],
                ["varb", "varc"],
                ["vara", "varb", "varc"],
            ],
        ),
        (["vara", "varb"], ["vara", "varb", ["vara", "varb"]]),
        (["vara"], ["vara"]),
    ],
)
def test_create_variable_combinations_when_None(vars, expected):
    transformer = DecisionTreeFeatures()
    combos = transformer._create_variable_combinations(
        variables=vars, how_to_combine=None
    )
    assert combos == expected


@pytest.mark.parametrize(
    "input_features, expected",
    [
        (
            [2, 3],
            [
                ["vara", "varb"],
                ["vara", "varc"],
                ["varb", "varc"],
                ["vara", "varb", "varc"],
            ],
        ),
        ([1, 3], ["vara", "varb", "varc", ["vara", "varb", "varc"]]),
    ],
)
def test_create_variable_combinations_when_list(input_features, expected):
    vars = ["vara", "varb", "varc"]
    transformer = DecisionTreeFeatures()
    combos = transformer._create_variable_combinations(
        variables=vars, how_to_combine=input_features
    )
    assert combos == expected


@pytest.mark.parametrize(
    "input_features, expected",
    [
        (
            (("vara", "varb"), ("vara"), ("vara", "varb", "varc")),
            [["vara", "varb"], "vara", ["vara", "varb", "varc"]],
        ),
        ((("vara", "varc"), ("vara", "varb")), [["vara", "varc"], ["vara", "varb"]]),
    ],
)
def test_create_variable_combinations_when_tuple(input_features, expected):
    vars = ["vara", "varb", "varc"]
    transformer = DecisionTreeFeatures()
    combos = transformer._create_variable_combinations(
        variables=vars, how_to_combine=input_features
    )
    assert combos == expected


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_feature_creation_regression(make_df):
    X = make_df(DATA)
    scoring = "neg_mean_squared_error"
    rs = 0
    tr = DecisionTreeFeatures(scoring=scoring, random_state=rs)
    Xt = tr.fit_transform(X, REGRESSION_Y)

    expected = dict(DATA)
    expected.update(_expected_tree_predictions(X, REGRESSION_Y, scoring, rs))
    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_feature_creation_regression_and_precision(make_df):
    X = make_df(DATA)
    scoring = "neg_mean_squared_error"
    rs = 0
    tr = DecisionTreeFeatures(scoring=scoring, random_state=rs, precision=1)
    Xt = tr.fit_transform(X, REGRESSION_Y)

    expected = dict(DATA)
    expected.update(
        _expected_tree_predictions(X, REGRESSION_Y, scoring, rs, precision=1)
    )
    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_feature_creation_regression_drop_original(make_df):
    X = make_df(DATA)
    scoring = "neg_mean_squared_error"
    rs = 0
    tr = DecisionTreeFeatures(scoring=scoring, random_state=rs, drop_original=True)
    Xt = tr.fit_transform(X, REGRESSION_Y)

    expected = {"Name": DATA["Name"]}
    expected.update(_expected_tree_predictions(X, REGRESSION_Y, scoring, rs))
    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_feature_creation_binary_classif(make_df):
    X = make_df(DATA)
    scoring = "roc_auc"
    rs = 0
    tr = DecisionTreeFeatures(scoring=scoring, random_state=rs, regression=False)
    Xt = tr.fit_transform(X, BINARY_Y)

    expected = dict(DATA)
    expected.update(
        _expected_tree_predictions(
            X, BINARY_Y, scoring, rs, regression=False, binary=True
        )
    )
    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_feature_creation_binary_classif_w_precision(make_df):
    X = make_df(DATA)
    scoring = "roc_auc"
    rs = 0
    tr = DecisionTreeFeatures(
        scoring=scoring, random_state=rs, regression=False, precision=2
    )
    Xt = tr.fit_transform(X, BINARY_Y)

    expected = dict(DATA)
    expected.update(
        _expected_tree_predictions(
            X, BINARY_Y, scoring, rs, regression=False, binary=True, precision=2
        )
    )
    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_feature_creation_binary_multiclass(make_df):
    X = make_df(DATA)
    scoring = "roc_auc"
    rs = 0
    tr = DecisionTreeFeatures(scoring=scoring, random_state=rs, regression=False)
    Xt = tr.fit_transform(X, MULTICLASS_Y)

    expected = dict(DATA)
    expected.update(
        _expected_tree_predictions(
            X, MULTICLASS_Y, scoring, rs, regression=False, binary=False
        )
    )
    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out(make_df):
    X = make_df(DATA)
    tr = DecisionTreeFeatures(variables=["Age", "Marks"])
    Xt = tr.fit_transform(X, REGRESSION_Y)
    feat_out = list(nw.from_native(Xt, eager_only=True).columns)
    assert tr.get_feature_names_out() == feat_out
    assert tr.get_feature_names_out(list(DATA.keys())) == feat_out


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out_from_pipeline(make_df):
    X = make_df(DATA)
    tr = DecisionTreeFeatures(variables=["Age", "Marks"])
    pipe = Pipeline([("transformer", tr)])
    Xt = pipe.fit_transform(X, REGRESSION_Y)
    feat_out = list(nw.from_native(Xt, eager_only=True).columns)
    assert pipe.get_feature_names_out(input_features=None) == feat_out
    assert pipe.get_feature_names_out(input_features=list(DATA.keys())) == feat_out


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("_input_features", ["hola", ["Age", "Marks"]])
def test_get_feature_names_out_raises_error_when_wrong_param(make_df, _input_features):
    X = make_df(DATA)
    tr = DecisionTreeFeatures(variables=["Age", "Marks"])
    tr.fit(X, REGRESSION_Y)
    with pytest.raises(ValueError):
        tr.get_feature_names_out(input_features=_input_features)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_when_regression_true_and_target_binary(make_df):
    X = make_df(DATA)
    tr = DecisionTreeFeatures(regression=True)

    msg = (
        "Trying to fit a regression to a binary target is not "
        "allowed by this transformer. Check the target values "
        "or set regression to False."
    )
    with pytest.raises(ValueError, match=msg):
        tr.fit(X, BINARY_Y)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_user_enter_param_grid(make_df):
    X = make_df(DATA)
    scoring = "roc_auc"
    rs = 0
    grid = {"max_depth": [1, 2, 3, 4]}
    tr = DecisionTreeFeatures(
        scoring=scoring, random_state=rs, regression=False, param_grid=grid
    )
    Xt = tr.fit_transform(X, BINARY_Y)

    expected = dict(DATA)
    expected.update(
        _expected_tree_predictions(
            X, BINARY_Y, scoring, rs, regression=False, binary=True, param_grid=grid
        )
    )
    assert_df_equal(Xt, expected)


def test_check_return_empty():
    # DecisionTreeFeatures is not part of the check_feature_engine_estimator
    # pipeline (test_check_estimator_creation.py only feeds MathFeatures,
    # RelativeFeatures and CyclicalFeatures into it), so return_empty is
    # tested directly here instead. check_return_empty is a shared,
    # pandas-only estimator-check helper used across the library.
    check_return_empty(DecisionTreeFeatures(regression=False))


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_n_jobs_parallel_matches_sequential(make_df):
    # core correctness check for n_jobs: parallelizing tree training across
    # feature combinations must produce identical trees, and therefore
    # identical predictions, to sequential training (n_jobs=None).
    X = make_df(DATA)
    tr_seq = DecisionTreeFeatures(n_jobs=None, random_state=0)
    tr_seq.fit(X, REGRESSION_Y)
    tr_par = DecisionTreeFeatures(n_jobs=2, random_state=0)
    tr_par.fit(X, REGRESSION_Y)

    Xt_seq = tr_seq.transform(X)
    Xt_par = tr_par.transform(X)

    expected = nw.from_native(Xt_seq, eager_only=True).to_dict(as_series=False)
    assert_df_equal(Xt_par, expected)


def test_transform_does_not_fragment_pandas_output():
    # regression test: transform() used to assign one new tree column at a
    # time (X[col_name] = preds), which triggers pandas' "DataFrame is
    # highly fragmented" PerformanceWarning once there are enough feature
    # combinations - fixed by building all new columns in one DataFrame
    # and joining once. Needs enough variables to cross pandas' internal
    # fragmentation threshold (a handful of combos won't trigger it).
    rng = np.random.RandomState(0)
    n_vars = 9
    X = pd.DataFrame(
        rng.rand(200, n_vars), columns=[f"v{i}" for i in range(n_vars)]
    )
    y = rng.rand(200)

    tr = DecisionTreeFeatures(
        features_to_combine=3, param_grid={"max_depth": [1, 2]}, random_state=0
    )
    tr.fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        tr.transform(X)


def test_single_int_named_feature_combo():
    # regression test: a single-variable combo with an integer column name
    # used to crash (isinstance(features, str) missed the int case), since
    # X[features] for a bare int returns a 1D Series, not the 2D input
    # sklearn requires - fixed to check isinstance(features, (str, int)).
    # Integer column names are pandas-only - polars requires string columns.
    df = pd.DataFrame({0: [1.0, 2, 3, 4, 5, 6, 7, 8], 1: [2.0, 3, 4, 5, 6, 7, 8, 9]})
    y = [1.0, 2, 3, 4, 5, 6, 7, 8]
    transformer = DecisionTreeFeatures(features_to_combine=1, random_state=0)
    transformer.fit(df, y)
    Xt = transformer.transform(df)
    assert "tree(0)" in Xt.columns
    assert "tree(1)" in Xt.columns
