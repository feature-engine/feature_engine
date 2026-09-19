import re

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold

from feature_engine.selection import MRMR
from tests.backend_helpers import frame_to_dict, make_series

VARIABLES = [f"var_{i}" for i in range(12)]
RF_GRID = {"max_depth": [2, 3], "n_estimators": [10]}

MI_CLASSIF = [
    0.052936601624909096,
    0.012431039630421026,
    0.0031179453717387062,
    0.0004679804673619614,
    0.5770540935952642,
    0.011045380844762365,
    0.5728270276712453,
    0.6566020567069197,
    0.012040401687987368,
    0.5035424468925758,
    0.0,
    0.0,
]
MI_REGRESSION = [
    0.4827568966040907,
    0.1498525858592843,
    0.0,
    0.04212366814778079,
    0.1824019466447857,
    0.011297428359838158,
    0.19292720067399838,
    0.2236206517399788,
    0.5277855986492561,
    0.22394827371463455,
    0.0,
    0.002100737962146937,
]
F_CLASSIF = [
    79.04849944065174,
    7.852911596746189,
    1.4628293436965305,
    0.7454302324350504,
    2360.5567490946137,
    0.78416950052947,
    2296.80729048166,
    5206.0776362490415,
    0.059549527394634295,
    1739.598743686081,
    0.0453654613398977,
    0.5613535054454387,
]
F_REGRESSION = [
    1426.4396061315033,
    424.0297119316108,
    3.060543877897646,
    65.31838343563534,
    100.41062036724897,
    1.622842273663943,
    105.20415656422067,
    4.142473196617828,
    1683.2517259758326,
    160.1177438792669,
    0.07650822171810838,
    0.18645998061038163,
]
RF_CLASSIF = [
    0.022247324184615325,
    0.003955855934811008,
    0.0011991561893078399,
    0.00026914064555155195,
    0.19395815835718982,
    0.0004983747672734107,
    0.09261141003009202,
    0.41634361330706404,
    0.004294463160439444,
    0.26440681633767793,
    0.00019819544535523348,
    1.7491640622450164e-05,
]
RF_REGRESSION = [
    0.08500311779594995,
    0.21949616499172103,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.695500717212329,
    0.0,
    0.0,
    0.0,
]


@pytest.fixture(scope="module")
def data_regression(data_classification):
    data = {k: v for k, v in data_classification.items() if k != "target"}
    data["target"] = [
        0.5 * a - 0.3 * b + c
        for a, b, c in zip(data["var_1"], data["var_3"], data["var_8"])
    ]
    return data


def _split_target(make_df, data):
    X = make_df({k: v for k, v in data.items() if k != "target"})
    return X, make_series(make_df, data["target"])


# init parameters
@pytest.mark.parametrize("method", [10, "string", False, None, ["MIQ"]])
def test_error_if_method_not_allowed(method):
    msg = (
        "method must be one of 'MIQ', 'MID', 'FCQ', 'FCD', 'RFCQ'. "
        f"Got {method} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MRMR(method=method)


@pytest.mark.parametrize("max_features", ["string", -1, 0, 1.5, [0, 1]])
def test_error_if_max_features_not_positive_integer(max_features):
    msg = (
        "max_features must be an integer with the number of features to "
        f"select. Got {max_features} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MRMR(max_features=max_features)


@pytest.mark.parametrize(
    "variables, max_features",
    [(["var1", "var2"], 2), (["var1", "var2"], 3), (["var1", "var2", "var3"], 4)],
)
def test_error_if_max_features_not_less_than_number_of_variables(
    variables, max_features
):
    msg = (
        f"The number of variables to examine is {len(variables)}, which is "
        "less than or equal to the number of features to select indicated "
        f"in `max_features`, which is {max_features}. Please check the "
        "values entered in the parameters `variables` and `max_features`."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MRMR(variables=variables, max_features=max_features)


@pytest.mark.parametrize("scoring", ["roc_auc", "accuracy", "precision"])
def test_error_if_scoring_not_for_regression(scoring):
    msg = (
        f"The metric {scoring} is not suitable for regression. Set the "
        "parameter regression to False or choose a different performance "
        "metric."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MRMR(method="RFCQ", regression=True, scoring=scoring)


@pytest.mark.parametrize("scoring", ["mse", "mae", "r2"])
def test_error_if_scoring_not_for_classification(scoring):
    msg = (
        f"The metric {scoring} is not suitable for classification. Set the "
        "parameter regression to True or choose a different performance "
        "metric."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MRMR(method="RFCQ", regression=False, scoring=scoring)


@pytest.mark.parametrize(
    "method, max_features, discrete_features, n_neighbors, scoring, cv, "
    "param_grid, regression, confirm_variables, random_state, n_jobs",
    [
        ("MIQ", None, "auto", 3, "roc_auc", 3, None, False, False, None, None),
        ("MID", 5, [True, False], 5, "r2", 5, None, True, True, 0, 2),
        ("FCQ", 1, False, 1, "accuracy", KFold(), None, False, False, 42, -1),
        ("FCD", 10, True, 3, "mse", 3, None, True, False, None, None),
        ("RFCQ", 3, "auto", 3, "r2", 2, RF_GRID, True, True, 1, None),
        ("RFCQ", 3, "auto", 3, "accuracy", 2, None, False, False, 1, None),
    ],
)
def test_init_param_assignment(
    method,
    max_features,
    discrete_features,
    n_neighbors,
    scoring,
    cv,
    param_grid,
    regression,
    confirm_variables,
    random_state,
    n_jobs,
):
    sel = MRMR(
        method=method,
        max_features=max_features,
        discrete_features=discrete_features,
        n_neighbors=n_neighbors,
        scoring=scoring,
        cv=cv,
        param_grid=param_grid,
        regression=regression,
        confirm_variables=confirm_variables,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    assert sel.method == method
    assert sel.max_features == max_features
    assert sel.discrete_features == discrete_features
    assert sel.n_neighbors == n_neighbors
    assert sel.scoring == scoring
    assert sel.cv is cv
    assert sel.param_grid == param_grid
    assert sel.regression is regression
    assert sel.confirm_variables is confirm_variables
    assert sel.random_state == random_state
    assert sel.n_jobs == n_jobs


# fit and transform
@pytest.mark.parametrize(
    "method, relevance, selected",
    [
        ("MIQ", MI_CLASSIF, ["var_6", "var_7", "var_10", "var_11"]),
        ("MID", MI_CLASSIF, ["var_1", "var_4", "var_5", "var_7"]),
        ("FCQ", F_CLASSIF, ["var_4", "var_6", "var_7", "var_9"]),
        ("FCD", F_CLASSIF, ["var_4", "var_6", "var_7", "var_9"]),
        ("RFCQ", RF_CLASSIF, ["var_0", "var_4", "var_7", "var_9"]),
    ],
)
def test_fit_transform_classification(
    make_df, data_classification, method, relevance, selected
):
    X, y = _split_target(make_df, data_classification)
    sel = MRMR(
        method=method,
        max_features=4,
        param_grid=RF_GRID,
        regression=False,
        random_state=0,
    )
    Xt = sel.fit_transform(X, y)

    assert sel.variables_ == VARIABLES
    assert sel.relevance_.tolist() == pytest.approx(relevance)
    assert sel.features_to_drop_ == [f for f in VARIABLES if f not in selected]
    assert sel.feature_names_in_ == VARIABLES
    assert sel.n_features_in_ == 12
    assert isinstance(Xt, make_df)
    assert list(frame_to_dict(Xt)) == selected
    assert frame_to_dict(Xt) == {f: data_classification[f] for f in selected}


@pytest.mark.parametrize(
    "method, relevance, selected",
    [
        ("MIQ", MI_REGRESSION, ["var_1", "var_2", "var_5", "var_8"]),
        ("MID", MI_REGRESSION, ["var_1", "var_3", "var_7", "var_8"]),
        ("FCQ", F_REGRESSION, ["var_0", "var_1", "var_3", "var_8"]),
        ("FCD", F_REGRESSION, ["var_0", "var_1", "var_8", "var_9"]),
        ("RFCQ", RF_REGRESSION, ["var_0", "var_1", "var_2", "var_8"]),
    ],
)
def test_fit_transform_regression(
    make_df, data_regression, method, relevance, selected
):
    X, y = _split_target(make_df, data_regression)
    sel = MRMR(
        method=method,
        max_features=4,
        scoring="r2",
        param_grid=RF_GRID,
        regression=True,
        random_state=0,
    )
    Xt = sel.fit_transform(X, y)

    assert sel.relevance_.tolist() == pytest.approx(relevance)
    assert sel.features_to_drop_ == [f for f in VARIABLES if f not in selected]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {f: data_regression[f] for f in selected}


def test_default_max_features_selects_20_percent(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = MRMR(method="FCQ").fit(X, y)
    assert sel.features_to_drop_ == [
        f for f in VARIABLES if f not in ["var_4", "var_7"]
    ]


def test_variables_and_discrete_features(make_df, data_classification):
    data = dict(data_classification)
    data["var_1"] = [round(2 * v) for v in data["var_1"]]
    data["var_5"] = [round(v) for v in data["var_5"]]
    X, y = _split_target(make_df, data)
    sel = MRMR(
        variables=["var_0", "var_1", "var_4", "var_5", "var_7"],
        method="MID",
        max_features=3,
        discrete_features=[False, True, False, True, False],
        random_state=0,
    )
    Xt = sel.fit_transform(X, y)

    assert sel.variables_ == ["var_0", "var_1", "var_4", "var_5", "var_7"]
    assert sel.relevance_.tolist() == pytest.approx(
        [
            0.052936601624909096,
            0.01026964337663472,
            0.5770540935952642,
            0.0017639968720600564,
            0.6566020567069197,
        ]
    )
    assert sel.features_to_drop_ == ["var_0", "var_4"]
    assert isinstance(Xt, make_df)
    assert list(frame_to_dict(Xt)) == [
        f for f in VARIABLES if f not in ["var_0", "var_4"]
    ]


def test_confirm_variables(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = MRMR(
        variables=["var_0", "var_2", "var_4", "var_6", "var_8", "var_20"],
        method="MIQ",
        max_features=3,
        confirm_variables=True,
        random_state=0,
    )
    sel.fit(X, y)

    assert sel.variables_ == ["var_0", "var_2", "var_4", "var_6", "var_8"]
    assert sel.relevance_.tolist() == pytest.approx(
        [
            0.052936601624909096,
            0.0031179453717387062,
            0.5770540935952642,
            0.5728270276712453,
            0.012040401687987368,
        ]
    )
    assert sel.features_to_drop_ == ["var_0", "var_8"]


def test_cv_generator(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    sel = MRMR(
        method="RFCQ",
        max_features=3,
        param_grid=RF_GRID,
        cv=StratifiedKFold(n_splits=3).split(X, y),
        random_state=0,
    )
    sel.fit(X, y)

    assert sel.relevance_.tolist() == pytest.approx(RF_CLASSIF)
    assert sel.features_to_drop_ == [
        f for f in VARIABLES if f not in ["var_4", "var_7", "var_9"]
    ]


def test_missing_values_with_random_forests(make_df, data_classification):
    # only random forests take missing values; the redundance then uses the rows
    # where both features have values.
    data = dict(data_classification)
    for var, step in [("var_2", 7), ("var_5", 5), ("var_9", 3)]:
        data[var] = [None if i % step == 0 else v for i, v in enumerate(data[var])]
    X, y = _split_target(make_df, data)
    sel = MRMR(method="RFCQ", max_features=4, param_grid=RF_GRID, random_state=0)
    sel.fit(X, y)

    assert sel.relevance_.tolist() == pytest.approx(
        [
            0.0076636392509130055,
            0.005491342491500148,
            0.0037204783072902247,
            0.0,
            0.2392943866283686,
            3.639947920429752e-05,
            0.17849998892103483,
            0.46742647746615623,
            0.0011218247849137536,
            0.09508246490768371,
            0.0016629977629352137,
            0.0,
        ]
    )
    assert sel.features_to_drop_ == [
        f for f in VARIABLES if f not in ["var_4", "var_6", "var_7", "var_10"]
    ]


@pytest.mark.parametrize("target_type", [list, np.array])
def test_target_as_list_or_array(make_df, data_classification, target_type):
    X, _ = _split_target(make_df, data_classification)
    y = target_type(data_classification["target"])
    sel = MRMR(method="FCD", max_features=3).fit(X, y)

    assert sel.relevance_.tolist() == pytest.approx(F_CLASSIF)
    assert sel.features_to_drop_ == [
        f for f in VARIABLES if f not in ["var_4", "var_6", "var_7"]
    ]


def test_error_if_less_than_2_variables(make_df, data_classification):
    X, y = _split_target(make_df, data_classification)
    msg = (
        "The selector needs at least 2 or more variables to select from. "
        "Got only 1 variable: ['var_0']."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MRMR(variables=["var_0"]).fit(X, y)


def test_error_if_transform_before_fit(make_df, data_classification):
    X, _ = _split_target(make_df, data_classification)
    msg = (
        "This MRMR instance is not fitted yet. Call 'fit' with appropriate "
        "arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        MRMR().transform(X)


@pytest.mark.parametrize("method", ["MIQ", "FCQ"])
def test_integer_column_names(data_classification, method):
    data = {i: data_classification[f"var_{i}"] for i in range(12)}
    X = pd.DataFrame(data)
    y = pd.Series(data_classification["target"])
    sel = MRMR(method=method, max_features=4, random_state=0)
    Xt = sel.fit_transform(X, y)

    expected = {"MIQ": [6, 7, 10, 11], "FCQ": [4, 6, 7, 9]}[method]
    assert sel.variables_ == list(range(12))
    assert sel.features_to_drop_ == [i for i in range(12) if i not in expected]
    pd.testing.assert_frame_equal(Xt, X[expected])


def test_keeps_pandas_index(data_classification):
    X, y = _split_target(pd.DataFrame, data_classification)
    X.index = X.index + 100
    y.index = X.index
    Xt = MRMR(method="FCQ", max_features=4).fit_transform(X, y)
    pd.testing.assert_frame_equal(Xt, X[["var_4", "var_6", "var_7", "var_9"]])
