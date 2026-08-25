import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import DecisionTreeDiscretiser


def _normal_dist_data():
    np.random.seed(0)
    mu, sigma = 0, 0.1
    return {"var": list(np.random.normal(mu, sigma, 100))}


def _discretise_data():
    np.random.seed(42)
    mu1, sigma1 = 0, 3
    s1 = np.random.normal(mu1, sigma1, 20)
    mu2, sigma2 = 3, 5
    s2 = np.random.normal(mu2, sigma2, 20)
    return {
        "var_A": list(s1),
        "var_B": list(s2),
        "target": [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    }


def _unique_sorted(X, col):
    # polars' Series.unique() doesn't preserve first-occurrence order like
    # pandas does, so compare the resulting *set* of values, sorted, rather
    # than relying on unique()'s order (which differs across backends).
    return sorted(nw.from_native(X, eager_only=True).get_column(col).unique().to_list())


# init parameters
@pytest.mark.parametrize(
    "params",
    [("prediction", 3, True), ("bin_number", 10, False), ("boundaries", 1, False)],
)
def test_init_param_assignment(params):
    dsc = DecisionTreeDiscretiser(
        bin_output=params[0],
        precision=params[1],
        regression=params[2],
    )
    assert dsc.bin_output == params[0]
    assert dsc.precision == params[1]
    assert dsc.regression == params[2]


@pytest.mark.parametrize("bin_output_", ["arbitrary", False, 1])
def test_error_if_binoutput_not_permitted_value(bin_output_):
    msg = (
        "bin_output takes values  'prediction', 'bin_number' or 'boundaries'. "
        f"Got {bin_output_} instead."
    )
    with pytest.raises(ValueError, match=msg):
        DecisionTreeDiscretiser(bin_output=bin_output_)


@pytest.mark.parametrize("precision_", ["arbitrary", -1, 0.3])
def test_error_if_precision_not_permitted_value(precision_):
    msg = "precision must be None or a positive integer. " f"Got {precision_} instead."
    with pytest.raises(ValueError, match=msg):
        DecisionTreeDiscretiser(precision=precision_)


def test_precision_errors_if_none_when_bin_output_is_boundaries():
    msg = (
        "When `bin_output == 'boundaries', `precision` cannot be None. "
        "Change precision's value to a positive integer."
    )
    with pytest.raises(ValueError, match=msg):
        DecisionTreeDiscretiser(precision=None, bin_output="boundaries")

    dsc = DecisionTreeDiscretiser(precision=None, bin_output="bin_number")
    assert dsc.precision is None


@pytest.mark.parametrize("regression_", ["arbitrary", -1, 0.3])
def test_error_if_regression_is_not_bool(regression_):
    msg = "regression can only take True or False. " f"Got {regression_} instead."
    with pytest.raises(ValueError, match=msg):
        DecisionTreeDiscretiser(regression=regression_)


# fit
@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_if_y_not_passed(make_df):
    X = make_df(_normal_dist_data())
    encoder = DecisionTreeDiscretiser()
    with pytest.raises(TypeError):
        encoder.fit(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_when_regression_is_true_and_target_is_binary(make_df):
    data = _discretise_data()
    X = make_df({"var_A": data["var_A"], "var_B": data["var_B"]})
    y = data["target"]
    msg = (
        "Trying to fit a regression to a binary target is not "
        "allowed by this transformer. Check the target values "
        "or set regression to False."
    )
    transformer = DecisionTreeDiscretiser(regression=True)
    with pytest.raises(ValueError, match=msg):
        transformer.fit(X, y)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_classification_predictions(make_df):
    X = make_df(_normal_dist_data())

    transformer = DecisionTreeDiscretiser(
        cv=3,
        scoring="roc_auc",
        variables=None,
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=False,
        random_state=0,
    )
    np.random.seed(0)
    y = list(np.random.binomial(1, 0.7, 100))
    Xt = transformer.fit_transform(X, y)
    X_t = [1.0, 0.71, 0.93, 0.0]

    # init params
    assert transformer.cv == 3
    assert transformer.variables is None
    assert transformer.scoring == "roc_auc"
    assert transformer.regression is False
    # fit params
    assert transformer.variables_ == ["var"]
    assert transformer.n_features_in_ == 1
    # transform params
    unique_vals = _unique_sorted(Xt, "var")
    assert all(x for x in np.round(unique_vals, 2) if x not in X_t)
    assert np.round(transformer.scores_dict_["var"], 3) == np.round(
        0.717391304347826, 3
    )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "params",
    [
        (1, [1.0, 0.7, 0.9, 0.0]),
        (2, [1.0, 0.71, 0.93, 0.0]),
        (3, [1.0, 0.712, 0.933, 0.0]),
    ],
)
def test_classification_rounds_predictions(make_df, params):
    X = make_df(_normal_dist_data())

    transformer = DecisionTreeDiscretiser(
        precision=params[0],
        cv=3,
        scoring="roc_auc",
        variables=None,
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=False,
        random_state=0,
    )
    np.random.seed(0)
    y = list(np.random.binomial(1, 0.7, 100))
    Xt = transformer.fit_transform(X, y)
    bins = params[1]

    assert _unique_sorted(Xt, "var") == sorted(bins)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_classification_bin_number(make_df):
    X = make_df(_normal_dist_data())
    transformer = DecisionTreeDiscretiser(
        bin_output="bin_number",
        scoring="roc_auc",
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=False,
        random_state=0,
    )
    np.random.seed(0)
    y = list(np.random.binomial(1, 0.7, 100))
    Xt = transformer.fit_transform(X, y)
    bins = [0, 1, 2, 3, 4]
    limits = [
        -np.inf,
        -0.22668930888175964,
        -0.09422881528735161,
        0.10165948793292046,
        0.11590901389718056,
        np.inf,
    ]

    assert transformer.binner_dict_["var"] == limits
    assert np.round(transformer.scores_dict_["var"], 3) == np.round(
        0.717391304347826, 3
    )
    assert _unique_sorted(Xt, "var") == bins


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_classification_boundaries(make_df):
    X = make_df(_normal_dist_data())
    transformer = DecisionTreeDiscretiser(
        bin_output="boundaries",
        precision=3,
        scoring="roc_auc",
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=False,
        random_state=0,
    )
    np.random.seed(0)
    y = list(np.random.binomial(1, 0.7, 100))
    Xt = transformer.fit_transform(X, y)
    bins = sorted(
        [
            "(0.116, inf]",
            "(-0.0942, 0.102]",
            "(-0.227, -0.0942]",
            "(-inf, -0.227]",
            "(0.102, 0.116]",
        ]
    )
    limits = [
        -np.inf,
        -0.22668930888175964,
        -0.09422881528735161,
        0.10165948793292046,
        0.11590901389718056,
        np.inf,
    ]

    assert transformer.binner_dict_["var"] == limits
    assert np.round(transformer.scores_dict_["var"], 3) == np.round(
        0.717391304347826, 3
    )
    assert _unique_sorted(Xt, "var") == bins


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_regression(make_df):
    X = make_df(_normal_dist_data())

    transformer = DecisionTreeDiscretiser(
        cv=3,
        scoring="neg_mean_squared_error",
        variables=None,
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=True,
        random_state=0,
    )
    np.random.seed(0)
    y = list(np.random.normal(0, 0.1, 100))
    Xt = transformer.fit_transform(X, y)
    X_t = [
        0.19,
        0.04,
        0.11,
        0.23,
        -0.09,
        -0.02,
        0.01,
        0.15,
        0.07,
        -0.26,
        0.09,
        -0.07,
        -0.16,
        -0.2,
        -0.04,
        -0.12,
    ]

    # init params
    assert transformer.cv == 3
    assert transformer.variables is None
    assert transformer.scoring == "neg_mean_squared_error"
    assert transformer.regression is True
    # fit params
    assert transformer.variables_ == ["var"]
    assert transformer.n_features_in_ == 1
    assert np.round(transformer.scores_dict_["var"], 3) == np.round(
        -4.4373314584616444e-05, 3
    )
    # transform params
    unique_vals = _unique_sorted(Xt, "var")
    assert all(x for x in np.round(unique_vals, 2) if x not in X_t)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "params",
    [
        (1, [0.2, 0.0, 0.1, -0.1, -0.3, -0.2]),
        (
            2,
            [
                0.19,
                0.04,
                0.11,
                0.23,
                -0.09,
                -0.02,
                0.01,
                0.15,
                0.07,
                -0.26,
                0.09,
                -0.07,
                -0.16,
                -0.2,
                -0.04,
                -0.12,
            ],
        ),
    ],
)
def test_regression_rounds_predictions(make_df, params):
    X = make_df(_normal_dist_data())

    transformer = DecisionTreeDiscretiser(
        precision=params[0],
        cv=3,
        scoring="neg_mean_squared_error",
        variables=None,
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=True,
        random_state=0,
    )
    np.random.seed(0)
    y = list(np.random.normal(0, 0.1, 100))
    Xt = transformer.fit_transform(X, y)
    bins = params[1]

    assert _unique_sorted(Xt, "var") == sorted(bins)


# transform
@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_non_fitted_error(make_df):
    X = make_df(_normal_dist_data())
    with pytest.raises(NotFittedError):
        transformer = DecisionTreeDiscretiser()
        transformer.transform(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_when_regression_is_false_and_target_is_continuous(make_df):
    data = _discretise_data()
    X = make_df({"var_A": data["var_A"], "var_B": data["var_B"]})
    np.random.seed(42)
    mu, sigma = 0, 3
    y = list(np.random.normal(mu, sigma, len(data["var_A"])))
    transformer = DecisionTreeDiscretiser(regression=False)
    with pytest.raises(ValueError):
        transformer.fit(X, y)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_n_jobs_parallel_matches_sequential(make_df):
    # core correctness check for n_jobs: parallelizing tree training across
    # variables must produce identical trees, and therefore identical
    # predictions, to sequential training (n_jobs=None).
    data = _discretise_data()
    X = make_df({"var_A": data["var_A"], "var_B": data["var_B"]})
    np.random.seed(0)
    y = list(np.random.normal(0, 1, len(data["var_A"])))

    tr_seq = DecisionTreeDiscretiser(
        n_jobs=None, random_state=0, param_grid={"max_depth": [1, 2, 3]}
    )
    tr_seq.fit(X, y)
    tr_par = DecisionTreeDiscretiser(
        n_jobs=2, random_state=0, param_grid={"max_depth": [1, 2, 3]}
    )
    tr_par.fit(X, y)

    Xt_seq = tr_seq.transform(X)
    Xt_par = tr_par.transform(X)

    expected = nw.from_native(Xt_seq, eager_only=True).to_dict(as_series=False)
    result = nw.from_native(Xt_par, eager_only=True).to_dict(as_series=False)
    assert result == expected
