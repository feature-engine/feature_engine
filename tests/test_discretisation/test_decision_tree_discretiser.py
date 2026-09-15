import re

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import DecisionTreeDiscretiser
from tests.backend_helpers import make_series, frame_to_dict

_rng = np.random.RandomState(42)
DATA_TWO_VARS = {
    "var_A": _rng.normal(0, 3, 20).tolist(),
    "var_B": _rng.normal(3, 5, 20).tolist(),
}
TARGET_TWO_VARS = [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]


def _binary_target():
    np.random.seed(0)
    return np.random.binomial(1, 0.7, 100).tolist()


def _continuous_target():
    np.random.seed(0)
    return np.random.normal(0, 0.1, 100).tolist()


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
    with pytest.raises(ValueError, match=re.escape(msg)):
        DecisionTreeDiscretiser(bin_output=bin_output_)


@pytest.mark.parametrize("precision_", ["arbitrary", -1, 0.3])
def test_error_if_precision_not_permitted_value(precision_):
    msg = "precision must be None or a positive integer. " f"Got {precision_} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DecisionTreeDiscretiser(precision=precision_)


def test_precision_errors_if_none_when_bin_output_is_boundaries():
    msg = (
        "When `bin_output == 'boundaries', `precision` cannot be None. "
        "Change precision's value to a positive integer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DecisionTreeDiscretiser(precision=None, bin_output="boundaries")

    dsc = DecisionTreeDiscretiser(precision=None, bin_output="bin_number")
    assert dsc.precision is None


@pytest.mark.parametrize("regression_", ["arbitrary", -1, 0.3])
def test_error_if_regression_is_not_bool(regression_):
    msg = "regression can only take True or False. " f"Got {regression_} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DecisionTreeDiscretiser(regression=regression_)


# fit
def test_error_if_y_not_passed(make_df, data_normal_dist):
    encoder = DecisionTreeDiscretiser()
    with pytest.raises(TypeError):
        encoder.fit(make_df(data_normal_dist))


def test_error_when_regression_is_true_and_target_is_binary(make_df):
    X = make_df(DATA_TWO_VARS)
    y = make_series(make_df, TARGET_TWO_VARS)
    msg = (
        "Trying to fit a regression to a binary target is not "
        "allowed by this transformer. Check the target values "
        "or set regression to False."
    )
    transformer = DecisionTreeDiscretiser(regression=True)
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.fit(X, y)


def test_classification_predictions(make_df, data_normal_dist):
    X = make_df(data_normal_dist)
    y = make_series(make_df, _binary_target())

    transformer = DecisionTreeDiscretiser(
        cv=3,
        scoring="roc_auc",
        variables=None,
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=False,
        random_state=0,
    )
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
    assert isinstance(Xt, make_df)
    unique_vals = sorted(set(frame_to_dict(Xt)["var"]))
    assert all(x for x in np.round(unique_vals, 2) if x not in X_t)
    assert np.round(transformer.scores_dict_["var"], 3) == np.round(
        0.717391304347826, 3
    )


@pytest.mark.parametrize("to_target", [list, np.array])
def test_target_as_list_or_array(make_df, data_normal_dist, to_target):
    # a list or numpy array target must give the same result as a Series
    X = make_df(data_normal_dist)
    params = dict(
        bin_output="bin_number",
        scoring="roc_auc",
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=False,
        random_state=0,
    )

    from_series = DecisionTreeDiscretiser(**params)
    from_series.fit(X, make_series(make_df, _binary_target()))
    transformer = DecisionTreeDiscretiser(**params)
    transformer.fit(X, to_target(_binary_target()))
    Xt = transformer.transform(X)

    assert transformer.binner_dict_ == from_series.binner_dict_
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == frame_to_dict(from_series.transform(X))


@pytest.mark.parametrize(
    "params",
    [
        (1, [1.0, 0.7, 0.9, 0.0]),
        (2, [1.0, 0.71, 0.93, 0.0]),
        (3, [1.0, 0.712, 0.933, 0.0]),
    ],
)
def test_classification_rounds_predictions(make_df, data_normal_dist, params):
    X = make_df(data_normal_dist)
    y = make_series(make_df, _binary_target())

    transformer = DecisionTreeDiscretiser(
        precision=params[0],
        cv=3,
        scoring="roc_auc",
        variables=None,
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=False,
        random_state=0,
    )
    Xt = transformer.fit_transform(X, y)

    assert isinstance(Xt, make_df)
    assert sorted(set(frame_to_dict(Xt)["var"])) == sorted(params[1])


def test_classification_bin_number(make_df, data_normal_dist):
    X = make_df(data_normal_dist)
    y = make_series(make_df, _binary_target())
    transformer = DecisionTreeDiscretiser(
        bin_output="bin_number",
        scoring="roc_auc",
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=False,
        random_state=0,
    )
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
    assert isinstance(Xt, make_df)
    assert sorted(set(frame_to_dict(Xt)["var"])) == bins


def test_classification_boundaries(make_df, data_normal_dist):
    X = make_df(data_normal_dist)
    y = make_series(make_df, _binary_target())
    transformer = DecisionTreeDiscretiser(
        bin_output="boundaries",
        precision=3,
        scoring="roc_auc",
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=False,
        random_state=0,
    )
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
    assert isinstance(Xt, make_df)
    assert sorted(set(frame_to_dict(Xt)["var"])) == bins


def test_regression(make_df, data_normal_dist):
    X = make_df(data_normal_dist)
    y = make_series(make_df, _continuous_target())

    transformer = DecisionTreeDiscretiser(
        cv=3,
        scoring="neg_mean_squared_error",
        variables=None,
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=True,
        random_state=0,
    )
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
    assert isinstance(Xt, make_df)
    unique_vals = sorted(set(frame_to_dict(Xt)["var"]))
    assert all(x for x in np.round(unique_vals, 2) if x not in X_t)


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
def test_regression_rounds_predictions(make_df, data_normal_dist, params):
    X = make_df(data_normal_dist)
    y = make_series(make_df, _continuous_target())

    transformer = DecisionTreeDiscretiser(
        precision=params[0],
        cv=3,
        scoring="neg_mean_squared_error",
        variables=None,
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=True,
        random_state=0,
    )
    Xt = transformer.fit_transform(X, y)

    assert isinstance(Xt, make_df)
    assert sorted(set(frame_to_dict(Xt)["var"])) == sorted(params[1])


# transform
def test_non_fitted_error(make_df, data_normal_dist):
    transformer = DecisionTreeDiscretiser()
    with pytest.raises(NotFittedError):
        transformer.transform(make_df(data_normal_dist))


def test_error_when_regression_is_false_and_target_is_continuous(make_df):
    X = make_df(DATA_TWO_VARS)
    np.random.seed(42)
    y = make_series(make_df, np.random.normal(0, 3, 20).tolist())
    transformer = DecisionTreeDiscretiser(regression=False)
    with pytest.raises(ValueError):
        transformer.fit(X, y)


def test_n_jobs_parallel_matches_sequential(make_df):
    # core correctness check for n_jobs: parallelizing tree training across
    # variables must produce identical trees, and therefore identical
    # predictions, to sequential training (n_jobs=None).
    X = make_df(DATA_TWO_VARS)
    np.random.seed(0)
    y = make_series(make_df, np.random.normal(0, 1, 20).tolist())

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

    assert isinstance(Xt_par, make_df)
    assert frame_to_dict(Xt_par) == frame_to_dict(Xt_seq)
