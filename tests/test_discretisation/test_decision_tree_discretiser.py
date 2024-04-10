import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import DecisionTreeDiscretiser, EqualWidthDiscretiser


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
    with pytest.raises(ValueError) as record:
        DecisionTreeDiscretiser(bin_output=bin_output_)
    assert str(record.value) == msg


@pytest.mark.parametrize("precision_", ["arbitrary", -1, 0.3])
def test_error_if_precision_not_permitted_value(precision_):
    msg = "precision must be None or a positive integer. " f"Got {precision_} instead."
    with pytest.raises(ValueError) as record:
        DecisionTreeDiscretiser(precision=precision_)
    assert str(record.value) == msg


def test_precision_errors_if_none_when_bin_output_is_boundaries():
    msg = (
        "When `bin_output == 'boundaries', `precision` cannot be None. "
        "Change precision's value to a positive integer."
    )
    with pytest.raises(ValueError) as record:
        DecisionTreeDiscretiser(precision=None, bin_output="boundaries")
    assert str(record.value) == msg

    dsc = DecisionTreeDiscretiser(precision=None, bin_output="bin_number")
    assert dsc.precision is None


@pytest.mark.parametrize("regression_", ["arbitrary", -1, 0.3])
def test_error_if_regression_is_not_bool(regression_):
    msg = "regression can only take True or False. " f"Got {regression_} instead."
    with pytest.raises(ValueError) as record:
        DecisionTreeDiscretiser(regression=regression_)
    assert str(record.value) == msg


# fit
def test_error_if_y_not_passed(df_normal_dist):
    encoder = DecisionTreeDiscretiser()
    with pytest.raises(TypeError):
        encoder.fit(df_normal_dist)


def test_error_when_regression_is_true_and_target_is_binary(df_discretise):
    msg = (
        "Trying to fit a regression to a binary target is not "
        "allowed by this transformer. Check the target values "
        "or set regression to False."
    )
    transformer = DecisionTreeDiscretiser(regression=True)
    with pytest.raises(ValueError) as record:
        transformer.fit(df_discretise[["var_A", "var_B"]], df_discretise["target"])
    assert str(record.value) == msg


def test_classification_predictions(df_normal_dist):

    transformer = DecisionTreeDiscretiser(
        cv=3,
        scoring="roc_auc",
        variables=None,
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=False,
        random_state=0,
    )
    np.random.seed(0)
    y = pd.Series(np.random.binomial(1, 0.7, 100))
    X = transformer.fit_transform(df_normal_dist, y)
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
    assert all(x for x in np.round(X["var"].unique(), 2) if x not in X_t)
    assert np.round(transformer.scores_dict_["var"], 3) == np.round(
        0.717391304347826, 3
    )


@pytest.mark.parametrize(
    "params",
    [
        (1, [1.0, 0.7, 0.9, 0.0]),
        (2, [1.0, 0.71, 0.93, 0.0]),
        (3, [1.0, 0.712, 0.933, 0.0]),
    ],
)
def test_classification_rounds_predictions(df_normal_dist, params):

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
    y = pd.Series(np.random.binomial(1, 0.7, 100))
    X = transformer.fit_transform(df_normal_dist, y)
    bins = params[1]

    assert list(X["var"].unique()) == bins


def test_classification_bin_number(df_normal_dist):
    transformer = DecisionTreeDiscretiser(
        bin_output="bin_number",
        scoring="roc_auc",
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=False,
        random_state=0,
    )
    np.random.seed(0)
    y = pd.Series(np.random.binomial(1, 0.7, 100))
    X = transformer.fit_transform(df_normal_dist, y)
    bins = [4, 2, 1, 0, 3]
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
    assert list(X["var"].unique()) == bins


def test_classification_boundaries(df_normal_dist):
    transformer = DecisionTreeDiscretiser(
        bin_output="boundaries",
        precision=3,
        scoring="roc_auc",
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=False,
        random_state=0,
    )
    np.random.seed(0)
    y = pd.Series(np.random.binomial(1, 0.7, 100))
    X = transformer.fit_transform(df_normal_dist, y)
    bins = [
        "(0.116, inf]",
        "(-0.0942, 0.102]",
        "(-0.227, -0.0942]",
        "(-inf, -0.227]",
        "(0.102, 0.116]",
    ]
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
    assert list(X["var"].unique()) == bins


def test_regression(df_normal_dist):

    transformer = DecisionTreeDiscretiser(
        cv=3,
        scoring="neg_mean_squared_error",
        variables=None,
        param_grid={"max_depth": [1, 2, 3, 4]},
        regression=True,
        random_state=0,
    )
    np.random.seed(0)
    y = pd.Series(pd.Series(np.random.normal(0, 0.1, 100)))
    X = transformer.fit_transform(df_normal_dist, y)
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
    assert all(x for x in np.round(X["var"].unique(), 2) if x not in X_t)


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
def test_regression_rounds_predictions(df_normal_dist, params):

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
    y = pd.Series(pd.Series(np.random.normal(0, 0.1, 100)))
    X = transformer.fit_transform(df_normal_dist, y)
    bins = params[1]

    assert list(X["var"].unique()) == bins


# transform
def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = EqualWidthDiscretiser()
        transformer.transform(df_vartypes)


@pytest.fixture(scope="module")
def df_discretise():
    np.random.seed(42)
    mu1, sigma1 = 0, 3
    s1 = np.random.normal(mu1, sigma1, 20)
    mu2, sigma2 = 3, 5
    s2 = np.random.normal(mu2, sigma2, 20)
    data = {
        "var_A": s1,
        "var_B": s2,
        "target": [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    }

    df = pd.DataFrame(data)

    return df


def test_error_when_regression_is_false_and_target_is_continuous(df_discretise):
    np.random.seed(42)
    mu, sigma = 0, 3
    y = np.random.normal(mu, sigma, len(df_discretise))
    transformer = DecisionTreeDiscretiser(regression=False)
    with pytest.raises(ValueError):
        transformer.fit(df_discretise[["var_A", "var_B"]], y)
