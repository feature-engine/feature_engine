import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.discretisation import DecisionTreeDiscretiser, EqualWidthDiscretiser


def test_classification(df_normal_dist):

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


def test_error_when_regression_is_not_bool():
    with pytest.raises(ValueError):
        DecisionTreeDiscretiser(regression="other")


def test_error_if_y_not_passed(df_normal_dist):
    # test case 3: raises error if target is not passed
    with pytest.raises(TypeError):
        encoder = DecisionTreeDiscretiser()
        encoder.fit(df_normal_dist)


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


def test_error_when_regression_is_true_and_target_is_binary(df_discretise):
    with pytest.raises(ValueError):
        transformer = DecisionTreeDiscretiser(regression=True)
        transformer.fit(df_discretise[["var_A", "var_B"]], df_discretise["target"])


def test_error_when_regression_is_false_and_target_is_continuous(df_discretise):
    np.random.seed(42)
    mu, sigma = 0, 3
    y = np.random.normal(mu, sigma, len(df_discretise))
    with pytest.raises(ValueError):
        transformer = DecisionTreeDiscretiser(regression=False)
        transformer.fit(df_discretise[["var_A", "var_B"]], y)
