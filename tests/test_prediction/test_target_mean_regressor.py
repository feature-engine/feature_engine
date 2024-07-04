import numpy as np
import pytest

from feature_engine._prediction.target_mean_regressor import TargetMeanRegressor


def test_regressor_categorical_variables(df_regression):

    X, y = df_regression

    tr = TargetMeanRegressor(variables="cat_var_A")
    tr.fit(X, y)
    pred = tr.predict(X)

    exp_pred = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
        ]
    )

    assert np.array_equal(pred, exp_pred)

    tr = TargetMeanRegressor(variables="cat_var_B")
    tr.fit(X, y)

    pred = tr.predict(X)

    exp_pred = np.array(
        [
            0.16666667,
            0.16666667,
            0.16666667,
            0.16666667,
            0.16666667,
            0.16666667,
            1.5,
            1.5,
            1.5,
            1.5,
            1.5,
            1.5,
            1.5,
            1.5,
            2.83333333,
            2.83333333,
            2.83333333,
            2.83333333,
            2.83333333,
            2.83333333,
        ]
    )

    assert np.allclose(pred, exp_pred)

    tr = TargetMeanRegressor(variables=["cat_var_A", "cat_var_B"])
    tr.fit(X, y)

    pred = tr.predict(X)

    exp_pred = np.array(
        [
            0.08333333,
            0.08333333,
            0.08333333,
            0.08333333,
            0.08333333,
            0.58333333,
            1.25,
            1.25,
            1.25,
            1.25,
            1.75,
            1.75,
            1.75,
            1.75,
            2.41666667,
            2.91666667,
            2.91666667,
            2.91666667,
            2.91666667,
            2.91666667,
        ]
    )

    assert np.allclose(pred, exp_pred)


def test_classifier_numerical_variables(df_regression):

    X, y = df_regression

    tr = TargetMeanRegressor(variables="num_var_A", bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)

    exp_pred = np.array(
        [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
        ]
    )

    assert np.array_equal(pred, exp_pred)

    tr = TargetMeanRegressor(variables="num_var_B", bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)

    exp_pred = np.array(
        [
            0.7,
            0.7,
            0.7,
            0.7,
            0.7,
            0.7,
            0.7,
            0.7,
            2.3,
            2.3,
            0.7,
            0.7,
            2.3,
            2.3,
            2.3,
            2.3,
            2.3,
            2.3,
            2.3,
            2.3,
        ]
    )

    np.array_equal(pred, exp_pred)

    tr = TargetMeanRegressor(variables=["num_var_A", "num_var_B"], bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)

    exp_pred = np.array(
        [
            0.6,
            0.6,
            0.6,
            0.6,
            0.6,
            0.6,
            0.6,
            0.6,
            1.4,
            1.4,
            1.6,
            1.6,
            2.4,
            2.4,
            2.4,
            2.4,
            2.4,
            2.4,
            2.4,
            2.4,
        ]
    )

    assert np.array_equal(pred, exp_pred)


def test_classifier_all_variables(df_regression):

    X, y = df_regression

    tr = TargetMeanRegressor(bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)

    exp_pred = np.array(
        [
            0.34166667,
            0.34166667,
            0.34166667,
            0.34166667,
            0.34166667,
            0.59166667,
            0.925,
            0.925,
            1.325,
            1.325,
            1.675,
            1.675,
            2.075,
            2.075,
            2.40833333,
            2.65833333,
            2.65833333,
            2.65833333,
            2.65833333,
            2.65833333,
        ]
    )

    assert np.allclose(pred, exp_pred)


def test_error_when_y_is_binary(df_regression):
    X, y = df_regression
    y = [1.0, 2.0]
    tr = TargetMeanRegressor(bins=2)
    with pytest.raises(ValueError):
        tr.fit(X, y)
