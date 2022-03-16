import numpy as np
from feature_engine._prediction.target_mean_regressor import TargetMeanRegressor


def test_regressor_categorical_variables(df_regression):

    X, y = df_regression

    tr = TargetMeanRegressor(variables="cat_var_A")
    tr.fit(X, y)
    pred = tr.predict(X)

    exp_pred = np.array(
        [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3.]
    )

    assert np.array_equal(pred, exp_pred)

    tr = TargetMeanRegressor(variables="cat_var_B")
    tr.fit(X, y)

    pred = tr.predict(X)

    exp_pred = np.array(
        [0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667,
         0.16666667, 1.5, 1.5, 1.5, 1.5,
         1.5, 1.5, 1.5, 1.5, 2.83333333,
         2.83333333, 2.83333333, 2.83333333, 2.83333333, 2.83333333]
    )

    assert np.array_equal(pred, exp_pred)

    tr = TargetMeanRegressor(variables=["cat_var_A","cat_var_B"])
    tr.fit(X, y)

    pred = tr.predict(X)

    exp_pred = np.array(
        [0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.08333333,
         0.58333333, 1.25, 1.25, 1.25, 1.25,
         1.75, 1.75, 1.75, 1.75, 2.41666667,
         2.91666667, 2.91666667, 2.91666667, 2.91666667, 2.91666667]
    )

    assert np.array_equal(pred, exp_pred)


def test_classifier_numerical_variables(df_regression):

    X, y = df_regression

    tr = TargetMeanRegressor(variables="num_var_A", bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)

    exp_pred = np.array(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.5, 2.5, 2.5,
         2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
    )

    assert np.array_equal(pred, exp_pred)

    tr = TargetMeanRegressor(variables="cat_var_B", bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    np.array_equal(pred, exp_pred)

    tr = TargetMeanRegressor(variables=["num_var_A","num_var_B"], bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    assert np.array_equal(pred, exp_pred)


def test_classifier_all_variables(df_regression):

    X, y = df_regression

    tr = TargetMeanRegressor(bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    assert np.array_equal(pred, exp_pred)
