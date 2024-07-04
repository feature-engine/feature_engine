import numpy as np
import pandas as pd
import pytest

from feature_engine._prediction.target_mean_classifier import TargetMeanClassifier


def test_attr_classes(df_classification):
    X, y = df_classification
    tr = TargetMeanClassifier(variables="cat_var_A")
    tr.fit(X, y)
    assert tr.classes_.tolist() == [0, 1]

    y = pd.Series([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    tr.fit(X, y)
    assert tr.classes_.tolist() == [1, 2]


def test_categorical_variables(df_classification):

    X, y = df_classification

    tr = TargetMeanClassifier(variables="cat_var_A")
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    assert np.array_equal(pred, exp_pred)
    assert np.array_equal(prob, exp_prob)

    tr = TargetMeanClassifier(variables="cat_var_B")
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    assert np.array_equal(pred, exp_pred)
    assert np.array_equal(prob, exp_prob)

    tr = TargetMeanClassifier(variables=["cat_var_A", "cat_var_B"])
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.75, 0.25],
            [0.75, 0.25],
            [0.75, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.25, 0.75],
            [0.25, 0.75],
            [0.25, 0.75],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    assert np.array_equal(pred, exp_pred)
    assert np.array_equal(prob, exp_prob)


def test_numerical_variables(df_classification):

    X, y = df_classification

    tr = TargetMeanClassifier(variables="num_var_A", bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    assert np.array_equal(pred, exp_pred)
    assert np.array_equal(prob, exp_prob)

    tr = TargetMeanClassifier(variables="num_var_B", bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array(
        [
            [0.8, 0.2],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.2, 0.8],
            [0.2, 0.8],
            [0.2, 0.8],
            [0.2, 0.8],
            [0.2, 0.8],
            [0.2, 0.8],
            [0.2, 0.8],
        ]
    )
    assert np.array_equal(pred, exp_pred)
    assert np.allclose(prob, exp_prob)

    tr = TargetMeanClassifier(variables=["num_var_A", "num_var_B"], bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array(
        [
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.6, 0.4],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.4, 0.6],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
        ]
    )

    assert np.array_equal(pred, exp_pred)
    assert np.allclose(prob, exp_prob)


def test_classifier_all_variables(df_classification):

    X, y = df_classification

    tr = TargetMeanClassifier(bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)
    prob_log = tr.predict_log_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array(
        [
            [0.95, 0.05],
            [0.95, 0.05],
            [0.95, 0.05],
            [0.95, 0.05],
            [0.95, 0.05],
            [0.95, 0.05],
            [0.825, 0.175],
            [0.825, 0.175],
            [0.675, 0.325],
            [0.675, 0.325],
            [0.325, 0.675],
            [0.325, 0.675],
            [0.175, 0.825],
            [0.175, 0.825],
            [0.05, 0.95],
            [0.05, 0.95],
            [0.05, 0.95],
            [0.05, 0.95],
            [0.05, 0.95],
            [0.05, 0.95],
        ]
    )

    exp_prob_log = np.array(
        [
            [-0.05129329, -2.99573227],
            [-0.05129329, -2.99573227],
            [-0.05129329, -2.99573227],
            [-0.05129329, -2.99573227],
            [-0.05129329, -2.99573227],
            [-0.05129329, -2.99573227],
            [-0.19237189, -1.74296931],
            [-0.19237189, -1.74296931],
            [-0.39304259, -1.1239301],
            [-0.39304259, -1.1239301],
            [-1.1239301, -0.39304259],
            [-1.1239301, -0.39304259],
            [-1.74296931, -0.19237189],
            [-1.74296931, -0.19237189],
            [-2.99573227, -0.05129329],
            [-2.99573227, -0.05129329],
            [-2.99573227, -0.05129329],
            [-2.99573227, -0.05129329],
            [-2.99573227, -0.05129329],
            [-2.99573227, -0.05129329],
        ]
    )

    assert np.array_equal(pred, exp_pred)
    assert np.allclose(prob, exp_prob)
    assert np.allclose(prob_log, exp_prob_log)


def test_error_when_target_not_binary(df_classification):
    X, y = df_classification
    tr = TargetMeanClassifier(variables="cat_var_A")
    y = pd.Series([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    with pytest.raises(NotImplementedError):
        tr.fit(X, y)
