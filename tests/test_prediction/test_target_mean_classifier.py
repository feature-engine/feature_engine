import numpy as np
import pandas as pd
import pytest
from feature_engine._prediction.target_mean_classifier import TargetMeanClassifier

#TODO assert attribute classes
#assert tr.classes_.tolist() == [0, 1]


def test_categorical_variables(df_classification):

    X, y = df_classification

    tr = TargetMeanClassifier(variables="cat_var_A")
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array([[1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.]])

    assert np.array_equal(pred, exp_pred)
    assert np.array_equal(prob, exp_prob)

    tr = TargetMeanClassifier(variables="cat_var_B")
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array([[1. , 0. ],
       [1. , 0. ],
       [1. , 0. ],
       [1. , 0. ],
       [1. , 0. ],
       [1. , 0. ],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0. , 1. ],
       [0. , 1. ],
       [0. , 1. ],
       [0. , 1. ],
       [0. , 1. ],
       [0. , 1. ]])

    assert np.array_equal(pred, exp_pred)
    assert np.array_equal(prob, exp_prob)

    tr = TargetMeanClassifier(variables=["cat_var_A","cat_var_B"])
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array([[1.  , 0.  ],
       [1.  , 0.  ],
       [1.  , 0.  ],
       [1.  , 0.  ],
       [1.  , 0.  ],
       [1.  , 0.  ],
       [0.75, 0.25],
       [0.75, 0.25],
       [0.75, 0.25],
       [0.75, 0.25],
       [0.25, 0.75],
       [0.25, 0.75],
       [0.25, 0.75],
       [0.25, 0.75],
       [0.  , 1.  ],
       [0.  , 1.  ],
       [0.  , 1.  ],
       [0.  , 1.  ],
       [0.  , 1.  ],
       [0.  , 1.  ]])

    assert np.array_equal(pred, exp_pred)
    assert np.array_equal(prob, exp_prob)


def test_numerical_variables(df_classification):

    X, y = df_classification

    tr = TargetMeanClassifier(variables="num_var_A", bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array([[1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [1., 0.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.],
                    [0., 1.]])

    assert np.array_equal(pred, exp_pred)
    assert np.array_equal(prob, exp_prob)

    tr = TargetMeanClassifier(variables="cat_var_B", bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    exp_prob = np.array([[0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5],
       [0.5, 0.5]])

    assert np.array_equal(pred, exp_pred)
    assert np.array_equal(prob, exp_prob)

    tr = TargetMeanClassifier(variables=["num_var_A","num_var_B"], bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array([[0.75, 0.25],
       [0.75, 0.25],
       [0.75, 0.25],
       [0.75, 0.25],
       [0.75, 0.25],
       [0.75, 0.25],
       [0.75, 0.25],
       [0.75, 0.25],
       [0.75, 0.25],
       [0.75, 0.25],
       [0.25, 0.75],
       [0.25, 0.75],
       [0.25, 0.75],
       [0.25, 0.75],
       [0.25, 0.75],
       [0.25, 0.75],
       [0.25, 0.75],
       [0.25, 0.75],
       [0.25, 0.75],
       [0.25, 0.75]])

    assert np.array_equal(pred, exp_pred)
    assert np.array_equal(prob, exp_prob)


def test_classifier_all_variables(df_classification):

    X, y = df_classification

    tr = TargetMeanClassifier(bins=2)
    tr.fit(X, y)

    pred = tr.predict(X)
    prob = tr.predict_proba(X)
    prob_log = tr.predict_proba(X)

    exp_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    exp_prob = np.array([[0.875, 0.125],
       [0.875, 0.125],
       [0.875, 0.125],
       [0.875, 0.125],
       [0.875, 0.125],
       [0.875, 0.125],
       [0.75 , 0.25 ],
       [0.75 , 0.25 ],
       [0.75 , 0.25 ],
       [0.75 , 0.25 ],
       [0.25 , 0.75 ],
       [0.25 , 0.75 ],
       [0.25 , 0.75 ],
       [0.25 , 0.75 ],
       [0.125, 0.875],
       [0.125, 0.875],
       [0.125, 0.875],
       [0.125, 0.875],
       [0.125, 0.875],
       [0.125, 0.875]])

    exp_prob_log = np.array([[-0.13353139, -2.07944154],
       [-0.13353139, -2.07944154],
       [-0.13353139, -2.07944154],
       [-0.13353139, -2.07944154],
       [-0.13353139, -2.07944154],
       [-0.13353139, -2.07944154],
       [-0.28768207, -1.38629436],
       [-0.28768207, -1.38629436],
       [-0.28768207, -1.38629436],
       [-0.28768207, -1.38629436],
       [-1.38629436, -0.28768207],
       [-1.38629436, -0.28768207],
       [-1.38629436, -0.28768207],
       [-1.38629436, -0.28768207],
       [-2.07944154, -0.13353139],
       [-2.07944154, -0.13353139],
       [-2.07944154, -0.13353139],
       [-2.07944154, -0.13353139],
       [-2.07944154, -0.13353139],
       [-2.07944154, -0.13353139]])

    assert np.array_equal(pred, exp_pred)
    assert np.array_equal(prob, exp_prob)
    assert np.array_equal(prob_log, exp_prob_log)
