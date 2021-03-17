import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import CyclicalTransformer


def test_general_transformation(df_ciclycal_trans):
    # test case 1: just one variable.
    ciclycal = CyclicalTransformer(variables=['day'])
    X = ciclycal.fit_transform(df_ciclycal_trans)

    transf_df = df_ciclycal_trans.copy()

    # expected output
    transf_df['day_sin'] = [
        -0.78183,
        0.0,
        -0.97493,
        0.43388,
        0.78183,
        0.97493,
        -0.43388,
    ]
    transf_df['day_cos'] = [
        0.623490,
        1.0,
        -0.222521,
        -0.900969,
        0.623490,
        -0.222521,
        -0.900969
    ]
    transf_df = transf_df.drop(columns='day')

    # test init params
    assert ciclycal.variables == ['day']

    # test fit attr
    assert ciclycal.input_shape_ == (7, 2)
    assert ciclycal.max_values_ == {
        'day': 7,
    }

    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)


def test_automatically_find_variables(df_ciclycal_trans):
    # test case 2: automatically select variables
    ciclycal = CyclicalTransformer(variables=None)
    X = ciclycal.fit_transform(df_ciclycal_trans)
    transf_df = df_ciclycal_trans.copy()

    # expected output
    transf_df['day_sin'] = [
        -0.78183,
        0.0,
        -0.97493,
        0.43388,
        0.78183,
        0.97493,
        -0.43388,
    ]
    transf_df['day_cos'] = [
        0.62349,
        1.0,
        -0.222521,
        -0.900969,
        0.62349,
        -0.222521,
        -0.900969
    ]
    transf_df['months_sin'] = [
        1.0,
        -0.5,
        -1.0,
        0.0,
        0.86603,
        0.0,
        0.0,
    ]
    transf_df['months_cos'] = [
        0.0,
        -0.86603,
        -0.0,
        1.0,
        -0.5,
        -1.0,
        1.0,
    ]
    transf_df = transf_df.drop(columns=['day', 'months'])

    # test init params
    assert ciclycal.variables == ['day', 'months']

    # test fit attr
    assert ciclycal.input_shape_ == (7, 2)
    assert ciclycal.max_values_ == {
        'day': 7,
        'months': 12,
    }

    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)





