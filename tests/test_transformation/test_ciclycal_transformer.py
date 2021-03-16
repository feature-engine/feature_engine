import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import CyclicalTransformer


def test_general_transformation(df_ciclycal_trans):
    """Only one variable"""
    ciclycal = CyclicalTransformer(variables=['day'])
    X = ciclycal.fit_transform(df_ciclycal_trans)

    transf_df = df_ciclycal_trans.copy()
    # transf_df['day_sin'] = [
    #     -7.818315e-01,
    #     -2.449294e-16,
    #     -9.749279e-01,
    #     4.338837e-01,
    #     7.818315e-01,
    #     9.749279e-01,
    #     -4.338837e-01
    # ]

    transf_df['day_sin'] = [
        -0.78183,
        -0.0,
        -0.97493,
        0.43388,
        0.78183,
        0.97493,
        -0.43388,
    ]
    transf_df['day_cos'] = [
        0.623490,
        1.000000,
        -0.222521,
        -0.900969,
        0.623490,
        -0.222521,
        -0.900969
    ]
    transf_df = transf_df.drop(columns='day')

    # test init params
    assert ciclycal.variables == ['day']
    assert ciclycal.max_values == {
        'day': 7,
    }

    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)





