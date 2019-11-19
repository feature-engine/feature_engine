# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import pandas as pd

from feature_engine.categorical_encoders import RareLabelCategoricalEncoder
from feature_engine.categorical_encoders import CountFrequencyCategoricalEncoder
from feature_engine.categorical_encoders import OrdinalCategoricalEncoder
from feature_engine.categorical_encoders import MeanCategoricalEncoder
from feature_engine.categorical_encoders import WoERatioCategoricalEncoder
from feature_engine.categorical_encoders import OneHotCategoricalEncoder


def test_CountFrequencyCategoricalEncoder():
    # Test dataframe
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, ]}
    df = pd.DataFrame(df)

    # TEST COUNTS:
    # transformed dataframe
    transf_df = {'category': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4],
                 'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    transf_df = pd.DataFrame(transf_df)

    encoder = CountFrequencyCategoricalEncoder(encoding_method='count', variables=['category'])
    encoder.fit(df)
    X = encoder.transform(df)

    pd.testing.assert_frame_equal(X, transf_df)
    assert encoder.variables == ['category']
    assert encoder.encoder_dict_ == {'category': {'A': 10, 'B': 6, 'C': 4}}
    assert encoder.input_shape_ == (20, 2)

    # TEST FREQUENCY:
    # transformed dataframe 
    transf_df = {'category': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3,
                              0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2],
                 'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    transf_df = pd.DataFrame(transf_df)

    encoder = CountFrequencyCategoricalEncoder(encoding_method='frequency', variables=['category'])
    encoder.fit(df)
    X = encoder.transform(df)

    pd.testing.assert_frame_equal(X, transf_df)
    assert encoder.variables == ['category']
    assert encoder.encoder_dict_ == {'category': {'A': 0.5, 'B': 0.3, 'C': 0.2}}
    assert encoder.input_shape_ == (20, 2)


def test_OrdinalCategoricalEncoder():
    # Test dataframe
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, ]}
    df = pd.DataFrame(df)

    # # TODO: TEST UNORDERED ORDINAL ENCODING
    # transf_df = {TO DEFINE}
    # transf_df = pd.DataFrame(transf_df)

    # encoder = OrdinalCategoricalEncoder(encoding_method='arbitrary', variables = ['category'])
    # encoder.fit(df['category'].to_frame(), df['target'] )
    # X = encoder.transform(df['category'].to_frame())

    # pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())
    # assert encoder.variables == ['category']
    # assert encoder.encoder_dict_ == {'category': {'A':0, 'B':1, 'C':2}}
    # assert encoder.input_shape_ == (20,1)

    # TEST ORDERED ORDINAL ENCODING
    # transformed dataframe
    transf_df = {'category': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                 'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    transf_df = pd.DataFrame(transf_df)

    encoder = OrdinalCategoricalEncoder(encoding_method='ordered', variables=['category'])
    encoder.fit(df['category'].to_frame(), df['target'])
    X = encoder.transform(df['category'].to_frame())

    pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())
    assert encoder.variables == ['category']
    assert encoder.encoder_dict_ == {'category': {'A': 0, 'B': 1, 'C': 2}}
    assert encoder.input_shape_ == (20, 1)


def test_MeanCategoricalEncoder():
    # test dataframe
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, ]}
    df = pd.DataFrame(df)

    # transformed dataframe
    transf_df = {'category': [0.200000, 0.200000, 0.200000, 0.200000, 0.200000,
                              0.200000, 0.200000, 0.200000, 0.200000, 0.200000,
                              0.333333, 0.333333, 0.333333, 0.333333, 0.333333,
                              0.333333, 0.500000, 0.500000, 0.500000, 0.500000],
                 'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    transf_df = pd.DataFrame(transf_df)

    encoder = MeanCategoricalEncoder(variables=['category'])
    encoder.fit(df['category'].to_frame(), df['target'])
    X = encoder.transform(df['category'].to_frame())

    pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())
    assert encoder.variables == ['category']
    assert encoder.encoder_dict_ == {'category': {'A': 0.20000000000000001,
                                                  'B': 0.33333333333333331,
                                                  'C': 0.5}}
    assert encoder.input_shape_ == (20, 1)


def test_WoERatioCategoricalEncoder():
    # test dataframe
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, ]}
    df = pd.DataFrame(df)

    # TEST RATIO ENCODING:
    # transformed dataframe
    transf_df = {'category': [0.25, 0.25, 0.25, 0.25, 0.25,
                              0.25, 0.25, 0.25, 0.25, 0.25,
                              0.50, 0.50, 0.50, 0.50, 0.50,
                              0.50, 1.00, 1.00, 1.00, 1.00],
                 'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    transf_df = pd.DataFrame(transf_df)

    encoder = WoERatioCategoricalEncoder(encoding_method='ratio', variables=['category'])
    encoder.fit(df['category'].to_frame(), df['target'])
    X = encoder.transform(df['category'].to_frame())

    pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())
    assert encoder.variables == ['category']
    assert encoder.encoder_dict_ == {'category': {'A': 0.25,
                                                  'B': 0.49999999999999994,
                                                  'C': 1.00}}
    assert encoder.input_shape_ == (20, 1)

    # # test when one of the probabilities is zero
    # df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
    #       'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, ]}
    # df = pd.DataFrame(df)
    #
    # transf_df = {'category': [0.25, 0.25, 0.25, 0.25, 0.25,
    #                           0.25, 0.25, 0.25, 0.25, 0.25,
    #                           0.50, 0.50, 0.50, 0.50, 0.50,
    #                           0.50, 10000.00, 10000.00, 10000.00, 10000.00],
    #              'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    # transf_df = pd.DataFrame(transf_df)
    #
    # encoder = WoERatioCategoricalEncoder(encoding_method='ratio', variables=['category'])
    # encoder.fit(df['category'].to_frame(), df['target'])
    # X = encoder.transform(df['category'].to_frame())
    #
    # pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())

    # TEST WOE:
    # test dataframe
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, ]}
    df = pd.DataFrame(df)

    # transformed dataframe
    transf_df = {'category': [-1.386294, -1.386294, -1.386294, -1.386294, -1.386294,
                              -1.386294, -1.386294, -1.386294, -1.386294, -1.386294,
                              0.693147, 0.693147, 0.693147, 0.693147, 0.693147,
                              0.693147, 0.000000, 0.000000, 0.000000, 0.000000],
                 'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    transf_df = pd.DataFrame(transf_df)

    encoder = WoERatioCategoricalEncoder(encoding_method='woe', variables=['category'])
    encoder.fit(df['category'].to_frame(), df['target'])
    X = encoder.transform(df['category'].to_frame())

    pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())
    assert encoder.variables == ['category']
    assert encoder.encoder_dict_ == {'category': {'A': -1.3862943611198906,
                                                  'B': 0.69314718055994518,
                                                  'C': 0.0}}
    assert encoder.input_shape_ == (20, 1)

    # # prob(1)==1
    # df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
    #       'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, ]}
    # df = pd.DataFrame(df)
    #
    # transf_df = {'category': [-1.386294, -1.386294, -1.386294, -1.386294, -1.386294,
    #                           -1.386294, -1.386294, -1.386294, -1.386294, -1.386294,
    #                           0.693147, 0.693147, 0.693147, 0.693147, 0.693147,
    #                           0.693147, 9.210340, 9.210340, 9.210340, 9.210340],
    #              'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    # transf_df = pd.DataFrame(transf_df)
    #
    # encoder = WoERatioCategoricalEncoder(encoding_method='woe', variables=['category'])
    # encoder.fit(df['category'].to_frame(), df['target'])
    # X = encoder.transform(df['category'].to_frame())
    # pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())
    #
    # # prob(1)==0
    # df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
    #       'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, ]}
    # df = pd.DataFrame(df)
    #
    # transf_df = {'category': [-1.386294, -1.386294, -1.386294, -1.386294, -1.386294,
    #                           -1.386294, -1.386294, -1.386294, -1.386294, -1.386294,
    #                           0.693147, 0.693147, 0.693147, 0.693147, 0.693147,
    #                           0.693147, -9.210340, -9.210340, -9.210340, -9.210340],
    #              'target': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    # transf_df = pd.DataFrame(transf_df)
    #
    # encoder = WoERatioCategoricalEncoder(encoding_method='woe', variables=['category'])
    # encoder.fit(df['category'].to_frame(), df['target'])
    # X = encoder.transform(df['category'].to_frame())
    # pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())


def test_RareLabelEncoder():
    df = {'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 +
                      ['E'] * 2 + ['F'] * 2 + ['G'] * 2 + ['H'] * 2 + ['I'] * 10 + ['K'] * 5,
          'target': [1] * 63}
    df = pd.DataFrame(df)

    transf_df = {'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 +
                             ['Rare'] * 8 + ['I'] * 10 + ['K'] * 5,
                 'target': [1] * 63}
    transf_df = pd.DataFrame(transf_df)

    encoder = RareLabelCategoricalEncoder(tol=0.05, n_categories=9, variables=['category'])
    encoder.fit(df)
    X = encoder.transform(df)

    pd.testing.assert_frame_equal(X, transf_df)
    assert encoder.variables == ['category']
    assert encoder.input_shape_ == (63, 2)

    df = {'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 +
                      ['E'] * 2 + ['F'] * 2 + ['G'] * 2 + ['H'] * 2 + ['I'] * 10 + ['K'] * 5,
          'target': [1] * 63}
    df = pd.DataFrame(df)

    transf_df = {'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 +
                             ['E'] * 2 + ['F'] * 2 + ['G'] * 2 + ['H'] * 2 + ['I'] * 10 + ['K'] * 5,
                 'target': [1] * 63}
    transf_df = pd.DataFrame(transf_df)

    encoder = RareLabelCategoricalEncoder(tol=0.01, n_categories=9, variables=['category'])
    encoder.fit(df)
    X = encoder.transform(df)

    pd.testing.assert_frame_equal(X, transf_df)
    assert encoder.variables == ['category']
    assert encoder.input_shape_ == (63, 2)
