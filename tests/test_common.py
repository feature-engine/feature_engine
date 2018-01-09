# Authors: Soledad Galli <solegalli1@gmail.com>

# License: BSD 3 clause

import pandas as pd
import pytest

from categorical_encoder import CategoricalEncoder, RareLabelEncoder


def test_CategoricalEncoder_count():
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0,]}
    
    df = pd.DataFrame(df)
    
    transf_df = {'category': [10,10,10,10,10,10,10,10,10,10,6,6,6,6,6,6,4,4,4,4],
                 'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0]}
    
    transf_df = pd.DataFrame(transf_df)
    
    encoder = CategoricalEncoder(encoding_method='count')
    encoder.fit(df, variables = ['category'])
    X = encoder.transform(df)
    
    pd.testing.assert_frame_equal(X, transf_df)
    
    
def test_CategoricalEncoder_frequency():
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0,]}
    
    df = pd.DataFrame(df)
    
    transf_df = {'category': [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.3,0.3,
                              0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2],
                 'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0]}
    
    transf_df = pd.DataFrame(transf_df)
    
    encoder = CategoricalEncoder(encoding_method='frequency')
    encoder.fit(df, variables = ['category'])
    X = encoder.transform(df)
    
    pd.testing.assert_frame_equal(X, transf_df)
    
    
def test_CategoricalEncoder_ordinal():
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0,]}
    
    df = pd.DataFrame(df)
    
    transf_df = {'category': [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2],
                 'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0]}
    
    transf_df = pd.DataFrame(transf_df)
    
    encoder = CategoricalEncoder(encoding_method='ordinal')
    encoder.fit(df['category'].to_frame(), df['target'] )
    X = encoder.transform(df['category'].to_frame())
    
    pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())
    
    
def test_CategoricalEncoder_mean():
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0,]}
    
    df = pd.DataFrame(df)
    
    transf_df = {'category': [0.200000,0.200000,0.200000,0.200000,0.200000,
                              0.200000,0.200000,0.200000,0.200000,0.200000,
                              0.333333,0.333333,0.333333,0.333333,0.333333,
                              0.333333,0.500000,0.500000,0.500000,0.500000],
                 'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0]}
    
    transf_df = pd.DataFrame(transf_df)
    
    encoder = CategoricalEncoder(encoding_method='mean')
    encoder.fit(df['category'].to_frame(), df['target'] )
    X = encoder.transform(df['category'].to_frame())
    
    pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())
    
    
def test_CategoricalEncoder_ratio():
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0,]}
    
    df = pd.DataFrame(df)
    
    transf_df = {'category': [0.25,0.25,0.25,0.25,0.25,
                              0.25,0.25,0.25,0.25,0.25,
                              0.50,0.50,0.50,0.50,0.50,
                              0.50,1.00,1.00,1.00,1.00],
                 'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0]}
    
    transf_df = pd.DataFrame(transf_df)
    
    encoder = CategoricalEncoder(encoding_method='ratio')
    encoder.fit(df['category'].to_frame(), df['target'] )
    X = encoder.transform(df['category'].to_frame())
    
    pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())
    
    # when one of the probabilities is zero
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,1,1,]}
    
    df = pd.DataFrame(df)
    
    transf_df = {'category': [0.25,0.25,0.25,0.25,0.25,
                              0.25,0.25,0.25,0.25,0.25,
                              0.50,0.50,0.50,0.50,0.50,
                              0.50,10000.00,10000.00,10000.00,10000.00],
                 'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0]}
    
    transf_df = pd.DataFrame(transf_df)
    
    encoder = CategoricalEncoder(encoding_method='ratio')
    encoder.fit(df['category'].to_frame(), df['target'] )
    X = encoder.transform(df['category'].to_frame())
    
    pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())
    
    
def test_CategoricalEncoder_woe():
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target' : [1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0, 1,1,0,0,]}
    
    df = pd.DataFrame(df)
    
    transf_df = {'category': [-1.386294,-1.386294,-1.386294,-1.386294,-1.386294,
                              -1.386294,-1.386294,-1.386294,-1.386294,-1.386294,
                              0.693147,0.693147,0.693147,0.693147,0.693147,
                              0.693147,0.000000,0.000000,0.000000,0.000000],
                 'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0]}
    
    transf_df = pd.DataFrame(transf_df)
    
    encoder = CategoricalEncoder(encoding_method='woe')
    encoder.fit(df['category'].to_frame(), df['target'] )
    X = encoder.transform(df['category'].to_frame())
    
    pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())
    
    # prob(1)==1
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target' : [1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0, 1,1,1,1,]}
    
    df = pd.DataFrame(df)
    
    transf_df = {'category': [-1.386294,-1.386294,-1.386294,-1.386294,-1.386294,
                              -1.386294,-1.386294,-1.386294,-1.386294,-1.386294,
                              0.693147,0.693147,0.693147,0.693147,0.693147,
                              0.693147,9.210340,9.210340,9.210340,9.210340],
                 'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0]}
    
    transf_df = pd.DataFrame(transf_df)
    
    encoder = CategoricalEncoder(encoding_method='woe')
    encoder.fit(df['category'].to_frame(), df['target'] )
    X = encoder.transform(df['category'].to_frame())
    
    pd.testing.assert_frame_equal(X, transf_df['category'].to_frame())
    
    # prob(1)==0
    df = {'category': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
          'target' : [1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0, 0,0,0,0,]}
    
    df = pd.DataFrame(df)
    
    transf_df = {'category': [-1.386294,-1.386294,-1.386294,-1.386294,-1.386294,
                              -1.386294,-1.386294,-1.386294,-1.386294,-1.386294,
                              0.693147,0.693147,0.693147,0.693147,0.693147,
                              0.693147,-9.210340,-9.210340,-9.210340,-9.210340],
                 'target' : [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0, 1,1,0,0]}
    
    transf_df = pd.DataFrame(transf_df)
    
    encoder = CategoricalEncoder(encoding_method='woe')
    encoder.fit(df['category'].to_frame(), df['target'] )
    X = encoder.transform(df['category'].to_frame())
    
    pd.testing.assert_frame_equal(X, transf_df['category'].to_frame()) 
    
    
def test_RareLabelEncoder():
    df = {'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 +
          ['E'] * 2 + ['F'] * 2 + ['G'] * 2 + ['H'] * 2 + [ 'I']* 10 + ['K']*5,
          'target' : [1]* 63}
    
    df = pd.DataFrame(df)
    
    transf_df = {'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 +
          ['Rare'] * 8  + [ 'I']* 10 + ['K']*5,
          'target' : [1]* 63}
    
    transf_df = pd.DataFrame(transf_df)
    
    encoder = RareLabelEncoder(tol = 0.05, n_categories = 9)
    encoder.fit(df, variables = ['category'])
    X = encoder.transform(df)
    
    pd.testing.assert_frame_equal(X, transf_df)
    
    
    df = {'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 +
          ['E'] * 2 + ['F'] * 2 + ['G'] * 2 + ['H'] * 2 + [ 'I']* 10 + ['K']*5,
          'target' : [1]* 63}
    
    df = pd.DataFrame(df)
    
    transf_df = {'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 +
          ['E'] * 2 + ['F'] * 2 + ['G'] * 2 + ['H'] * 2 + [ 'I']* 10 + ['K']*5,
          'target' : [1]* 63 }
    
    transf_df = pd.DataFrame(transf_df)
    
    encoder = RareLabelEncoder(tol = 0.01, n_categories = 9)
    encoder.fit(df, variables = ['category'])
    X = encoder.transform(df)
    
    pd.testing.assert_frame_equal(X, transf_df)
    
    
    
if __name__ == '__main__':
    test_CategoricalEncoder_count()
    test_CategoricalEncoder_frequency()
    test_CategoricalEncoder_ordinal()
    test_CategoricalEncoder_mean()
    test_CategoricalEncoder_ratio()
    test_CategoricalEncoder_woe()
    test_RareLabelEncoder()
    

