# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import pandas as pd

from feature_engine.categorical_encoders import RareLabelCategoricalEncoder


def test_RareLabelEncoder():
    df = {'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 +
          ['E'] * 2 + ['F'] * 2 + ['G'] * 2 + ['H'] * 2 + [ 'I']* 10 + ['K']*5,
          'target' : [1]* 63}
    df = pd.DataFrame(df)
    
    transf_df = {'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 +
          ['Rare'] * 8  + [ 'I']* 10 + ['K']*5,
          'target' : [1]* 63}
    transf_df = pd.DataFrame(transf_df)
    
    encoder = RareLabelCategoricalEncoder(tol=0.05, n_categories=9, variables=['category'])
    encoder.fit(df)
    X = encoder.transform(df)
    
    pd.testing.assert_frame_equal(X, transf_df)
    enc_list = list(encoder.encoder_dict_['category'])
    enc_list.sort()
    assert enc_list == ['A', 'B', 'C', 'D', 'I', 'K']
    assert encoder.input_shape_ == (63,2)
       
    df = {'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 +
          ['E'] * 2 + ['F'] * 2 + ['G'] * 2 + ['H'] * 2 + [ 'I']* 10 + ['K']*5,
          'target' : [1]* 63}
    df = pd.DataFrame(df)
    
    transf_df = {'category': ['A'] * 10 + ['B'] * 10 + ['C'] * 10 + ['D'] * 10 +
          ['E'] * 2 + ['F'] * 2 + ['G'] * 2 + ['H'] * 2 + [ 'I']* 10 + ['K']*5,
          'target' : [1]* 63 }
    transf_df = pd.DataFrame(transf_df)
    
    encoder = RareLabelCategoricalEncoder(tol=0.01, n_categories=9, variables=['category'])
    encoder.fit(df)
    X = encoder.transform(df)
    
    pd.testing.assert_frame_equal(X, transf_df)
    enc_list = list(encoder.encoder_dict_['category'])
    enc_list.sort()
    assert enc_list == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K']
    assert encoder.input_shape_ == (63,2)
