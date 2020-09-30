import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import MeanEncoder, OrdinalEncoder


def test_MeanEncoder(dataframe_enc, dataframe_enc_rare, dataframe_enc_na):
    # test case 1: 1 variable
    encoder = MeanEncoder(variables=['var_A'])
    encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
    X = encoder.transform(dataframe_enc[['var_A', 'var_B']])

    # transformed dataframe
    transf_df = dataframe_enc.copy()
    transf_df['var_A'] = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
                          0.3333333333333333, 0.3333333333333333, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                          0.2, 0.2, 0.5, 0.5, 0.5, 0.5]

    # init params
    assert encoder.variables == ['var_A']
    # fit params
    assert encoder.encoder_dict_ == {'var_A': {'A': 0.3333333333333333, 'B': 0.2, 'C': 0.5}}
    assert encoder.input_shape_ == (20, 2)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[['var_A', 'var_B']])

    # test case 2: automatically select variables
    encoder = MeanEncoder(variables=None)
    encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
    X = encoder.transform(dataframe_enc[['var_A', 'var_B']])

    # transformed dataframe
    transf_df['var_A'] = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
                          0.3333333333333333, 0.3333333333333333, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                          0.2, 0.2, 0.5, 0.5, 0.5, 0.5]
    transf_df['var_B'] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3333333333333333, 0.3333333333333333,
                          0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333,
                          0.5, 0.5, 0.5, 0.5]

    # init params
    assert encoder.variables == ['var_A', 'var_B']
    # fit params
    assert encoder.encoder_dict_ == {'var_A': {'A': 0.3333333333333333, 'B': 0.2, 'C': 0.5},
                                     'var_B': {'A': 0.2, 'B': 0.3333333333333333, 'C': 0.5}}
    assert encoder.input_shape_ == (20, 2)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[['var_A', 'var_B']])

    # test case 3: raises error if target is not passed
    with pytest.raises(TypeError):
        encoder = MeanEncoder()
        encoder.fit(dataframe_enc)

    # test case 4: when dataset to be transformed contains categories not present in training dataset
    with pytest.warns(UserWarning):
        encoder = MeanEncoder()
        encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
        encoder.transform(dataframe_enc_rare[['var_A', 'var_B']])

    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = MeanEncoder()
        encoder.fit(dataframe_enc_na[['var_A', 'var_B']], dataframe_enc_na['target'])

    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = MeanEncoder()
        encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
        encoder.transform(dataframe_enc_na)

    with pytest.raises(NotFittedError):
        imputer = OrdinalEncoder()
        imputer.transform(dataframe_enc)