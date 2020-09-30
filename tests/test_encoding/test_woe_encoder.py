import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import WoEEncoder


def test_WoEEncoder(dataframe_enc, dataframe_enc_rare, dataframe_enc_na):
    # test case 1: 1 variable, ratio
    encoder = WoEEncoder(encoding_method='ratio', variables=['var_A'])
    encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
    X = encoder.transform(dataframe_enc[['var_A', 'var_B']])

    # transformed dataframe
    transf_df = dataframe_enc.copy()
    transf_df['var_A'] = [0.49999999999999994, 0.49999999999999994, 0.49999999999999994, 0.49999999999999994,
                          0.49999999999999994, 0.49999999999999994, 0.25, 0.25,
                          0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0]

    # init params
    assert encoder.encoding_method == 'ratio'
    assert encoder.variables == ['var_A']
    # fit params
    assert encoder.encoder_dict_ == {'var_A': {'A': 0.49999999999999994, 'B': 0.25, 'C': 1.0}}
    assert encoder.input_shape_ == (20, 2)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[['var_A', 'var_B']])

    # test case 2: automatically select variables, log_ratio
    encoder = WoEEncoder(encoding_method='log_ratio', variables=None)
    encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
    X = encoder.transform(dataframe_enc[['var_A', 'var_B']])

    # transformed dataframe
    transf_df['var_A'] = [-0.6931471805599454, -0.6931471805599454, -0.6931471805599454, -0.6931471805599454,
                          -0.6931471805599454, -0.6931471805599454, -1.3862943611198906, -1.3862943611198906,
                          -1.3862943611198906, -1.3862943611198906, -1.3862943611198906, -1.3862943611198906,
                          -1.3862943611198906, -1.3862943611198906, -1.3862943611198906, -1.3862943611198906,
                          0.0, 0.0, 0.0, 0.0]
    transf_df['var_B'] = [-1.3862943611198906, -1.3862943611198906, -1.3862943611198906, -1.3862943611198906,
                          -1.3862943611198906, -1.3862943611198906, -1.3862943611198906, -1.3862943611198906,
                          -1.3862943611198906, -1.3862943611198906, -0.6931471805599454, -0.6931471805599454,
                          -0.6931471805599454, -0.6931471805599454, -0.6931471805599454, -0.6931471805599454,
                          0.0, 0.0, 0.0, 0.0]

    # init params
    assert encoder.encoding_method == 'log_ratio'
    assert encoder.variables == ['var_A', 'var_B']
    # fit params
    assert encoder.encoder_dict_ == {'var_A': {'A': -0.6931471805599454, 'B': -1.3862943611198906, 'C': 0.0},
                                     'var_B': {'A': -1.3862943611198906, 'B': -0.6931471805599454, 'C': 0.0}}
    assert encoder.input_shape_ == (20, 2)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[['var_A', 'var_B']])

    # test case 3: automatically select variables, woe
    encoder = WoEEncoder(encoding_method='woe', variables=None)
    encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
    X = encoder.transform(dataframe_enc[['var_A', 'var_B']])

    # transformed dataframe
    transf_df['var_A'] = [0.15415067982725836, 0.15415067982725836, 0.15415067982725836, 0.15415067982725836,
                          0.15415067982725836, 0.15415067982725836, -0.5389965007326869, -0.5389965007326869,
                          -0.5389965007326869, -0.5389965007326869, -0.5389965007326869, -0.5389965007326869,
                          -0.5389965007326869, -0.5389965007326869, -0.5389965007326869, -0.5389965007326869,
                          0.8472978603872037, 0.8472978603872037, 0.8472978603872037, 0.8472978603872037]
    transf_df['var_B'] = [-0.5389965007326869, -0.5389965007326869, -0.5389965007326869, -0.5389965007326869,
                          -0.5389965007326869, -0.5389965007326869, -0.5389965007326869, -0.5389965007326869,
                          -0.5389965007326869, -0.5389965007326869, 0.15415067982725836, 0.15415067982725836,
                          0.15415067982725836, 0.15415067982725836, 0.15415067982725836, 0.15415067982725836,
                          0.8472978603872037, 0.8472978603872037, 0.8472978603872037, 0.8472978603872037]

    # init params
    assert encoder.encoding_method == 'woe'
    assert encoder.variables == ['var_A', 'var_B']
    # fit params
    assert encoder.encoder_dict_ == {'var_A': {'A': 0.15415067982725836, 'B': -0.5389965007326869, 'C': 0.8472978603872037},
                                     'var_B': {'A': -0.5389965007326869, 'B': 0.15415067982725836, 'C': 0.8472978603872037}}
    assert encoder.input_shape_ == (20, 2)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[['var_A', 'var_B']])

    # test error raise
    with pytest.raises(ValueError):
        WoEEncoder(encoding_method='other')

    # test case 4: raises error if target is  not passed
    with pytest.raises(TypeError):
        encoder = WoEEncoder(encoding_method='woe')
        encoder.fit(dataframe_enc)

    # test case 5: when dataset to be transformed contains categories not present in training dataset
    with pytest.warns(UserWarning):
        encoder = WoEEncoder(encoding_method='woe')
        encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
        encoder.transform(dataframe_enc_rare[['var_A', 'var_B']])

    # test case 6: the target is not binary
    with pytest.raises(ValueError):
        df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
              'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
              'target': [1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
        df = pd.DataFrame(df)
        encoder = WoEEncoder(encoding_method='woe')
        encoder.fit(df[['var_A', 'var_B']], df['target'])

    # test case 7: when the denominator probability is zero, ratio
    with pytest.raises(ValueError):
        df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
              'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
              'target': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
        df = pd.DataFrame(df)
        encoder = WoEEncoder(encoding_method='ratio')
        encoder.fit(df[['var_A', 'var_B']], df['target'])

    # test case 8: when the denominator probability is zero, log_ratio
    with pytest.raises(ValueError):
        df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
              'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
              'target': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
        df = pd.DataFrame(df)
        encoder = WoEEncoder(encoding_method='log_ratio')
        encoder.fit(df[['var_A', 'var_B']], df['target'])

    # test case 9: when the numerator probability is zero, only applies to log_ratio
    with pytest.raises(ValueError):
        df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
              'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
              'target': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
        df = pd.DataFrame(df)
        encoder = WoEEncoder(encoding_method='log_ratio')
        encoder.fit(df[['var_A', 'var_B']], df['target'])

    # # test case 10: when the numerator probability is zero, woe
    # with pytest.raises(ValueError):
    #     df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
    #           'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
    #           'target': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    #     df = pd.DataFrame(df)
    #     encoder = WoEEncoder(encoding_method='woe')
    #     encoder.fit(df[['var_A', 'var_B']], df['target'])

    # # test case 11: when the denominator probability is zero, woe
    # with pytest.raises(ValueError):
    #     df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
    #           'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
    #           'target': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    #     df = pd.DataFrame(df)
    #     encoder = WoEEncoder(encoding_method='woe')
    #     encoder.fit(df[['var_A', 'var_B']], df['target'])

    # test case 12: non fitted error
    with pytest.raises(NotFittedError):
        imputer = WoEEncoder()
        imputer.transform(dataframe_enc)

    # test case 13: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = WoEEncoder(encoding_method='woe')
        encoder.fit(dataframe_enc_na[['var_A', 'var_B']], dataframe_enc_na['target'])

    # test case 14: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = WoEEncoder(encoding_method='woe')
        encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
        encoder.transform(dataframe_enc_na)