import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import WoEEncoder


def test_WoEEncoder(dataframe_enc, dataframe_enc_rare, dataframe_enc_na):

    # test case 1: automatically select variables, woe
    encoder = WoEEncoder(variables=None)
    encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
    X = encoder.transform(dataframe_enc[['var_A', 'var_B']])

    # transformed dataframe
    transf_df = dataframe_enc.copy()
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
    assert encoder.variables == ['var_A', 'var_B']
    # fit params
    assert encoder.encoder_dict_ == {'var_A': {'A': 0.15415067982725836, 'B': -0.5389965007326869, 'C': 0.8472978603872037},
                                     'var_B': {'A': -0.5389965007326869, 'B': 0.15415067982725836, 'C': 0.8472978603872037}}
    assert encoder.input_shape_ == (20, 2)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[['var_A', 'var_B']])

    # test case 2: raises error if target is  not passed
    with pytest.raises(TypeError):
        encoder.fit(dataframe_enc)

    # test case 3: when dataset to be transformed contains categories not present in training dataset
    with pytest.warns(UserWarning):
        encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
        encoder.transform(dataframe_enc_rare[['var_A', 'var_B']])

    # test case 4: the target is not binary
    with pytest.raises(ValueError):
        df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
              'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
              'target': [1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
        df = pd.DataFrame(df)
        encoder.fit(df[['var_A', 'var_B']], df['target'])

    # test case 5: when the denominator probability is zero
    with pytest.raises(ValueError):
        df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
              'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
              'target': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
        df = pd.DataFrame(df)
        encoder.fit(df[['var_A', 'var_B']], df['target'])

    # # test case 6: when the numerator probability is zero, woe
    # with pytest.raises(ValueError):
    #     df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
    #           'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
    #           'target': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    #     df = pd.DataFrame(df)
    #     encoder.fit(df[['var_A', 'var_B']], df['target'])

    # # test case 7: when the denominator probability is zero, woe
    # with pytest.raises(ValueError):
    #     df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
    #           'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
    #           'target': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    #     df = pd.DataFrame(df)
    #     encoder.fit(df[['var_A', 'var_B']], df['target'])

    # test case 8: non fitted error
    with pytest.raises(NotFittedError):
        imputer = WoEEncoder()
        imputer.transform(dataframe_enc)

    # test case 9: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder.fit(dataframe_enc_na[['var_A', 'var_B']], dataframe_enc_na['target'])

    # test case 10: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
        encoder.transform(dataframe_enc_na)