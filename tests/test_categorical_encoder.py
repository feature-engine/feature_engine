# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause
import pytest
import pandas as pd
from sklearn.exceptions import NotFittedError

from feature_engine.categorical_encoders import CountFrequencyCategoricalEncoder
from feature_engine.categorical_encoders import OrdinalCategoricalEncoder
from feature_engine.categorical_encoders import MeanCategoricalEncoder
from feature_engine.categorical_encoders import WoERatioCategoricalEncoder
from feature_engine.categorical_encoders import OneHotCategoricalEncoder
from feature_engine.categorical_encoders import RareLabelCategoricalEncoder


def test_CountFrequencyCategoricalEncoder(dataframe_enc, dataframe_enc_rare, dataframe_enc_na):
    # test case 1: 1 variable, counts
    encoder = CountFrequencyCategoricalEncoder(encoding_method='count', variables=['var_A'])
    X = encoder.fit_transform(dataframe_enc)

    # transformed dataframe
    transf_df = dataframe_enc.copy()
    transf_df['var_A'] = [6, 6, 6, 6, 6, 6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 4, 4, 4]

    # init params
    assert encoder.encoding_method == 'count'
    assert encoder.variables == ['var_A']
    # fit params
    assert encoder.encoder_dict_ == {'var_A': {'A': 6, 'B': 10, 'C': 4}}
    assert encoder.input_shape_ == (20, 3)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    # test case 2: automatically select variables, frequency
    encoder = CountFrequencyCategoricalEncoder(encoding_method='frequency', variables=None)
    X = encoder.fit_transform(dataframe_enc)

    # transformed dataframe
    transf_df['var_A'] = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                          0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2]
    transf_df['var_B'] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3,
                          0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]

    # init params
    assert encoder.encoding_method == 'frequency'
    assert encoder.variables == ['var_A', 'var_B']
    # fit params
    assert encoder.encoder_dict_ == {'var_A': {'A': 0.3, 'B': 0.5, 'C': 0.2},
                                     'var_B': {'A': 0.5, 'B': 0.3, 'C': 0.2}}
    assert encoder.input_shape_ == (20, 3)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    with pytest.raises(ValueError):
        CountFrequencyCategoricalEncoder(encoding_method='arbitrary')

    # test case 3: when dataset to be transformed contains categories not present in training dataset
    with pytest.warns(UserWarning):
        encoder = CountFrequencyCategoricalEncoder()
        encoder.fit(dataframe_enc)
        encoder.transform(dataframe_enc_rare)

    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = CountFrequencyCategoricalEncoder()
        encoder.fit(dataframe_enc_na)

    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = CountFrequencyCategoricalEncoder()
        encoder.fit(dataframe_enc)
        encoder.transform(dataframe_enc_na)

    with pytest.raises(NotFittedError):
        imputer = CountFrequencyCategoricalEncoder()
        imputer.transform(dataframe_enc)


def test_OrdinalCategoricalEncoder(dataframe_enc, dataframe_enc_rare, dataframe_enc_na):
    # test case 1: 1 variable, ordered encoding
    encoder = OrdinalCategoricalEncoder(encoding_method='ordered', variables=['var_A'])
    encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
    X = encoder.transform(dataframe_enc[['var_A', 'var_B']])

    # transformed dataframe
    transf_df = dataframe_enc.copy()
    transf_df['var_A'] = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]

    # init params
    assert encoder.encoding_method == 'ordered'
    assert encoder.variables == ['var_A']
    # fit params
    assert encoder.encoder_dict_ == {'var_A': {'A': 1, 'B': 0, 'C': 2}}
    assert encoder.input_shape_ == (20, 2)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df[['var_A', 'var_B']])

    # test case 2: automatically select variables, unordered encoding
    encoder = OrdinalCategoricalEncoder(encoding_method='arbitrary', variables=None)
    X = encoder.fit_transform(dataframe_enc)

    # transformed dataframe
    transf_df['var_A'] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    transf_df['var_B'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]

    # init params
    assert encoder.encoding_method == 'arbitrary'
    assert encoder.variables == ['var_A', 'var_B']
    # fit params
    assert encoder.encoder_dict_ == {'var_A': {'A': 0, 'B': 1, 'C': 2},
                                     'var_B': {'A': 0, 'B': 1, 'C': 2}}
    assert encoder.input_shape_ == (20, 3)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    with pytest.raises(ValueError):
        OrdinalCategoricalEncoder(encoding_method='other')

    # test case 3: raises error if target is  not passed
    with pytest.raises(ValueError):
        encoder = OrdinalCategoricalEncoder(encoding_method='ordered')
        encoder.fit(dataframe_enc)

    # test case 4: when dataset to be transformed contains categories not present in training dataset
    with pytest.warns(UserWarning):
        encoder = OrdinalCategoricalEncoder(encoding_method='arbitrary')
        encoder.fit(dataframe_enc)
        encoder.transform(dataframe_enc_rare)

    with pytest.raises(NotFittedError):
        imputer = OrdinalCategoricalEncoder()
        imputer.transform(dataframe_enc)

    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = OrdinalCategoricalEncoder(encoding_method='arbitrary')
        encoder.fit(dataframe_enc_na)

    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = OrdinalCategoricalEncoder(encoding_method='arbitrary')
        encoder.fit(dataframe_enc)
        encoder.transform(dataframe_enc_na)


def test_MeanCategoricalEncoder(dataframe_enc, dataframe_enc_rare, dataframe_enc_na):
    # test case 1: 1 variable
    encoder = MeanCategoricalEncoder(variables=['var_A'])
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
    encoder = MeanCategoricalEncoder(variables=None)
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
        encoder = MeanCategoricalEncoder()
        encoder.fit(dataframe_enc)

    # test case 4: when dataset to be transformed contains categories not present in training dataset
    with pytest.warns(UserWarning):
        encoder = MeanCategoricalEncoder()
        encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
        encoder.transform(dataframe_enc_rare[['var_A', 'var_B']])

    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = MeanCategoricalEncoder()
        encoder.fit(dataframe_enc_na[['var_A', 'var_B']], dataframe_enc_na['target'])

    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = MeanCategoricalEncoder()
        encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
        encoder.transform(dataframe_enc_na)

    with pytest.raises(NotFittedError):
        imputer = OrdinalCategoricalEncoder()
        imputer.transform(dataframe_enc)


def test_WoERatioCategoricalEncoder(dataframe_enc, dataframe_enc_rare, dataframe_enc_na):
    # test case 1: 1 variable, ratio
    encoder = WoERatioCategoricalEncoder(encoding_method='ratio', variables=['var_A'])
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
    encoder = WoERatioCategoricalEncoder(encoding_method='log_ratio', variables=None)
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
    encoder = WoERatioCategoricalEncoder(encoding_method='woe', variables=None)
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
        WoERatioCategoricalEncoder(encoding_method='other')

    # test case 4: raises error if target is  not passed
    with pytest.raises(TypeError):
        encoder = WoERatioCategoricalEncoder(encoding_method='woe')
        encoder.fit(dataframe_enc)

    # test case 5: when dataset to be transformed contains categories not present in training dataset
    with pytest.warns(UserWarning):
        encoder = WoERatioCategoricalEncoder(encoding_method='woe')
        encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
        encoder.transform(dataframe_enc_rare[['var_A', 'var_B']])

    # test case 6: the target is not binary
    with pytest.raises(ValueError):
        df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
              'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
              'target': [1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
        df = pd.DataFrame(df)
        encoder = WoERatioCategoricalEncoder(encoding_method='woe')
        encoder.fit(df[['var_A', 'var_B']], df['target'])

    # test case 7: when the denominator probability is zero, ratio
    with pytest.raises(ValueError):
        df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
              'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
              'target': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
        df = pd.DataFrame(df)
        encoder = WoERatioCategoricalEncoder(encoding_method='ratio')
        encoder.fit(df[['var_A', 'var_B']], df['target'])

    # test case 8: when the denominator probability is zero, log_ratio
    with pytest.raises(ValueError):
        df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
              'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
              'target': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
        df = pd.DataFrame(df)
        encoder = WoERatioCategoricalEncoder(encoding_method='log_ratio')
        encoder.fit(df[['var_A', 'var_B']], df['target'])

    # test case 9: when the numerator probability is zero, only applies to log_ratio
    with pytest.raises(ValueError):
        df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
              'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
              'target': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
        df = pd.DataFrame(df)
        encoder = WoERatioCategoricalEncoder(encoding_method='log_ratio')
        encoder.fit(df[['var_A', 'var_B']], df['target'])

    # # test case 10: when the numerator probability is zero, woe
    # with pytest.raises(ValueError):
    #     df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
    #           'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
    #           'target': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    #     df = pd.DataFrame(df)
    #     encoder = WoERatioCategoricalEncoder(encoding_method='woe')
    #     encoder.fit(df[['var_A', 'var_B']], df['target'])

    # # test case 11: when the denominator probability is zero, woe
    # with pytest.raises(ValueError):
    #     df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4,
    #           'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4,
    #           'target': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]}
    #     df = pd.DataFrame(df)
    #     encoder = WoERatioCategoricalEncoder(encoding_method='woe')
    #     encoder.fit(df[['var_A', 'var_B']], df['target'])

    # test case 12: non fitted error
    with pytest.raises(NotFittedError):
        imputer = WoERatioCategoricalEncoder()
        imputer.transform(dataframe_enc)

    # test case 13: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = WoERatioCategoricalEncoder(encoding_method='woe')
        encoder.fit(dataframe_enc_na[['var_A', 'var_B']], dataframe_enc_na['target'])

    # test case 14: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = WoERatioCategoricalEncoder(encoding_method='woe')
        encoder.fit(dataframe_enc[['var_A', 'var_B']], dataframe_enc['target'])
        encoder.transform(dataframe_enc_na)


def test_OneHotCategoricalEncoder(dataframe_enc_big, dataframe_enc_big_na):
    # test case 1: encode all categories into k binary variables, select variables automatically
    encoder = OneHotCategoricalEncoder(top_categories=None, variables=None, drop_last=False)
    X = encoder.fit_transform(dataframe_enc_big)

    # init params
    assert encoder.top_categories is None
    assert encoder.variables == ['var_A', 'var_B', 'var_C']
    assert encoder.drop_last == False
    # fit params
    transf = {'var_A_A': 6, 'var_A_B': 10, 'var_A_C': 4, 'var_A_D': 10, 'var_A_E': 2, 'var_A_F': 2, 'var_A_G': 6,
              'var_B_A': 10, 'var_B_B': 6, 'var_B_C': 4, 'var_B_D': 10, 'var_B_E': 2, 'var_B_F': 2, 'var_B_G': 6,
              'var_C_A': 4, 'var_C_B': 6, 'var_C_C': 10, 'var_C_D': 10, 'var_C_E': 2, 'var_C_F': 2, 'var_C_G': 6}

    assert encoder.input_shape_ == (40, 3)
    # transform params
    assert X.sum().to_dict() == transf
    assert 'var_A' not in X.columns

    # test case 2: encode all categories into k-1 binary variables, pass list of variables
    encoder = OneHotCategoricalEncoder(top_categories=None, variables=['var_A', 'var_B'], drop_last=True)
    X = encoder.fit_transform(dataframe_enc_big)

    # init params
    assert encoder.top_categories is None
    assert encoder.variables == ['var_A', 'var_B']
    assert encoder.drop_last == True
    # fit params
    transf = {'var_A_A': 6, 'var_A_B': 10, 'var_A_C': 4, 'var_A_D': 10, 'var_A_E': 2, 'var_A_F': 2,
              'var_B_A': 10, 'var_B_B': 6, 'var_B_C': 4, 'var_B_D': 10, 'var_B_E': 2, 'var_B_F': 2}
    assert encoder.input_shape_ == (40, 3)
    # transform params
    for col in transf.keys():
        assert X[col].sum() == transf[col]
    assert 'var_B' not in X.columns
    assert 'var_B_G' not in X.columns
    assert 'var_C' in X.columns

    # test case 3: encode only the most popular categories
    encoder = OneHotCategoricalEncoder(top_categories=4, variables=None, drop_last=False)
    X = encoder.fit_transform(dataframe_enc_big)

    # init params
    assert encoder.top_categories == 4
    # fit params
    transf = {'var_A_D': 10, 'var_A_B': 10, 'var_A_A': 6, 'var_A_G': 6,
              'var_B_A': 10, 'var_B_D': 10, 'var_B_G': 6, 'var_B_B': 6,
              'var_C_D': 10, 'var_C_C': 10, 'var_C_G': 6, 'var_C_B': 6}

    assert encoder.input_shape_ == (40, 3)
    # transform params
    for col in transf.keys():
        assert X[col].sum() == transf[col]
    assert 'var_B' not in X.columns
    assert 'var_B_F' not in X.columns

    with pytest.raises(ValueError):
        OneHotCategoricalEncoder(top_categories=0.5)

    with pytest.raises(ValueError):
        OneHotCategoricalEncoder(drop_last=0.5)

    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = OneHotCategoricalEncoder()
        encoder.fit(dataframe_enc_big_na)

    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = OneHotCategoricalEncoder()
        encoder.fit(dataframe_enc_big)
        encoder.transform(dataframe_enc_big_na)


def test_RareLabelEncoder(dataframe_enc_big, dataframe_enc_big_na):
    # test case 1: defo params, automatically select variables
    encoder = RareLabelCategoricalEncoder(tol=0.06, n_categories=5, variables=None, replace_with='Rare')
    X = encoder.fit_transform(dataframe_enc_big)

    df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['C'] * 4 + ['D'] * 10 + ['Rare'] * 4 + ['G'] * 6,
          'var_B': ['A'] * 10 + ['B'] * 6 + ['C'] * 4 + ['D'] * 10 + ['Rare'] * 4 + ['G'] * 6,
          'var_C': ['A'] * 4 + ['B'] * 6 + ['C'] * 10 + ['D'] * 10 + ['Rare'] * 4 + ['G'] * 6, }
    df = pd.DataFrame(df)

    # init params
    assert encoder.tol == 0.06
    assert encoder.n_categories == 5
    assert encoder.replace_with == 'Rare'
    assert encoder.variables == ['var_A', 'var_B', 'var_C']
    # fit params
    assert encoder.input_shape_ == (40, 3)
    # transform params
    pd.testing.assert_frame_equal(X, df)

    # test case 2: user provides alternative grouping value and variable list
    encoder = RareLabelCategoricalEncoder(tol=0.15, n_categories=5, variables=['var_A', 'var_B'], replace_with='Other')
    X = encoder.fit_transform(dataframe_enc_big)

    df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['Other'] * 4 + ['D'] * 10 + ['Other'] * 4 + ['G'] * 6,
          'var_B': ['A'] * 10 + ['B'] * 6 + ['Other'] * 4 + ['D'] * 10 + ['Other'] * 4 + ['G'] * 6,
          'var_C': ['A'] * 4 + ['B'] * 6 + ['C'] * 10 + ['D'] * 10 + ['E'] * 2 + ['F'] * 2 + ['G'] * 6}
    df = pd.DataFrame(df)

    # init params
    assert encoder.tol == 0.15
    assert encoder.n_categories == 5
    assert encoder.replace_with == 'Other'
    assert encoder.variables == ['var_A', 'var_B']
    # fit params
    assert encoder.input_shape_ == (40, 3)
    # transform params
    pd.testing.assert_frame_equal(X, df)

    with pytest.raises(ValueError):
        encoder = RareLabelCategoricalEncoder(tol=5)

    with pytest.raises(ValueError):
        encoder = RareLabelCategoricalEncoder(n_categories=0.5)

    with pytest.raises(ValueError):
        encoder = RareLabelCategoricalEncoder(replace_with=0.5)

    # test case 3: when the variable has low cardinality
    with pytest.warns(UserWarning):
        encoder = RareLabelCategoricalEncoder(n_categories=10)
        encoder.fit(dataframe_enc_big)

    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = RareLabelCategoricalEncoder(n_categories=4)
        encoder.fit(dataframe_enc_big_na)

    # test case 5: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = RareLabelCategoricalEncoder(n_categories=4)
        encoder.fit(dataframe_enc_big)
        encoder.transform(dataframe_enc_big_na)
