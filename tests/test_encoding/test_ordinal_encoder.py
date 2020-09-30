import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import OrdinalEncoder


def test_OrdinalEncoder(dataframe_enc, dataframe_enc_rare, dataframe_enc_na):
    # test case 1: 1 variable, ordered encoding
    encoder = OrdinalEncoder(encoding_method='ordered', variables=['var_A'])
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
    encoder = OrdinalEncoder(encoding_method='arbitrary', variables=None)
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
        OrdinalEncoder(encoding_method='other')

    # test case 3: raises error if target is  not passed
    with pytest.raises(ValueError):
        encoder = OrdinalEncoder(encoding_method='ordered')
        encoder.fit(dataframe_enc)

    # test case 4: when dataset to be transformed contains categories not present in training dataset
    with pytest.warns(UserWarning):
        encoder = OrdinalEncoder(encoding_method='arbitrary')
        encoder.fit(dataframe_enc)
        encoder.transform(dataframe_enc_rare)

    with pytest.raises(NotFittedError):
        imputer = OrdinalEncoder()
        imputer.transform(dataframe_enc)

    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = OrdinalEncoder(encoding_method='arbitrary')
        encoder.fit(dataframe_enc_na)

    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = OrdinalEncoder(encoding_method='arbitrary')
        encoder.fit(dataframe_enc)
        encoder.transform(dataframe_enc_na)