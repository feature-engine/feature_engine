import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import CountFrequencyEncoder


def test_encode_1_variable_with_counts(dataframe_enc):
    # test case 1: 1 variable, counts
    encoder = CountFrequencyEncoder(encoding_method='count', variables=['var_A'])
    X = encoder.fit_transform(dataframe_enc)

    # expected result
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


def test_automatically_select_variables_encode_with_frequency(dataframe_enc):
    # test case 2: automatically select variables, frequency
    encoder = CountFrequencyEncoder(encoding_method='frequency', variables=None)
    X = encoder.fit_transform(dataframe_enc)

    # expected output
    transf_df = dataframe_enc.copy()
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


def test_raises_error_if_encoding_method_not_permitted_value():
    with pytest.raises(ValueError):
        CountFrequencyEncoder(encoding_method='arbitrary')


def test_raises_error_if_input_df_contains_categories_not_present_in_training_df(dataframe_enc, dataframe_enc_rare):
    # test case 3: when dataset to be transformed contains categories not present in training dataset
    with pytest.warns(UserWarning):
        encoder = CountFrequencyEncoder()
        encoder.fit(dataframe_enc)
        encoder.transform(dataframe_enc_rare)


def test_fit_raises_error_if_df_contains_na(dataframe_enc_na):
    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = CountFrequencyEncoder()
        encoder.fit(dataframe_enc_na)


def test_transform_raises_error_if_df_contains_na(dataframe_enc, dataframe_enc_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = CountFrequencyEncoder()
        encoder.fit(dataframe_enc)
        encoder.transform(dataframe_enc_na)


def test_raises_non_fitted_error(dataframe_enc):
    with pytest.raises(NotFittedError):
        encoder = CountFrequencyEncoder()
        encoder.transform(dataframe_enc)