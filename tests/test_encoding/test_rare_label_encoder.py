import pytest
import pandas as pd

from feature_engine.encoding import RareLabelEncoder


def test_RareLabelEncoder(dataframe_enc_big, dataframe_enc_big_na, dataframe_enc_rare):
    # test case 1: defo params, automatically select variables
    encoder = RareLabelEncoder(tol=0.06, n_categories=5, variables=None, replace_with='Rare')
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
    encoder = RareLabelEncoder(tol=0.15, n_categories=5, variables=['var_A', 'var_B'], replace_with='Other')
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
        encoder = RareLabelEncoder(tol=5)

    with pytest.raises(ValueError):
        encoder = RareLabelEncoder(n_categories=0.5)

    with pytest.raises(ValueError):
        encoder = RareLabelEncoder(replace_with=0.5)

    # test case 3: when the variable has low cardinality
    with pytest.warns(UserWarning):
        encoder = RareLabelEncoder(n_categories=10)
        encoder.fit(dataframe_enc_big)

    # test case 4: when dataset contains na, fit method
    with pytest.raises(ValueError):
        encoder = RareLabelEncoder(n_categories=4)
        encoder.fit(dataframe_enc_big_na)

    # test case 5: when dataset contains na, transform method
    with pytest.raises(ValueError):
        encoder = RareLabelEncoder(n_categories=4)
        encoder.fit(dataframe_enc_big)
        encoder.transform(dataframe_enc_big_na)

    # test case 6: user provides the maximum number of categories they want
    rare_encoder = RareLabelEncoder(tol=0.10, max_n_categories=4,
            n_categories=5)
    X = rare_encoder.fit_transform(dataframe_enc_big)
    df = {'var_A': ['A'] * 6 + ['B'] * 10 + ['Rare'] * 4 + ['D'] * 10 + ['Rare'] * 4 + ['G'] * 6,
          'var_B': ['A'] * 10 + ['B'] * 6 + ['Rare'] * 4 + ['D'] * 10 + ['Rare'] * 4 + ['G'] * 6,
          'var_C': ['Rare'] * 4 + ['B'] * 6 + ['C'] * 10 + ['D'] * 10 + ['Rare'] * 4 + ['G'] * 6, }
    df = pd.DataFrame(df)
    pd.testing.assert_frame_equal(X, df)
