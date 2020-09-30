import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import LogTransformer


def test_LogTransformer(dataframe_vartypes, dataframe_na):
    # test case 1: log base e, automatically select variables
    transformer = LogTransformer(base='e', variables=None)
    X = transformer.fit_transform(dataframe_vartypes)

    # transformed dataframe
    transf_df = dataframe_vartypes.copy()
    transf_df['Age'] = [2.99573, 3.04452, 2.94444, 2.89037]
    transf_df['Marks'] = [-0.105361, -0.223144, -0.356675, -0.510826]

    # init params
    assert transformer.base == 'e'
    assert transformer.variables == ['Age', 'Marks']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    # test case 2: log base 10, user passes variables
    transformer = LogTransformer(base='10', variables='Age')
    X = transformer.fit_transform(dataframe_vartypes)

    # transformed dataframe
    transf_df = dataframe_vartypes.copy()
    transf_df['Age'] = [1.30103, 1.32222, 1.27875, 1.25527]

    # init params
    assert transformer.base == '10'
    assert transformer.variables == ['Age']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    with pytest.raises(ValueError):
        LogTransformer(base='other')

    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(dataframe_na)

    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])

    # test error when data contains negative values
    df_neg = dataframe_vartypes.copy()
    df_neg.loc[1, 'Age'] = -1

    # test case 5: when variable contains negative value, fit
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(df_neg)

    # test case 6: when variable contains negative value, transform
    with pytest.raises(ValueError):
        transformer = LogTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(df_neg)

    with pytest.raises(NotFittedError):
        transformer = LogTransformer()
        transformer.transform(dataframe_vartypes)