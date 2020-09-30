import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import BoxCoxTransformer


def test_BoxCoxTransformer(dataframe_vartypes, dataframe_na):
    # test case 1: automatically select variables
    transformer = BoxCoxTransformer(variables=None)
    X = transformer.fit_transform(dataframe_vartypes)

    # transformed dataframe
    transf_df = dataframe_vartypes.copy()
    transf_df['Age'] = [9.78731, 10.1666, 9.40189, 9.0099]
    transf_df['Marks'] = [-0.101687, -0.207092, -0.316843, -0.431788]

    # init params
    assert transformer.variables == ['Age', 'Marks']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    # test case 2: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = BoxCoxTransformer()
        transformer.fit(dataframe_na)

    # test case 3: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = BoxCoxTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])

    # test error when data contains negative values
    df_neg = dataframe_vartypes.copy()
    df_neg.loc[1, 'Age'] = -1

    # test case 4: when variable contains negative value, fit
    with pytest.raises(ValueError):
        transformer = BoxCoxTransformer()
        transformer.fit(df_neg)

    # test case 5: when variable contains negative value, transform
    with pytest.raises(ValueError):
        transformer = BoxCoxTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(df_neg)

    with pytest.raises(NotFittedError):
        transformer = BoxCoxTransformer()
        transformer.transform(dataframe_vartypes)