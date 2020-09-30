import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import ReciprocalTransformer


def test_ReciprocalTransformer(dataframe_vartypes, dataframe_na):
    # test case 1: automatically select variables
    transformer = ReciprocalTransformer(variables=None)
    X = transformer.fit_transform(dataframe_vartypes)

    # transformed dataframe
    transf_df = dataframe_vartypes.copy()
    transf_df['Age'] = [0.05, 0.047619, 0.0526316, 0.0555556]
    transf_df['Marks'] = [1.11111, 1.25, 1.42857, 1.66667]

    # init params
    assert transformer.variables == ['Age', 'Marks']
    # fit params
    assert transformer.input_shape_ == (4, 5)
    # transform params
    pd.testing.assert_frame_equal(X, transf_df)

    # test case 2: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(dataframe_na)

    # test case 3: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])

    # test error when data contains value zero
    df_neg = dataframe_vartypes.copy()
    df_neg.loc[1, 'Age'] = 0

    # test case 4: when variable contains zero, fit
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(df_neg)

    # test case 5: when variable contains zero, transform
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(df_neg)

    with pytest.raises(NotFittedError):
        transformer = ReciprocalTransformer()
        transformer.transform(dataframe_vartypes)