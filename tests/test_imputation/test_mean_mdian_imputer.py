import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.imputation import MeanMedianImputer


def test_MeanMedianImputer(dataframe_na):

    # test case 1: automatically finds numerical variables
    imputer = MeanMedianImputer(imputation_method='mean', variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(28.714285714285715)
    ref_df['Marks'] = ref_df['Marks'].fillna(0.6833333333333332)

    # check init params
    assert imputer.imputation_method == 'mean'
    assert imputer.variables == ['Age', 'Marks']

    # check fit attributes
    assert imputer.imputer_dict_ == {'Age': 28.714285714285715, 'Marks': 0.6833333333333332}
    assert imputer.input_shape_ == (8, 6)

    # check transform output: indicated variables no NA
    # Not indicated variables still have NA
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() == 0
    assert X_transformed[['Name', 'City']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 2: single user determined variable
    imputer = MeanMedianImputer(imputation_method='median', variables=['Age'])
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(23.0)

    # init params
    assert imputer.imputation_method == 'median'
    assert imputer.variables == ['Age']

    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Age': 23.0}

    # transform params
    assert X_transformed['Age'].isnull().sum() == 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    with pytest.raises(ValueError):
        MeanMedianImputer(imputation_method='arbitrary')

    with pytest.raises(NotFittedError):
        imputer = MeanMedianImputer()
        imputer.transform(dataframe_na)