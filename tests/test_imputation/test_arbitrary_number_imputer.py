import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.imputation import ArbitraryNumberImputer


def test_ArbitraryNumberImputer(dataframe_na):

    # test case 1: automatically select variables
    imputer = ArbitraryNumberImputer(arbitrary_number=99, variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(99)
    ref_df['Marks'] = ref_df['Marks'].fillna(99)

    # init params
    assert imputer.arbitrary_number == 99
    assert imputer.variables == ['Age', 'Marks']
    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Age': 99, 'Marks': 99}
    # transform params
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() == 0
    assert X_transformed[['Name', 'City']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 2: user indicates variables
    imputer = ArbitraryNumberImputer(arbitrary_number=-1, variables=['Age'])
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(-1)

    # init params
    assert imputer.arbitrary_number == -1
    assert imputer.variables == ['Age']
    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Age': -1}
    # transform output
    assert X_transformed['Age'].isnull().sum() == 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    with pytest.raises(ValueError):
        ArbitraryNumberImputer(arbitrary_number='arbitrary')

    with pytest.raises(NotFittedError):
        imputer = ArbitraryNumberImputer()
        imputer.transform(dataframe_na)

    # test case 3: arbitrary numbers passed as dict
    imputer = ArbitraryNumberImputer(imputer_dict={'Age': -42, 'Marks': -999})
    X_transformed = imputer.fit_transform(dataframe_na)
    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(-42)
    ref_df['Marks'] = ref_df['Marks'].fillna(-999)

    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Age': -42, 'Marks': -999}
    # transform params
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() == 0
    assert X_transformed[['Name', 'City']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    with pytest.raises(ValueError):
        ArbitraryNumberImputer(imputer_dict={'Age': 'arbitrary_number'})