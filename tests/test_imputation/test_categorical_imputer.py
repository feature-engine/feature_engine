import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.imputation import CategoricalImputer


def test_CategoricalVariableImputer(dataframe_na):

    # test case 1: imputation with missing + automatically select variables
    imputer = CategoricalImputer(imputation_method='missing', variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Name'] = ref_df['Name'].fillna('Missing')
    ref_df['City'] = ref_df['City'].fillna('Missing')
    ref_df['Studies'] = ref_df['Studies'].fillna('Missing')

    # init params
    assert imputer.imputation_method == 'missing'
    assert imputer.variables == ['Name', 'City', 'Studies']
    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Name': 'Missing', 'City': 'Missing', 'Studies': 'Missing'}
    # transform output
    assert X_transformed[['Name', 'City', 'Studies']].isnull().sum().sum() == 0
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 2: imputing custom user-defined string + automatically select variables
    imputer = CategoricalImputer(imputation_method='missing', fill_value='Unknown', variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Name'] = ref_df['Name'].fillna('Unknown')
    ref_df['City'] = ref_df['City'].fillna('Unknown')
    ref_df['Studies'] = ref_df['Studies'].fillna('Unknown')

    # init params
    assert imputer.imputation_method == 'missing'
    assert imputer.fill_value == 'Unknown'
    assert imputer.variables == ['Name', 'City', 'Studies']
    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Name': 'Unknown', 'City': 'Unknown', 'Studies': 'Unknown'}
    # transform output
    assert X_transformed[['Name', 'City', 'Studies']].isnull().sum().sum() == 0
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 3: mode imputation + user indicates 1 variable ONLY
    imputer = CategoricalImputer(imputation_method='frequent', variables='City')
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['City'] = ref_df['City'].fillna('London')

    assert imputer.imputation_method == 'frequent'
    assert imputer.variables == ['City']
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'City': 'London'}
    assert X_transformed['City'].isnull().sum() == 0
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 4: mode imputation + user indicates multiple variables
    imputer = CategoricalImputer(imputation_method='frequent', variables=['Studies', 'City'])
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['City'] = ref_df['City'].fillna('London')
    ref_df['Studies'] = ref_df['Studies'].fillna('Bachelor')

    assert imputer.imputer_dict_ == {'Studies': 'Bachelor', 'City': 'London'}
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 5: imputing of numerical variables cast as object + return numeric
    dataframe_na['Marks'] = dataframe_na['Marks'].astype('O')
    imputer = CategoricalImputer(imputation_method='frequent', variables=['City', 'Studies', 'Marks'])
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Marks'] = ref_df['Marks'].fillna(0.8)
    ref_df['City'] = ref_df['City'].fillna('London')
    ref_df['Studies'] = ref_df['Studies'].fillna('Bachelor')
    assert imputer.variables == ['City', 'Studies', 'Marks']
    assert imputer.imputer_dict_ == {'Studies': 'Bachelor', 'City': 'London', 'Marks': 0.8}
    assert X_transformed['Marks'].dtype == 'float'
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 6: imputing of numerical variables cast as object + return as object after imputation
    dataframe_na['Marks'] = dataframe_na['Marks'].astype('O')
    imputer = CategoricalImputer(imputation_method='frequent', variables=['City', 'Studies', 'Marks'],
                                 return_object=True)
    X_transformed = imputer.fit_transform(dataframe_na)
    assert X_transformed['Marks'].dtype == 'O'

    with pytest.raises(ValueError):
        imputer = CategoricalImputer(imputation_method='arbitrary')

    with pytest.raises(ValueError):
        imputer = CategoricalImputer(imputation_method='frequent')
        imputer.fit(dataframe_na)

    with pytest.raises(NotFittedError):
        imputer = CategoricalImputer()
        imputer.transform(dataframe_na)