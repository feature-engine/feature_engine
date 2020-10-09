import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.imputation import CategoricalImputer


def test_imputer_with_string_missing_and_automatically_find_variables(dataframe_na):
    # set up transformer
    imputer = CategoricalImputer(imputation_method='missing', variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    # set up expected output
    X_reference = dataframe_na.copy()
    X_reference['Name'] = X_reference['Name'].fillna('Missing')
    X_reference['City'] = X_reference['City'].fillna('Missing')
    X_reference['Studies'] = X_reference['Studies'].fillna('Missing')

    # test init params
    assert imputer.imputation_method == 'missing'
    assert imputer.variables == ['Name', 'City', 'Studies']

    # tes fit attributes
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Name': 'Missing', 'City': 'Missing', 'Studies': 'Missing'}

    # test transform output
    # selected columns should have no NA
    # non selected columns should still have NA
    assert X_transformed[['Name', 'City', 'Studies']].isnull().sum().sum() == 0
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_imputer_with_user_defined_string_and_automatically_select_variables(dataframe_na):
    # set up imputer
    imputer = CategoricalImputer(imputation_method='missing', fill_value='Unknown', variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    # set up expected output
    X_reference = dataframe_na.copy()
    X_reference['Name'] = X_reference['Name'].fillna('Unknown')
    X_reference['City'] = X_reference['City'].fillna('Unknown')
    X_reference['Studies'] = X_reference['Studies'].fillna('Unknown')

    # test init params
    assert imputer.imputation_method == 'missing'
    assert imputer.fill_value == 'Unknown'
    assert imputer.variables == ['Name', 'City', 'Studies']

    # tes fit attributes
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Name': 'Unknown', 'City': 'Unknown', 'Studies': 'Unknown'}

    # test transform output:
    assert X_transformed[['Name', 'City', 'Studies']].isnull().sum().sum() == 0
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_mode_imputation_and_user_enters_single_variable(dataframe_na):
    # set up imputer
    imputer = CategoricalImputer(imputation_method='frequent', variables='City')
    X_transformed = imputer.fit_transform(dataframe_na)

    # set up expected result
    X_reference = dataframe_na.copy()
    X_reference['City'] = X_reference['City'].fillna('London')

    # test init, fit and transform params, attr and output
    assert imputer.imputation_method == 'frequent'
    assert imputer.variables == ['City']
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'City': 'London'}
    assert X_transformed['City'].isnull().sum() == 0
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_mode_imputation_when_user_enters_multiple_variables(dataframe_na):
    # set up imputer
    imputer = CategoricalImputer(imputation_method='frequent', variables=['Studies', 'City'])
    X_transformed = imputer.fit_transform(dataframe_na)

    # set up expected output
    X_reference = dataframe_na.copy()
    X_reference['City'] = X_reference['City'].fillna('London')
    X_reference['Studies'] = X_reference['Studies'].fillna('Bachelor')

    # test fit attr and transform output
    assert imputer.imputer_dict_ == {'Studies': 'Bachelor', 'City': 'London'}
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_imputation_of_numerical_variables_cast_as_object_and_returned_as_numerical(dataframe_na):
    # test case: imputing of numerical variables cast as object + return numeric
    dataframe_na['Marks'] = dataframe_na['Marks'].astype('O')
    imputer = CategoricalImputer(imputation_method='frequent', variables=['City', 'Studies', 'Marks'])
    X_transformed = imputer.fit_transform(dataframe_na)

    X_reference = dataframe_na.copy()
    X_reference['Marks'] = X_reference['Marks'].fillna(0.8)
    X_reference['City'] = X_reference['City'].fillna('London')
    X_reference['Studies'] = X_reference['Studies'].fillna('Bachelor')
    assert imputer.variables == ['City', 'Studies', 'Marks']
    assert imputer.imputer_dict_ == {'Studies': 'Bachelor', 'City': 'London', 'Marks': 0.8}
    assert X_transformed['Marks'].dtype == 'float'
    pd.testing.assert_frame_equal(X_transformed, X_reference)


def test_imputation_of_numerical_variables_cast_as_object_and_returned_as_object(dataframe_na):
    # test case 6: imputing of numerical variables cast as object + return as object after imputation
    dataframe_na['Marks'] = dataframe_na['Marks'].astype('O')
    imputer = CategoricalImputer(imputation_method='frequent', variables=['City', 'Studies', 'Marks'],
                                 return_object=True)
    X_transformed = imputer.fit_transform(dataframe_na)
    assert X_transformed['Marks'].dtype == 'O'


def test_raises_error_when_imputation_method_not_frequent_or_missing():
    with pytest.raises(ValueError):
        imputer = CategoricalImputer(imputation_method='arbitrary')


def test_raises_error_when_variable_contains_multiple_modes(dataframe_na):
    with pytest.raises(ValueError):
        imputer = CategoricalImputer(imputation_method='frequent')
        imputer.fit(dataframe_na)


def test_raises_non_fitted_error(dataframe_na):
    with pytest.raises(NotFittedError):
        imputer = CategoricalImputer()
        imputer.transform(dataframe_na)
