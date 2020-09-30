import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.imputation import AddMissingIndicator


def test_AddMissingIndicator(dataframe_na):

    # test case 1: automatically detect variables with missing data
    imputer = AddMissingIndicator(how='missing_only', variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    # init params
    assert imputer.how == 'missing_only'
    assert imputer.variables == None
    # fit params
    assert imputer.variables_ == ['Name', 'City', 'Studies', 'Age', 'Marks']
    assert imputer.input_shape_ == (8, 6)
    # transform outputs
    assert X_transformed.shape == (8, 11)
    assert 'Name_na' in X_transformed.columns
    assert X_transformed['Name_na'].sum() == 2

    # test case 2: automatically detect all columns
    imputer = AddMissingIndicator(how='all', variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)
    assert imputer.variables_ == ['Name', 'City', 'Studies', 'Age', 'Marks', 'dob']
    assert X_transformed.shape == (8, 12)
    assert 'dob_na' in X_transformed.columns
    assert X_transformed['dob_na'].sum() == 0

    # test case 3: find variables with NA, among those entered by the user
    imputer = AddMissingIndicator(how='missing_only', variables=['City', 'Studies', 'Age', 'dob'])
    X_transformed = imputer.fit_transform(dataframe_na)
    assert imputer.variables == ['City', 'Studies', 'Age', 'dob']
    assert imputer.variables_ == ['City', 'Studies', 'Age']
    assert X_transformed.shape == (8, 9)
    assert 'City_na' in X_transformed.columns
    assert 'dob_na' not in X_transformed.columns
    assert X_transformed['City_na'].sum() == 2

    with pytest.raises(ValueError):
        AddMissingIndicator(how='arbitrary')

    with pytest.raises(NotFittedError):
        imputer = AddMissingIndicator()
        imputer.transform(dataframe_na)