import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.imputation import DropMissingData


def test_detect_variables_with_na(df_na):
    # test case 1: automatically detect variables with missing data
    imputer = DropMissingData(missing_only=True, variables=None)
    X_transformed = imputer.fit_transform(df_na)
    # init params
    assert imputer.missing_only is True
    assert imputer.variables is None
    # fit params
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.input_shape_ == (8, 6)
    # transform outputs
    assert X_transformed.shape == (5, 6)
    assert X_transformed["Name"].shape[0] == 5
    assert X_transformed.isna().sum().sum() == 0


def test_selelct_all_variables_with_na_when_variables_is_none(df_na):
    imputer = DropMissingData(missing_only=False, variables=None)
    X_transformed = imputer.fit_transform(df_na)
    assert imputer.input_shape_ == (8, 6)
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks", "dob"]
    assert X_transformed.shape == (5, 6)
    assert X_transformed[imputer.variables_].isna().sum().sum() == 0


def test_detect_variables_with_na_in_variables_entered_by_user(df_na):
    imputer = DropMissingData(
        missing_only=True, variables=["City", "Studies", "Age", "dob"]
    )
    X_transformed = imputer.fit_transform(df_na)
    assert imputer.variables == ["City", "Studies", "Age", "dob"]
    assert imputer.variables_ == ["City", "Studies", "Age"]
    assert X_transformed.shape == (6, 6)


def test_return_na_data_method(df_na):
    imputer = DropMissingData(
        missing_only=True, variables=["City", "Studies", "Age", "dob"]
    )
    imputer.fit(df_na)
    X_nona = imputer.return_na_data(df_na)
    assert X_nona.shape == (2, 6)


def test_error_when_missing_only_not_bool():
    with pytest.raises(ValueError):
        DropMissingData(missing_only="missing_only")


def test_non_fitted_error(df_na):
    with pytest.raises(NotFittedError):
        imputer = DropMissingData()
        imputer.transform(df_na)
