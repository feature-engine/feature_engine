import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.imputation import AddMissingIndicator


def test_detect_variables_with_missing_data_and_variables_is_none(df_na):
    # test case 1: automatically detect variables with missing data
    imputer = AddMissingIndicator(missing_only=True, variables=None)
    X_transformed = imputer.fit_transform(df_na)

    # init params
    assert imputer.missing_only is True
    assert imputer.variables is None
    # fit params
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.input_shape_ == (8, 6)
    # transform outputs
    assert X_transformed.shape == (8, 11)
    assert "Name_na" in X_transformed.columns
    assert X_transformed["Name_na"].sum() == 2


def test_add_indicators_to_all_variables_when_variables_is_none(df_na):
    imputer = AddMissingIndicator(missing_only=False, variables=None)
    X_transformed = imputer.fit_transform(df_na)
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks", "dob"]
    assert X_transformed.shape == (8, 12)
    assert "dob_na" in X_transformed.columns
    assert X_transformed["dob_na"].sum() == 0


def test_detect_variables_with_missing_data_in_variables_entered_by_user(df_na):
    imputer = AddMissingIndicator(
        missing_only=True, variables=["City", "Studies", "Age", "dob"]
    )
    X_transformed = imputer.fit_transform(df_na)
    assert imputer.variables == ["City", "Studies", "Age", "dob"]
    assert imputer.variables_ == ["City", "Studies", "Age"]
    assert X_transformed.shape == (8, 9)
    assert "City_na" in X_transformed.columns
    assert "dob_na" not in X_transformed.columns
    assert X_transformed["City_na"].sum() == 2


def test_error_when_missing_only_not_bool():
    with pytest.raises(ValueError):
        AddMissingIndicator(missing_only="missing_only")


def test_non_fitted_error(df_na):
    with pytest.raises(NotFittedError):
        imputer = AddMissingIndicator()
        imputer.transform(df_na)
