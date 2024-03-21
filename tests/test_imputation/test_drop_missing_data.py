import numpy as np
import pandas as pd
import pytest

from feature_engine.imputation import DropMissingData


def test_detect_variables_with_na(df_na):
    # test case 1: automatically detect variables with missing data
    imputer = DropMissingData(missing_only=True, variables=None)
    X_transformed = imputer.fit_transform(df_na)
    # init params
    assert imputer.missing_only is True
    assert imputer.threshold is None
    assert imputer.variables is None
    # fit params
    assert imputer.variables_ == ["Name", "City", "Studies", "Age", "Marks"]
    assert imputer.n_features_in_ == 6
    # transform outputs
    assert X_transformed.shape == (5, 6)
    assert X_transformed["Name"].shape[0] == 5
    assert X_transformed.isna().sum().sum() == 0


def test_transform_x_y(df_na):
    y = pd.Series(np.zeros(len(df_na)))
    imputer = DropMissingData(missing_only=True, variables=None)
    X_transformed = imputer.fit_transform(df_na)
    # transform outputs
    assert X_transformed.shape == (5, 6)
    assert X_transformed.isna().sum().sum() == 0
    assert len(X_transformed) != len(y)

    Xt, yt = imputer.transform_x_y(df_na, y)
    assert len(Xt) == len(yt)
    assert (Xt.index == yt.index).all()
    assert len(df_na) != len(Xt)


def test_selelct_all_variables_when_variables_is_none(df_na):
    imputer = DropMissingData(missing_only=False, variables=None)
    X_transformed = imputer.fit_transform(df_na)
    assert imputer.n_features_in_ == 6
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

    # test with vars
    imputer = DropMissingData(
        threshold=0.5, variables=["City", "Studies", "Age", "Marks"]
    )
    imputer.fit_transform(df_na)
    X_nona = imputer.return_na_data(df_na)
    assert list(X_nona.index) == [2, 3]

    # test without vars & threshold
    imputer = DropMissingData()
    imputer.fit_transform(df_na)
    X_nona = imputer.return_na_data(df_na)
    assert list(X_nona.index) == [2, 3, 5]


def test_error_when_missing_only_not_bool():
    with pytest.raises(ValueError):
        DropMissingData(missing_only="missing_only")


def test_threshold(df_na):

    # Each row must have 100% data available
    imputer = DropMissingData(threshold=1)
    X = imputer.fit_transform(df_na)
    assert list(X.index) == [0, 1, 4, 6, 7]

    # Each row must have at least 1% data available
    imputer = DropMissingData(threshold=0.01)
    X = imputer.fit_transform(df_na)
    assert list(X.index) == [0, 1, 2, 3, 4, 5, 6, 7]

    # Each row must have at least 50% data available
    imputer = DropMissingData(threshold=0.50)
    X = imputer.fit_transform(df_na)
    assert list(X.index) == [0, 1, 2, 4, 5, 6, 7]

    # Each row must have 100% data available
    imputer = DropMissingData(threshold=1, missing_only=False)
    X = imputer.fit_transform(df_na)
    assert list(X.index) == [0, 1, 4, 6, 7]

    # Each row must have at least 1% data available
    imputer = DropMissingData(threshold=0.01, missing_only=False)
    X = imputer.fit_transform(df_na)
    assert list(X.index) == [0, 1, 2, 3, 4, 5, 6, 7]

    # Each row must have at least 50% data available
    imputer = DropMissingData(threshold=0.50, missing_only=False)
    X = imputer.fit_transform(df_na)
    assert list(X.index) == [0, 1, 2, 4, 5, 6, 7]


def test_threshold_value_error(df_na):
    with pytest.raises(ValueError):
        DropMissingData(threshold=1.01)

    with pytest.raises(ValueError):
        DropMissingData(threshold=-0.01)

    with pytest.raises(ValueError):
        DropMissingData(threshold=0)


def test_threshold_with_variables(df_na):

    # Each row must have 100% data avaiable for columns ['Marks']
    imputer = DropMissingData(threshold=1, variables=["Marks"])
    X = imputer.fit_transform(df_na)
    assert list(X.index) == [0, 1, 2, 4, 6, 7]

    # Each row must have 25% data avaiable for ['City', 'Studies', 'Age', 'Marks']
    imputer = DropMissingData(
        threshold=0.75, variables=["City", "Studies", "Age", "Marks"]
    )
    X = imputer.fit_transform(df_na)
    assert list(X.index) == [0, 1, 4, 5, 6, 7]
