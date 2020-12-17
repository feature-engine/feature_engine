import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.selection import DropDuplicateFeatures


@pytest.fixture(scope="module")
def df_duplicate_features():
    data = {
        "Name": ["tom", "nick", "krish", "jack"],
        "dob2": pd.date_range("2020-02-24", periods=4, freq="T"),
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
        "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
        "City2": ["London", "Manchester", "Liverpool", "Bristol"],
        "dob3": pd.date_range("2020-02-24", periods=4, freq="T"),
        "Age2": [20, 21, 19, 18],
    }

    df = pd.DataFrame(data)

    return df


@pytest.fixture(scope="module")
def df_duplicate_features_with_na():
    data = {
        "Name": ["tom", "nick", "krish", "jack", np.nan],
        "dob2": pd.date_range("2020-02-24", periods=5, freq="T"),
        "City": ["London", "Manchester", "Liverpool", "Bristol", np.nan],
        "Age": [20, 21, np.nan, 18, 34],
        "Marks": [0.9, 0.8, 0.7, 0.6, 0.5],
        "dob": pd.date_range("2020-02-24", periods=5, freq="T"),
        "City2": ["London", "Manchester", "Liverpool", "Bristol", np.nan],
        "dob3": pd.date_range("2020-02-24", periods=5, freq="T"),
        "Age2": [20, 21, np.nan, 18, 34],
    }

    df = pd.DataFrame(data)

    return df


def test_drop_duplicates_features(df_duplicate_features):
    transformer = DropDuplicateFeatures()
    X = transformer.fit_transform(df_duplicate_features)

    # expected result
    df = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "dob2": pd.date_range("2020-02-24", periods=4, freq="T"),
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
        }
    )
    pd.testing.assert_frame_equal(X, df)


def test_variables_assigned_correctly(df_duplicate_features):
    transformer = DropDuplicateFeatures()
    assert transformer.variables is None

    transformer.fit(df_duplicate_features)
    assert transformer.variables == (list(df_duplicate_features.columns))


def test_fit_attributes(df_duplicate_features):
    transformer = DropDuplicateFeatures()
    transformer.fit(df_duplicate_features)

    assert transformer.features_to_drop_ == {"dob", "dob3", "City2", "Age2"}
    assert transformer.duplicated_feature_sets_ == [
        {"dob", "dob2", "dob3"},
        {"City", "City2"},
        {"Age", "Age2"},
    ]
    assert transformer.input_shape_ == (4, 9)


def test_with_df_with_na(df_duplicate_features_with_na):
    transformer = DropDuplicateFeatures()
    X = transformer.fit_transform(df_duplicate_features_with_na)

    # expected result
    df = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack", np.nan],
            "dob2": pd.date_range("2020-02-24", periods=5, freq="T"),
            "City": ["London", "Manchester", "Liverpool", "Bristol", np.nan],
            "Age": [20, 21, np.nan, 18, 34],
            "Marks": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )
    pd.testing.assert_frame_equal(X, df)

    assert transformer.features_to_drop_ == {"dob", "dob3", "City2", "Age2"}
    assert transformer.duplicated_feature_sets_ == [
        {"dob", "dob2", "dob3"},
        {"City", "City2"},
        {"Age", "Age2"},
    ]
    assert transformer.input_shape_ == (5, 9)


def test_error_if_fit_input_not_dataframe():
    with pytest.raises(TypeError):
        DropDuplicateFeatures().fit({"Name": ["Karthik"]})


def test_non_fitted_error(df_duplicate_features):
    # test case 3: when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        transformer = DropDuplicateFeatures()
        transformer.transform(df_duplicate_features)
