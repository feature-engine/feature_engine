import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from feature_engine.selection import DropDuplicateFeatures


def test_drops_duplicates_features(dataframe_duplicate_features):
    transformer = DropDuplicateFeatures()
    X = transformer.fit_transform(dataframe_duplicate_features)

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


def test_variables_assigned_correctly(dataframe_duplicate_features):
    transformer = DropDuplicateFeatures()
    assert transformer.variables is None

    transformer.fit(dataframe_duplicate_features)
    assert transformer.variables == (list(dataframe_duplicate_features.columns))


def test_fit_attributes(dataframe_duplicate_features):
    transformer = DropDuplicateFeatures()
    transformer.fit(dataframe_duplicate_features)

    assert transformer.duplicated_features_ == {"dob", "dob3", "City2", "Age2"}
    assert transformer.duplicated_feature_sets_ == [
        {"dob", "dob2", "dob3"},
        {"City", "City2"},
        {"Age", "Age2"},
    ]
    assert transformer.input_shape_ == (4, 9)


def test_with_df_with_na(dataframe_duplicate_features_with_na):
    transformer = DropDuplicateFeatures()
    X = transformer.fit_transform(dataframe_duplicate_features_with_na)

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

    assert transformer.duplicated_features_ == {"dob", "dob3", "City2", "Age2"}
    assert transformer.duplicated_feature_sets_ == [
        {"dob", "dob2", "dob3"},
        {"City", "City2"},
        {"Age", "Age2"},
    ]
    assert transformer.input_shape_ == (5, 9)


def test_raises_error__if_fit_input_not_dataframe():
    with pytest.raises(TypeError):
        DropDuplicateFeatures().fit({"Name": ["Karthik"]})


def test_raises_non_fitted_error(dataframe_duplicate_features):
    # test case 3: when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        transformer = DropDuplicateFeatures()
        transformer.transform(dataframe_duplicate_features)