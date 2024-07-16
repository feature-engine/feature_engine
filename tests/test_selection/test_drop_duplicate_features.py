import numpy as np
import pandas as pd
import pytest

from feature_engine.selection import DropDuplicateFeatures


@pytest.fixture(scope="module")
def df_duplicate_features():
    data = {
        "Name": ["tom", "nick", "krish", "jack"],
        "dob2": pd.date_range("2020-02-24", periods=4, freq="min"),
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
        "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
        "City2": ["London", "Manchester", "Liverpool", "Bristol"],
        "dob3": pd.date_range("2020-02-24", periods=4, freq="min"),
        "Age2": [20, 21, 19, 18],
    }

    df = pd.DataFrame(data)

    return df


@pytest.fixture(scope="module")
def df_duplicate_features_with_na():
    data = {
        "Name": ["tom", "nick", "krish", "jack", np.nan],
        "dob2": pd.date_range("2020-02-24", periods=5, freq="min"),
        "City": ["London", "Manchester", "Liverpool", "Bristol", np.nan],
        "Age": [20, 21, np.nan, 18, 34],
        "Marks": [0.9, 0.8, 0.7, 0.6, 0.5],
        "dob": pd.date_range("2020-02-24", periods=5, freq="min"),
        "City2": ["London", "Manchester", "Liverpool", "Bristol", np.nan],
        "dob3": pd.date_range("2020-02-24", periods=5, freq="min"),
        "Age2": [20, 21, np.nan, 18, 34],
    }

    df = pd.DataFrame(data)

    return df


@pytest.fixture(scope="module")
def df_duplicate_features_with_different_data_types():
    data = {
        "A": pd.Series([5.5] * 3).astype("float64"),
        "B": 1,
        "C": "foo",
        "D": pd.Timestamp("20010102"),
        "E": pd.Series([1.0] * 3).astype("float32"),
        "F": False,
        "G": pd.Series([1] * 3, dtype="int8"),
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
            "dob2": pd.date_range("2020-02-24", periods=4, freq="min"),
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
        }
    )
    pd.testing.assert_frame_equal(X, df)


def test_fit_attributes(df_duplicate_features):
    transformer = DropDuplicateFeatures()
    transformer.fit(df_duplicate_features)

    assert transformer.features_to_drop_ == {"dob", "dob3", "City2", "Age2"}
    assert transformer.duplicated_feature_sets_ == [
        {"dob", "dob2", "dob3"},
        {"City", "City2"},
        {"Age", "Age2"},
    ]


def test_with_df_with_na(df_duplicate_features_with_na):
    transformer = DropDuplicateFeatures()
    X = transformer.fit_transform(df_duplicate_features_with_na)

    # expected result
    df = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack", np.nan],
            "dob2": pd.date_range("2020-02-24", periods=5, freq="min"),
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


def test_with_different_data_types(df_duplicate_features_with_different_data_types):
    transformer = DropDuplicateFeatures()
    X = transformer.fit_transform(df_duplicate_features_with_different_data_types)
    df = pd.DataFrame(
        {
            "A": pd.Series([5.5] * 3).astype("float64"),
            "B": 1,
            "C": "foo",
            "D": pd.Timestamp("20010102"),
            "F": False,
        }
    )
    pd.testing.assert_frame_equal(X, df)
