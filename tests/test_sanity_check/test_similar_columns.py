import numpy as np
import pandas as pd
import pytest

from feature_engine.sanity_check import SimilarColumns


def test_similar_columns_when_more_columns_in_train_than_test(df_vartypes, df_na):

    # When columns are the same
    train = df_na.copy()
    test = df_vartypes.copy()

    similar_columns = SimilarColumns()
    similar_columns.fit(train)

    transformed_df = similar_columns.transform(test)

    expected_result = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Studies": [np.nan, np.nan, np.nan, np.nan],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
            "dob": pd.date_range("2020-02-24", periods=4, freq="T"),
        }
    )

    pd.testing.assert_frame_equal(expected_result, transformed_df)


def test_similar_columns_when_more_columns_in_test_than_train(df_vartypes, df_na):

    # When columns are the same
    train = df_vartypes
    test = df_na

    similar_columns = SimilarColumns()
    similar_columns.fit(train)

    transformed_df = similar_columns.transform(test)

    expected_result = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", np.nan, "peter", np.nan, "fred", "sam"],
            "City": [
                "London",
                "Manchester",
                np.nan,
                np.nan,
                "London",
                "London",
                "Bristol",
                "Manchester",
            ],
            "Age": [20, 21, 19, np.nan, 23, 40, 41, 37],
            "Marks": [0.9, 0.8, 0.7, np.nan, 0.3, np.nan, 0.8, 0.6],
            "dob": pd.date_range("2020-02-24", periods=8, freq="T"),
        }
    )

    pd.testing.assert_frame_equal(expected_result, transformed_df)


def test_similar_columns_raise_error(df_vartypes, df_na):

    # When columns are the same
    train = df_vartypes
    test = df_na

    similar_columns = SimilarColumns(drop_if_more_columns=False)
    similar_columns.fit(train)

    with pytest.raises(ValueError) as excinfo:
        similar_columns.transform(test)

    assert "'dob'" in str(excinfo.value) and "'Studies'" in str(excinfo.value)
