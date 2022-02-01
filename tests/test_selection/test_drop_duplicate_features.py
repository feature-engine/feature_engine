import numpy as np
import pandas as pd
import pytest

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


"Name", "dob2", "City", "Age", "Marks", "dob", "City2", "dob3", "Age2"

input_params = [
    (
        ["Name", "dob2", "City", "Age", "new_feat"],
        ["Name", "dob2", "City", "Age"],
    ),
    (
        None,
        ["Name", "dob2", "City", "Age", "Marks", "dob", "City2", "dob3", "Age2"],
    ),
    ("dob2", ["dob2"]),
]


@pytest.mark.parametrize("variables, expected", input_params)
def test_confirm_variables_argument(df_duplicate_features, variables, expected):
    """Test the confirm_variable argument."""
    # Test confirm_variables set to True allows to deal with elements of
    # variables that are not in the dataframe to fit.

    transformer = DropDuplicateFeatures(variables=variables, confirm_variables=True)
    transformer.fit(df_duplicate_features)

    assert transformer.variables_ == expected


# TODO: DropDuplicates does not fail when providing a single string that is not part
# of the dataframe. Confirm we want to keep this behavior or not.


def test_confirm_variables_argument_false(df_duplicate_features):
    """Test the confirm_variable argument when set to False."""
    # Test the default value gives an error when an element of variables is
    # not present in the dataframe to fit.
    variables = ["Name", "dob2", "City", "Age", "new_feat"]

    with pytest.raises(KeyError):
        assert DropDuplicateFeatures(variables=variables, confirm_variables=False).fit(
            df_duplicate_features
        )
