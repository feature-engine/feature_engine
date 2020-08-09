# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause
import pytest
import pandas as pd

from feature_engine.feature_selection import DropFeatures


def test_FeatureEliminator_initial(dataframe_vartypes, dataframe_na):
    # test case 1: with dataframe_vartypes
    transformer = DropFeatures(features_to_drop=["City", "dob"])
    X = transformer.fit_transform(dataframe_vartypes)

    # expected result
    df = pd.DataFrame(
        {'Name': ['tom', 'nick', 'krish', 'jack'],
         'Age': [20, 21, 19, 18],
         'Marks': [0.9, 0.8, 0.7, 0.6]}
    )

    # init params
    assert transformer.features == ['City', 'dob']
    # transform params
    assert X.shape == (4, 3)
    assert type(X) == pd.DataFrame
    pd.testing.assert_frame_equal(X, df)


def test_FeatureEliminator_non_existing_variables(dataframe_vartypes):
    # test case 2: passing variables that doesn't exist
    with pytest.raises(KeyError):
        transformer = DropFeatures(features_to_drop="last_name")
        X = transformer.fit_transform(dataframe_vartypes)


def test_FeatureEliminator_non_existing_index(dataframe_vartypes):
    # test case 3: passing an index inside the list
    with pytest.raises(KeyError):
        transformer = DropFeatures(features_to_drop=[10])
        X = transformer.fit_transform(dataframe_vartypes)


def test_FeatureEliminator_integer_as_variable(dataframe_vartypes):
    # test case 4: with integer as feature_to_drop
    with pytest.raises(KeyError):
        transformer = DropFeatures(features_to_drop=99)
        X = transformer.fit_transform(dataframe_vartypes)


def test_FeatureEliminator_different_input():
    # test case 5: passing a different input than dataframe
    with pytest.raises(TypeError):
        transformer = DropFeatures(features_to_drop=["Name"])
        X = transformer.fit_transform({"Name": ["Karthik"]})


def test_FeatureEliminator_drop_all_columns(dataframe_vartypes):
    # test case 6: dropping all columns
    transformer = DropFeatures(features_to_drop=list(dataframe_vartypes.columns))
    X = transformer.fit_transform(dataframe_vartypes)

    assert len(X) == len(dataframe_vartypes)
    assert X.shape == (4, 0)


def test_FeatureEliminator_drop_all_columns_warn(dataframe_vartypes):
    # test case 7: dropping all columns produces warning check
    with pytest.warns(UserWarning):
        transformer = DropFeatures(features_to_drop=list(dataframe_vartypes.columns))
        X = transformer.fit_transform(dataframe_vartypes)


def test_FeatureEliminator_empty_list(dataframe_vartypes):
    # test case 8: passing an empty list
    transformer = DropFeatures(features_to_drop=[])
    X = transformer.fit_transform(dataframe_vartypes)

    assert len(X) == len(dataframe_vartypes)
    assert X.shape == dataframe_vartypes.shape


def test_FeatureEliminator_valid_string(dataframe_vartypes):
    # test case 9: passing a valid variable as string
    transformer = DropFeatures(features_to_drop="dob")
    X = transformer.fit_transform(dataframe_vartypes)

    df = pd.DataFrame(
        {'Name': ['tom', 'nick', 'krish', 'jack'],
         'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
         'Age': [20, 21, 19, 18],
         'Marks': [0.9, 0.8, 0.7, 0.6]
         }
    )

    # init params
    transformer.features = ["dob"]
    # transform params
    assert X.shape == (4, 4)
    assert type(X) == pd.DataFrame
    pd.testing.assert_frame_equal(X, df)

