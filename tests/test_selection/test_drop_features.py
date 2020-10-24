import pandas as pd
import pytest

from feature_engine.selection import DropFeatures


def test_drop_2_variables(df_vartypes):
    transformer = DropFeatures(features_to_drop=["City", "dob"])
    X = transformer.fit_transform(df_vartypes)

    # expected result
    df = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
        }
    )

    # init params
    assert transformer.features_to_drop == ["City", "dob"]
    # transform params
    assert X.shape == (4, 3)
    assert type(X) == pd.DataFrame
    pd.testing.assert_frame_equal(X, df)


def test_error_if_non_existing_variables(df_vartypes):
    # test case 2: passing variables that doesn't exist
    with pytest.raises(KeyError):
        transformer = DropFeatures(features_to_drop="last_name")
        transformer.fit_transform(df_vartypes)


def test_error_if_fit_input_not_dataframe():
    # test case 3: passing a different input than dataframe
    with pytest.raises(TypeError):
        transformer = DropFeatures(features_to_drop=["Name"])
        transformer.fit({"Name": ["Karthik"]})


def test_error_when_returning_empty_dataframe(df_vartypes):
    # test case 5: dropping all columns produces warning check
    with pytest.raises(ValueError):
        transformer = DropFeatures(features_to_drop=list(df_vartypes.columns))
        transformer.fit_transform(df_vartypes)


def test_error_if_empty_list(df_vartypes):
    # test case 6: passing an empty list
    with pytest.raises(ValueError):
        transformer = DropFeatures(features_to_drop=[])
        transformer.fit_transform(df_vartypes)
