import pandas as pd
import pytest

from feature_engine.feature_selection import DropFeatures, SelectFeatures


def test_drop_features_drop_2_variables(dataframe_vartypes):

    transformer = DropFeatures(features_to_drop=["City", "dob"])
    X = transformer.fit_transform(dataframe_vartypes)

    # expected result
    df = pd.DataFrame(
        {'Name': ['tom', 'nick', 'krish', 'jack'],
         'Age': [20, 21, 19, 18],
         'Marks': [0.9, 0.8, 0.7, 0.6]}
    )

    # init params
    assert transformer.features_to_drop == ['City', 'dob']
    # transform params
    assert X.shape == (4, 3)
    assert type(X) == pd.DataFrame
    pd.testing.assert_frame_equal(X, df)


def test_drop_features_non_existing_variables(dataframe_vartypes):
    # test case 2: passing variables that doesn't exist
    with pytest.raises(KeyError):
        transformer = DropFeatures(features_to_drop="last_name")
        X = transformer.fit_transform(dataframe_vartypes)


def test_drop_features_errors_if_fit_input_not_dataframe():
    # test case 3: passing a different input than dataframe
    with pytest.raises(TypeError):
        transformer = DropFeatures(features_to_drop=["Name"])
        X = transformer.fit({"Name": ["Karthik"]})


def test_drop_features_raises_error_when_returning_empty_dataframe(dataframe_vartypes):
    # test case 5: dropping all columns produces warning check
    with pytest.raises(ValueError):
        transformer = DropFeatures(features_to_drop=list(dataframe_vartypes.columns))
        X = transformer.fit_transform(dataframe_vartypes)


def test_drop_features_empty_list(dataframe_vartypes):
    # test case 6: passing an empty list
    with pytest.raises(ValueError):
        transformer = DropFeatures(features_to_drop=[])
        transformer.fit_transform(dataframe_vartypes)


# SelectFeatures transformer tests
def test_select_features_drop_constant_features(dataframe_con_quasi_con):
    # test case 1: only drop constant features
    transformer = SelectFeatures(drop_quasi_constant_features=False, quasi_constant_threshold=None)

    X = transformer.fit_transform(dataframe_con_quasi_con)

    # expected result
    df = pd.DataFrame(
        {'Name': ['tom', 'nick', 'krish', 'jack'],
         'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
         'Age': [20, 21, 19, 18],
         'Marks': [0.9, 0.8, 0.7, 0.6],
         'dob': pd.date_range('2020-02-24', periods=4, freq='T'),
         'quasi_feat_num': [1, 1, 1, 2],
         'quasi_feat_cat': ['a', 'a', 'a', 'b']}
    )

    # init params
    assert transformer.drop_quasi_constant_features is False
    assert transformer.quasi_constant_threshold is None
    # transform params
    assert X.shape == (4, 7)
    assert type(X) == pd.DataFrame
    pd.testing.assert_frame_equal(X, df)


def test_select_features_drop_quasi_constant_features(dataframe_con_quasi_con):
    # test case 2: drop quasi constant features (also by default drops constant features)
    transformer = SelectFeatures(drop_quasi_constant_features=True, quasi_constant_threshold=0.7)

    X = transformer.fit_transform(dataframe_con_quasi_con)

    # expected result
    df = pd.DataFrame(
        {'Name': ['tom', 'nick', 'krish', 'jack'],
         'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
         'Age': [20, 21, 19, 18],
         'Marks': [0.9, 0.8, 0.7, 0.6],
         'dob': pd.date_range('2020-02-24', periods=4, freq='T'),
         }
    )

    # init params
    assert transformer.drop_quasi_constant_features is True
    assert transformer.quasi_constant_threshold == 0.7
    # transform params
    assert X.shape == (4, 5)
    assert type(X) == pd.DataFrame
    pd.testing.assert_frame_equal(X, df)


def test_select_features_input_not_dataframe():
    # test case 3: input is not a dataframe
    with pytest.raises(TypeError):
        SelectFeatures().fit({"Name": ["Karthik"]})


def test_select_features_threshold_out_of_range():
    # test case 4: threshold more than 1 or less than 0
    with pytest.raises(ValueError):
        SelectFeatures(drop_quasi_constant_features=True, quasi_constant_threshold=2)

    with pytest.raises(ValueError):
        SelectFeatures(drop_quasi_constant_features=True, quasi_constant_threshold=-1)

    with pytest.raises(ValueError):
        SelectFeatures(drop_quasi_constant_features=True, quasi_constant_threshold=1)


def test_select_features_all_constant_features():
    # test case 5: when input contains all constant features
    with pytest.raises(ValueError):
        SelectFeatures().fit(pd.DataFrame({'col1': [1, 1, 1], 'col2': [1, 1, 1]}))


def test_select_features_all_constant_and_quasi_constant_features():
    # test case 6: when input contains all constant and quasi constant features
    with pytest.raises(ValueError):
        transformer = SelectFeatures(drop_quasi_constant_features=True, quasi_constant_threshold=0.7)
        transformer.fit_transform(pd.DataFrame({'col1': [1, 1, 1, 1], 'col2': [1, 1, 1, 1],
                                                'col3': [1, 1, 1, 2], 'col4': [1, 1, 1, 2]
                                                }))

