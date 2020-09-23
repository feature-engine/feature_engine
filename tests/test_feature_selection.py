import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from feature_engine.feature_selection import DropFeatures, DropConstantAndQuasiConstantFeatures


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


# DropConstantAndQuasiConstantFeatures transformer tests
def test_drop_constant_and_quasi_constant_features(dataframe_constant_quasi_constant):
    # test case 1: only drop constant features
    transformer = DropConstantAndQuasiConstantFeatures(tol=1, variables=None)
    X = transformer.fit_transform(dataframe_constant_quasi_constant)

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
    assert transformer.tol == 1
    assert transformer.variables == ['Name', 'City', 'Age', 'Marks', 'dob', 'const_feat_num',
                                     'const_feat_cat', 'quasi_feat_num', 'quasi_feat_cat']
    # transform params
    assert X.shape == (4, 7)
    assert type(X) == pd.DataFrame
    pd.testing.assert_frame_equal(X, df)

    # test case 2: drop features showing threshold more than 0.7
    transformer = DropConstantAndQuasiConstantFeatures(tol=0.7, variables=None)
    X = transformer.fit_transform(dataframe_constant_quasi_constant)

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
    assert transformer.tol == 0.7
    assert transformer.variables == ['Name', 'City', 'Age', 'Marks', 'dob', 'const_feat_num',
                                     'const_feat_cat', 'quasi_feat_num', 'quasi_feat_cat']
    # transform params
    assert X.shape == (4, 5)
    assert type(X) == pd.DataFrame
    pd.testing.assert_frame_equal(X, df)

    # test case 3: drop features showing threshold more than 0.7 with variable list
    transformer = DropConstantAndQuasiConstantFeatures(tol=0.7, variables=['Name', 'const_feat_num', 'quasi_feat_num'])
    X = transformer.fit_transform(dataframe_constant_quasi_constant)

    # expected result
    df = pd.DataFrame(
        {'Name': ['tom', 'nick', 'krish', 'jack'],
         'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
         'Age': [20, 21, 19, 18],
         'Marks': [0.9, 0.8, 0.7, 0.6],
         'dob': pd.date_range('2020-02-24', periods=4, freq='T'),
         'const_feat_cat': ['a', 'a', 'a', 'a'],
         'quasi_feat_cat': ['a', 'a', 'a', 'b']
         }
    )

    # init params
    assert transformer.tol == 0.7
    assert transformer.variables == ['Name', 'const_feat_num', 'quasi_feat_num']
    # transform params
    assert X.shape == (4, 7)
    assert type(X) == pd.DataFrame
    pd.testing.assert_frame_equal(X, df)

    # test case 4: input is not a dataframe
    with pytest.raises(TypeError):
        DropConstantAndQuasiConstantFeatures().fit({"Name": ["Karthik"]})

    # test case 5: threshold not between 0 and 1
    with pytest.raises(ValueError):
        transformer = DropConstantAndQuasiConstantFeatures(tol=2)

    # test case 6: when input contains all constant features
    with pytest.raises(ValueError):
        DropConstantAndQuasiConstantFeatures().fit(pd.DataFrame({'col1': [1, 1, 1], 'col2': [1, 1, 1]}))

    # test case 7: when input contains all constant and quasi constant features
    with pytest.raises(ValueError):
        transformer = DropConstantAndQuasiConstantFeatures(tol=0.7)
        transformer.fit_transform(pd.DataFrame({'col1': [1, 1, 1, 1], 'col2': [1, 1, 1, 1],
                                                'col3': [1, 1, 1, 2], 'col4': [1, 1, 1, 2]
                                                }))

    # test case 8: when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        transformer = DropConstantAndQuasiConstantFeatures()
        transformer.transform(dataframe_constant_quasi_constant)


