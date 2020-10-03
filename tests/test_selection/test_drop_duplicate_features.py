import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from feature_engine.selection import DropDuplicateFeatures


# DropDuplicateFeatures transformer tests
def test_drop_duplicate_features(dataframe_duplicate_features):
    # test 1: drop duplicate features
    transformer = DropDuplicateFeatures()
    X = transformer.fit_transform(dataframe_duplicate_features)

    # expected result
    df = pd.DataFrame(
        {'Name': ['tom', 'nick', 'krish', 'jack'],
         'dob2': pd.date_range('2020-02-24', periods=4, freq='T'),
         'City': ['London', 'Manchester', 'Liverpool', 'Bristol'],
         'Age': [20, 21, 19, 18],
         'Marks': [0.9, 0.8, 0.7, 0.6]}
    )

    # fit attributes
    assert transformer.duplicate_feature_dict_ == {'Name': [],
                                                   'dob2': ['dob', 'dob3'],
                                                   'City': ['City2'],
                                                   'Age': [],
                                                   'Marks': []
                                                   }
    assert transformer.input_shape_ == (4, 8)

    # transform output
    pd.testing.assert_frame_equal(X, df)


def test_drop_duplicate_features_if_fit_input_not_dataframe():
    # test case 2: input is not a dataframe
    with pytest.raises(TypeError):
        DropDuplicateFeatures().fit({"Name": ["Karthik"]})


def test_drop_duplicate_features_if_fit_not_called(dataframe_duplicate_features):
    # test case 3: when fit is not called prior to transform
    with pytest.raises(NotFittedError):
        transformer = DropDuplicateFeatures()
        transformer.transform(dataframe_duplicate_features)



