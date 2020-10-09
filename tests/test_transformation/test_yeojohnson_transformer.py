import pytest
import pandas as pd
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import YeoJohnsonTransformer


def test_automatically_select_variables(dataframe_vartypes):
    # test case 1: automatically select variables
    transformer = YeoJohnsonTransformer(variables=None)
    X = transformer.fit_transform(dataframe_vartypes)

    # expected result
    transf_df = dataframe_vartypes.copy()
    transf_df['Age'] = [10.167, 10.5406, 9.78774, 9.40229]
    transf_df['Marks'] = [0.804449, 0.722367, 0.638807, 0.553652]

    # test init params
    assert transformer.variables == ['Age', 'Marks']
    # test fit attr
    assert transformer.input_shape_ == (4, 5)
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)


def test_fit_raises_error_if_na_in_df(dataframe_na):
    # test case 2: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = YeoJohnsonTransformer()
        transformer.fit(dataframe_na)


def test_transform_raises_error_if_na_in_df(dataframe_vartypes, dataframe_na):
    # test case 3: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = YeoJohnsonTransformer()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])


def test_raises_non_fitted_error(dataframe_vartypes):
    with pytest.raises(NotFittedError):
        transformer = YeoJohnsonTransformer()
        transformer.transform(dataframe_vartypes)
