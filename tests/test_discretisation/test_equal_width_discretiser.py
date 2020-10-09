import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from feature_engine.discretisation import EqualWidthDiscretiser


def test_automatically_find_variables_and_return_as_numeric(dataframe_normal_dist):
    # test case 1: automatically select variables, return_object=False
    transformer = EqualWidthDiscretiser(bins=10, variables=None, return_object=False)
    X = transformer.fit_transform(dataframe_normal_dist)

    # fit parameters
    _, bins = pd.cut(x=dataframe_normal_dist['var'], bins=10, retbins=True, duplicates='drop')
    bins[0] = float("-inf")
    bins[len(bins) - 1] = float("inf")

    # transform output
    X_t = [x for x in range(0, 10)]
    val_counts = [18, 17, 16, 13, 11, 7, 7, 5, 5, 1]

    # init params
    assert transformer.bins == 10
    assert transformer.variables == ['var']
    assert transformer.return_object is False
    # fit params
    assert transformer.input_shape_ == (100, 1)
    # transform params
    assert (transformer.binner_dict_['var'] == bins).all()
    assert len([x for x in X['var'].unique() if x not in X_t]) == 0
    # in equal width discretisation, intervals get different number of values
    assert len([x for x in X['var'].value_counts() if x not in val_counts]) == 0


def test_automatically_find_variables_and_return_as_object(dataframe_normal_dist):
    transformer = EqualWidthDiscretiser(bins=10, variables=None, return_object=True)
    X = transformer.fit_transform(dataframe_normal_dist)
    assert X['var'].dtypes == 'O'


def test_raises_error_when_bins_not_number():
    with pytest.raises(ValueError):
        EqualWidthDiscretiser(bins='other')


def test_raises_error_if_return_object_not_bool():
    with pytest.raises(ValueError):
        EqualWidthDiscretiser(return_object='other')


def test_fit_raises_error_if_input_df_contains_na(dataframe_na):
    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = EqualWidthDiscretiser()
        transformer.fit(dataframe_na)


def test_transform_raises_error_if_input_df_contains_na(dataframe_vartypes, dataframe_na):
    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = EqualWidthDiscretiser()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])


def test_raises_non_fitted_error(dataframe_vartypes):
    with pytest.raises(NotFittedError):
        transformer = EqualWidthDiscretiser()
        transformer.transform(dataframe_vartypes)