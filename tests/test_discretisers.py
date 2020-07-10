import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.datasets import load_boston

from feature_engine.discretisers import (
    EqualFrequencyDiscretiser,
    EqualWidthDiscretiser,
    DecisionTreeDiscretiser,
    UserInputDiscretiser
)


def test_EqualFrequencyDiscretiser(dataframe_normal_dist, dataframe_vartypes, dataframe_na):
    # test case 1: automatically select variables, return_object=False
    transformer = EqualFrequencyDiscretiser(q=10, variables=None, return_object=False)
    X = transformer.fit_transform(dataframe_normal_dist)

    # fit parameters
    _, bins = pd.qcut(x=dataframe_normal_dist['var'], q=10, retbins=True, duplicates='drop')
    bins[0] = float("-inf")
    bins[len(bins) - 1] = float("inf")

    # transform output
    X_t = [x for x in range(0, 10)]

    # init params
    assert transformer.q == 10
    assert transformer.variables == ['var']
    assert transformer.return_object is False
    # fit params
    assert transformer.input_shape_ == (100, 1)
    # transform params
    assert (transformer.binner_dict_['var'] == bins).all()
    assert len([x for x in X['var'].unique() if x not in X_t]) == 0
    # in equal frequency discretisation, all intervals get same proportion of values
    assert len((X['var'].value_counts()).unique()) == 1

    # test case 2: return variables cast as object
    transformer = EqualFrequencyDiscretiser(q=10, variables=None, return_object=True)
    X = transformer.fit_transform(dataframe_normal_dist)
    assert X['var'].dtypes == 'O'

    with pytest.raises(ValueError):
        EqualFrequencyDiscretiser(q='other')

    with pytest.raises(ValueError):
        EqualFrequencyDiscretiser(return_object='other')

    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = EqualFrequencyDiscretiser()
        transformer.fit(dataframe_na)

    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = EqualFrequencyDiscretiser()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])

    with pytest.raises(NotFittedError):
        transformer = EqualFrequencyDiscretiser()
        transformer.transform(dataframe_vartypes)


def test_EqualWidthDiscretiser(dataframe_normal_dist, dataframe_vartypes, dataframe_na):
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

    # test case 2: return variables cast as object
    transformer = EqualWidthDiscretiser(bins=10, variables=None, return_object=True)
    X = transformer.fit_transform(dataframe_normal_dist)
    assert X['var'].dtypes == 'O'

    with pytest.raises(ValueError):
        EqualWidthDiscretiser(bins='other')

    with pytest.raises(ValueError):
        EqualWidthDiscretiser(return_object='other')

    # test case 3: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = EqualWidthDiscretiser()
        transformer.fit(dataframe_na)

    # test case 4: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = EqualWidthDiscretiser()
        transformer.fit(dataframe_vartypes)
        transformer.transform(dataframe_na[['Name', 'City', 'Age', 'Marks', 'dob']])

    with pytest.raises(NotFittedError):
        transformer = EqualWidthDiscretiser()
        transformer.transform(dataframe_vartypes)


def test_DecisionTreeDiscretiser(dataframe_normal_dist, dataframe_vartypes, dataframe_na):
    # test case 1: classification
    transformer = DecisionTreeDiscretiser(cv=3, scoring='roc_auc', variables=None,
                                          param_grid={'max_depth': [1, 2, 3, 4]},
                                          regression=False, random_state=0)
    np.random.seed(0)
    y = pd.Series(np.random.binomial(1, 0.7, 100))
    X = transformer.fit_transform(dataframe_normal_dist, y)
    X_t = [1., 0.71, 0.93, 0.]

    # init params
    assert transformer.cv == 3
    assert transformer.variables == ['var']
    assert transformer.scoring == 'roc_auc'
    assert transformer.regression is False
    # fit params
    assert transformer.input_shape_ == (100, 1)
    # transform params
    assert len([x for x in np.round(X['var'].unique(), 2) if x not in X_t]) == 0
    assert transformer.scores_dict_ == {'var': 0.717391304347826}

    # test case 2: regression
    transformer = DecisionTreeDiscretiser(cv=3, scoring='neg_mean_squared_error', variables=None,
                                          param_grid={'max_depth': [1, 2, 3, 4]}, regression=True,
                                          random_state=0)
    np.random.seed(0)
    y = pd.Series(pd.Series(np.random.normal(0, 0.1, 100)))
    X = transformer.fit_transform(dataframe_normal_dist, y)
    X_t = [0.19, 0.04, 0.11, 0.23, -0.09, -0.02, 0.01, 0.15, 0.07,
           -0.26, 0.09, -0.07, -0.16, -0.2, -0.04, -0.12]

    # init params
    assert transformer.cv == 3
    assert transformer.variables == ['var']
    assert transformer.scoring == 'neg_mean_squared_error'
    assert transformer.regression is True
    # fit params
    assert transformer.input_shape_ == (100, 1)
    assert transformer.scores_dict_ == {'var': -4.4373314584616444e-05}
    # transform params
    assert len([x for x in np.round(X['var'].unique(), 2) if x not in X_t]) == 0

    with pytest.raises(ValueError):
        DecisionTreeDiscretiser(cv='other')

    with pytest.raises(ValueError):
        DecisionTreeDiscretiser(regression='other')

    # test case 3: raises error if target is not passed
    with pytest.raises(TypeError):
        encoder = DecisionTreeDiscretiser()
        encoder.fit(dataframe_normal_dist)

    with pytest.raises(NotFittedError):
        transformer = EqualWidthDiscretiser()
        transformer.transform(dataframe_vartypes)


def test_UserInputDiscretise():
    boston_dataset = load_boston()
    data = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    user_dict = {'LSTAT': [0, 10, 20, 30, np.Inf]}

    data_t1 = data.copy()
    data_t2 = data.copy()
    data_t1['LSTAT'] = pd.cut(data['LSTAT'], bins=[0, 10, 20, 30, np.Inf])
    data_t2['LSTAT'] = pd.cut(data['LSTAT'], bins=[0, 10, 20, 30, np.Inf], labels=False)

    transformer = UserInputDiscretiser(binning_dict=user_dict, return_object=False, return_boundaries=False)
    X = transformer.fit_transform(data)

    # init params
    assert transformer.variables == ['LSTAT']
    assert transformer.return_object is False
    assert transformer.return_boundaries is False
    # fit params
    assert transformer.binner_dict_ == user_dict
    # transform params
    pd.testing.assert_frame_equal(X, data_t2)

    transformer = UserInputDiscretiser(binning_dict=user_dict, return_object=False, return_boundaries=True)
    X = transformer.fit_transform(data)
    pd.testing.assert_frame_equal(X, data_t1)