import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from feature_engine.discretisation import EqualFrequencyDiscretiser


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