# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import pytest
import numpy as np
import pandas as pd

from feature_engine.imputation.random_sample import _define_seed
from feature_engine.imputation import RandomSampleImputer


def test_define_seed(dataframe_vartypes):
    assert _define_seed(dataframe_vartypes, 0, ['Age', 'Marks'], how='add') == 21
    assert _define_seed(dataframe_vartypes, 0, ['Age', 'Marks'], how='multiply') == 18
    assert _define_seed(dataframe_vartypes, 2, ['Age', 'Marks'], how='add') == 20
    assert _define_seed(dataframe_vartypes, 2, ['Age', 'Marks'], how='multiply') == 13
    assert _define_seed(dataframe_vartypes, 1, ['Age'], how='add') == 21
    assert _define_seed(dataframe_vartypes, 3, ['Marks'], how='multiply') == 1


def test_RandomSampleImputer(dataframe_na):

    # test case 1: imputer with general seed + automatically select variables
    imputer = RandomSampleImputer(variables=None, random_state=5, seed='general')
    X_transformed = imputer.fit_transform(dataframe_na)

    # fillna based on seed used (found experimenting on Jupyter notebook)
    ref = {'Name': ['tom', 'nick', 'krish', 'peter', 'peter', 'sam', 'fred', 'sam'],
           'City': ['London', 'Manchester', 'London', 'Manchester', 'London', 'London', 'Bristol', 'Manchester'],
           'Studies': ['Bachelor', 'Bachelor', 'PhD', 'Masters', 'Bachelor', 'PhD', 'None', 'Masters'],
           'Age': [20, 21, 19, 23, 23, 40, 41, 37],
           'Marks': [0.9, 0.8, 0.7, 0.3, 0.3, 0.6, 0.8, 0.6],
           'dob': pd.date_range('2020-02-24', periods=8, freq='T')
           }
    ref = pd.DataFrame(ref)
    # init params
    assert imputer.variables == ['Name', 'City', 'Studies', 'Age', 'Marks', 'dob']
    assert imputer.random_state == 5
    assert imputer.seed == 'general'
    # fit params
    assert imputer.input_shape_ == (8, 6)
    pd.testing.assert_frame_equal(imputer.X_, dataframe_na)
    # transform output
    pd.testing.assert_frame_equal(X_transformed, ref, check_dtype=False)

    # test case 2: imputer sed per observation using multiple variables to determine the random_state
    # Note the variables used as seed should not have missing data
    imputer = RandomSampleImputer(variables=['City', 'Studies'], random_state=['Marks', 'Age'], seed='observation')
    dataframe_na[['Marks', 'Age']] = dataframe_na[['Marks', 'Age']].fillna(1)
    X_transformed = imputer.fit_transform(dataframe_na)

    # how the imputed dataframe should look like
    ref = {'Name': ['tom', 'nick', 'krish', np.nan, 'peter', np.nan, 'fred', 'sam'],
           'City': ['London', 'Manchester', 'London', 'London', 'London', 'London', 'Bristol', 'Manchester'],
           'Studies': ['Bachelor', 'Bachelor', 'PhD', 'Bachelor', 'Bachelor', 'PhD', 'None', 'Masters'],
           'Age': [20, 21, 19, np.nan, 23, 40, 41, 37],
           'Marks': [0.9, 0.8, 0.7, np.nan, 0.3, np.nan, 0.8, 0.6],
           'dob': pd.date_range('2020-02-24', periods=8, freq='T')
           }
    ref = pd.DataFrame(ref)

    assert imputer.variables == ['City', 'Studies']
    assert imputer.random_state == ['Marks', 'Age']
    assert imputer.seed == 'observation'
    pd.testing.assert_frame_equal(imputer.X_[['City', 'Studies']], dataframe_na[['City', 'Studies']])
    pd.testing.assert_frame_equal(X_transformed[['City', 'Studies']], ref[['City', 'Studies']])

    # test case 3: observation seed, 2 variables as seed, product of seed variables
    imputer = RandomSampleImputer(variables=['City', 'Studies'], random_state=['Marks', 'Age'], seed='observation',
                                  seeding_method='multiply')
    dataframe_na[['Marks', 'Age']] = dataframe_na[['Marks', 'Age']].fillna(1)
    X_transformed = imputer.fit_transform(dataframe_na)

    ref = {'Name': ['tom', 'nick', 'krish', np.nan, 'peter', np.nan, 'fred', 'sam'],
           'City': ['London', 'Manchester', 'London', 'Manchester', 'London', 'London', 'Bristol', 'Manchester'],
           'Studies': ['Bachelor', 'Bachelor', 'Bachelor', 'Masters', 'Bachelor', 'PhD', 'None', 'Masters'],
           'Age': [20, 21, 19, np.nan, 23, 40, 41, 37],
           'Marks': [0.9, 0.8, 0.7, np.nan, 0.3, np.nan, 0.8, 0.6],
           'dob': pd.date_range('2020-02-24', periods=8, freq='T')
           }
    ref = pd.DataFrame(ref)

    assert imputer.variables == ['City', 'Studies']
    assert imputer.random_state == ['Marks', 'Age']
    assert imputer.seed == 'observation'
    pd.testing.assert_frame_equal(imputer.X_[['City', 'Studies']], dataframe_na[['City', 'Studies']])
    pd.testing.assert_frame_equal(X_transformed[['City', 'Studies']], ref[['City', 'Studies']], check_dtype=False)

    # test case 4: observation seed, only variable indicated as seed, method: addition
    # Note the variable used as seed should not have missing data
    imputer = RandomSampleImputer(variables=['City', 'Studies'], random_state='Age', seed='observation')
    dataframe_na['Age'] = dataframe_na['Age'].fillna(1)
    X_transformed = imputer.fit_transform(dataframe_na)

    ref = {'Name': ['tom', 'nick', 'krish', np.nan, 'peter', np.nan, 'fred', 'sam'],
           'City': ['London', 'Manchester', 'Manchester', 'Manchester', 'London', 'London', 'Bristol', 'Manchester'],
           'Studies': ['Bachelor', 'Bachelor', 'Masters', 'Masters', 'Bachelor', 'PhD', 'None', 'Masters'],
           'Age': [20, 21, 19, np.nan, 23, 40, 41, 37],
           'Marks': [0.9, 0.8, 0.7, np.nan, 0.3, np.nan, 0.8, 0.6],
           'dob': pd.date_range('2020-02-24', periods=8, freq='T')
           }
    ref = pd.DataFrame(ref)
    assert imputer.random_state == ['Age']
    pd.testing.assert_frame_equal(imputer.X_[['City', 'Studies']], dataframe_na[['City', 'Studies']])
    pd.testing.assert_frame_equal(X_transformed[['City', 'Studies']], ref[['City', 'Studies']], check_dtype=False)

    with pytest.raises(ValueError):
        RandomSampleImputer(seed='arbitrary')
    with pytest.raises(ValueError):
        RandomSampleImputer(seeding_method='arbitrary')
    with pytest.raises(ValueError):
        RandomSampleImputer(seed='general', random_state='arbitrary')
    with pytest.raises(ValueError):
        RandomSampleImputer(seed='observation', random_state=None)
    with pytest.raises(ValueError):
        imputer = RandomSampleImputer(seed='observation', random_state='arbitrary')
        imputer.fit(dataframe_na)
