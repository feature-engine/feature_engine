# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from feature_engine.imputation.random_sample import _define_seed
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import EndTailImputer
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.imputation import RandomSampleImputer
from feature_engine.imputation import AddMissingIndicator


def test_define_seed(dataframe_vartypes):
    assert _define_seed(dataframe_vartypes, 0, ['Age', 'Marks'], how='add') == 21
    assert _define_seed(dataframe_vartypes, 0, ['Age', 'Marks'], how='multiply') == 18
    assert _define_seed(dataframe_vartypes, 2, ['Age', 'Marks'], how='add') == 20
    assert _define_seed(dataframe_vartypes, 2, ['Age', 'Marks'], how='multiply') == 13
    assert _define_seed(dataframe_vartypes, 1, ['Age'], how='add') == 21
    assert _define_seed(dataframe_vartypes, 3, ['Marks'], how='multiply') == 1


def test_MeanMedianImputer(dataframe_na):

    # test case 1: automatically finds numerical variables
    imputer = MeanMedianImputer(imputation_method='mean', variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(28.714285714285715)
    ref_df['Marks'] = ref_df['Marks'].fillna(0.6833333333333332)

    # check init params
    assert imputer.imputation_method == 'mean'
    assert imputer.variables == ['Age', 'Marks']

    # check fit attributes
    assert imputer.imputer_dict_ == {'Age': 28.714285714285715, 'Marks': 0.6833333333333332}
    assert imputer.input_shape_ == (8, 6)

    # check transform output: indicated variables no NA
    # Not indicated variables still have NA
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() == 0
    assert X_transformed[['Name', 'City']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 2: single user determined variable
    imputer = MeanMedianImputer(imputation_method='median', variables=['Age'])
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(23.0)

    # init params
    assert imputer.imputation_method == 'median'
    assert imputer.variables == ['Age']

    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Age': 23.0}

    # transform params
    assert X_transformed['Age'].isnull().sum() == 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    with pytest.raises(ValueError):
        MeanMedianImputer(imputation_method='arbitrary')

    with pytest.raises(NotFittedError):
        imputer = MeanMedianImputer()
        imputer.transform(dataframe_na)


def test_EndTailImputer(dataframe_na):

    # test case 1: automatically find variables + gaussian limits + right tail
    imputer = EndTailImputer(distribution='gaussian', tail='right', fold=3, variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(58.94908118478389)
    ref_df['Marks'] = ref_df['Marks'].fillna(1.3244261503263175)

    # init params
    assert imputer.distribution == 'gaussian'
    assert imputer.tail == 'right'
    assert imputer.fold == 3
    assert imputer.variables == ['Age', 'Marks']
    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Age': 58.94908118478389, 'Marks': 1.3244261503263175}
    # transform params: indicated vars ==> no NA, not indicated vars with NA
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() == 0
    assert X_transformed[['City', 'Name']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 2: selected variables + IQR rule + right tail
    imputer = EndTailImputer(distribution='skewed', tail='right', fold=1.5, variables=['Age', 'Marks'])
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(65.5)
    ref_df['Marks'] = ref_df['Marks'].fillna(1.0625)
    # fit  and transform params
    assert imputer.imputer_dict_ == {'Age': 65.5, 'Marks': 1.0625}
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() == 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 3: selected variables + maximum value
    imputer = EndTailImputer(distribution='max', tail='right', fold=2, variables=['Age', 'Marks'])
    imputer.fit(dataframe_na)
    assert imputer.imputer_dict_ == {'Age': 82.0, 'Marks': 1.8}

    # test case 4: automatically select variables + gaussian limits + left tail
    imputer = EndTailImputer(distribution='gaussian', tail='left', fold=3)
    imputer.fit(dataframe_na)
    assert imputer.imputer_dict_ == {'Age': -1.520509756212462, 'Marks': 0.04224051634034898}

    # test case 5: IQR + left tail
    imputer = EndTailImputer(distribution='skewed', tail='left', fold=1.5, variables=['Age', 'Marks'])
    imputer.fit(dataframe_na)
    assert imputer.imputer_dict_ == {'Age': -6.5, 'Marks': 0.36249999999999993}

    with pytest.raises(ValueError):
        EndTailImputer(distribution='arbitrary')

    with pytest.raises(ValueError):
        EndTailImputer(tail='arbitrary')

    with pytest.raises(ValueError):
        EndTailImputer(fold=-1)

    with pytest.raises(NotFittedError):
        imputer = EndTailImputer()
        imputer.transform(dataframe_na)


def test_ArbitraryNumberImputer(dataframe_na):

    # test case 1: automatically select variables
    imputer = ArbitraryNumberImputer(arbitrary_number=99, variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(99)
    ref_df['Marks'] = ref_df['Marks'].fillna(99)

    # init params
    assert imputer.arbitrary_number == 99
    assert imputer.variables == ['Age', 'Marks']
    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Age': 99, 'Marks': 99}
    # transform params
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() == 0
    assert X_transformed[['Name', 'City']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 2: user indicates variables
    imputer = ArbitraryNumberImputer(arbitrary_number=-1, variables=['Age'])
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(-1)

    # init params
    assert imputer.arbitrary_number == -1
    assert imputer.variables == ['Age']
    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Age': -1}
    # transform output
    assert X_transformed['Age'].isnull().sum() == 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    with pytest.raises(ValueError):
        ArbitraryNumberImputer(arbitrary_number='arbitrary')

    with pytest.raises(NotFittedError):
        imputer = ArbitraryNumberImputer()
        imputer.transform(dataframe_na)

    # test case 3: arbitrary numbers passed as dict
    imputer = ArbitraryNumberImputer(imputer_dict={'Age': -42, 'Marks': -999})
    X_transformed = imputer.fit_transform(dataframe_na)
    ref_df = dataframe_na.copy()
    ref_df['Age'] = ref_df['Age'].fillna(-42)
    ref_df['Marks'] = ref_df['Marks'].fillna(-999)

    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Age': -42, 'Marks': -999}
    # transform params
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() == 0
    assert X_transformed[['Name', 'City']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    with pytest.raises(ValueError):
        ArbitraryNumberImputer(imputer_dict={'Age': 'arbitrary_number'})

def test_CategoricalVariableImputer(dataframe_na):

    # test case 1: imputation with missing + automatically select variables
    imputer = CategoricalImputer(imputation_method='missing', variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Name'] = ref_df['Name'].fillna('Missing')
    ref_df['City'] = ref_df['City'].fillna('Missing')
    ref_df['Studies'] = ref_df['Studies'].fillna('Missing')

    # init params
    assert imputer.imputation_method == 'missing'
    assert imputer.variables == ['Name', 'City', 'Studies']
    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Name': 'Missing', 'City': 'Missing', 'Studies': 'Missing'}
    # transform output
    assert X_transformed[['Name', 'City', 'Studies']].isnull().sum().sum() == 0
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)
    
    # test case 2: imputing custom user-defined string + automatically select variables
    imputer = CategoricalImputer(imputation_method='missing', fill_value='Unknown', variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Name'] = ref_df['Name'].fillna('Unknown')
    ref_df['City'] = ref_df['City'].fillna('Unknown')
    ref_df['Studies'] = ref_df['Studies'].fillna('Unknown')

    # init params
    assert imputer.imputation_method == 'missing'
    assert imputer.fill_value == 'Unknown'
    assert imputer.variables == ['Name', 'City', 'Studies']
    # fit params
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'Name': 'Unknown', 'City': 'Unknown', 'Studies': 'Unknown'}
    # transform output
    assert X_transformed[['Name', 'City', 'Studies']].isnull().sum().sum() == 0
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 3: mode imputation + user indicates 1 variable ONLY
    imputer = CategoricalImputer(imputation_method='frequent', variables='City')
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['City'] = ref_df['City'].fillna('London')

    assert imputer.imputation_method == 'frequent'
    assert imputer.variables == ['City']
    assert imputer.input_shape_ == (8, 6)
    assert imputer.imputer_dict_ == {'City': 'London'}
    assert X_transformed['City'].isnull().sum() == 0
    assert X_transformed[['Age', 'Marks']].isnull().sum().sum() > 0
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 4: mode imputation + user indicates multiple variables
    imputer = CategoricalImputer(imputation_method='frequent', variables=['Studies', 'City'])
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['City'] = ref_df['City'].fillna('London')
    ref_df['Studies'] = ref_df['Studies'].fillna('Bachelor')

    assert imputer.imputer_dict_ == {'Studies': 'Bachelor', 'City': 'London'}
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 5: imputing of numerical variables cast as object + return numeric
    dataframe_na['Marks'] = dataframe_na['Marks'].astype('O')
    imputer = CategoricalImputer(imputation_method='frequent', variables=['City', 'Studies', 'Marks'])
    X_transformed = imputer.fit_transform(dataframe_na)

    ref_df = dataframe_na.copy()
    ref_df['Marks'] = ref_df['Marks'].fillna(0.8)
    ref_df['City'] = ref_df['City'].fillna('London')
    ref_df['Studies'] = ref_df['Studies'].fillna('Bachelor')
    assert imputer.variables == ['City', 'Studies', 'Marks']
    assert imputer.imputer_dict_ == {'Studies': 'Bachelor', 'City': 'London', 'Marks': 0.8}
    assert X_transformed['Marks'].dtype == 'float'
    pd.testing.assert_frame_equal(X_transformed, ref_df)

    # test case 6: imputing of numerical variables cast as object + return as object after imputation
    dataframe_na['Marks'] = dataframe_na['Marks'].astype('O')
    imputer = CategoricalImputer(imputation_method='frequent', variables=['City', 'Studies', 'Marks'],
                                 return_object=True)
    X_transformed = imputer.fit_transform(dataframe_na)
    assert X_transformed['Marks'].dtype == 'O'

    with pytest.raises(ValueError):
        imputer = CategoricalImputer(imputation_method='arbitrary')

    with pytest.raises(ValueError):
        imputer = CategoricalImputer(imputation_method='frequent')
        imputer.fit(dataframe_na)

    with pytest.raises(NotFittedError):
        imputer = CategoricalImputer()
        imputer.transform(dataframe_na)


def test_AddMissingIndicator(dataframe_na):

    # test case 1: automatically detect variables with missing data
    imputer = AddMissingIndicator(how='missing_only', variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)

    # init params
    assert imputer.how == 'missing_only'
    assert imputer.variables == None
    # fit params
    assert imputer.variables_ == ['Name', 'City', 'Studies', 'Age', 'Marks']
    assert imputer.input_shape_ == (8, 6)
    # transform outputs
    assert X_transformed.shape == (8, 11)
    assert 'Name_na' in X_transformed.columns
    assert X_transformed['Name_na'].sum() == 2

    # test case 2: automatically detect all columns
    imputer = AddMissingIndicator(how='all', variables=None)
    X_transformed = imputer.fit_transform(dataframe_na)
    assert imputer.variables_ == ['Name', 'City', 'Studies', 'Age', 'Marks', 'dob']
    assert X_transformed.shape == (8, 12)
    assert 'dob_na' in X_transformed.columns
    assert X_transformed['dob_na'].sum() == 0

    # test case 3: find variables with NA, among those entered by the user
    imputer = AddMissingIndicator(how='missing_only', variables=['City', 'Studies', 'Age', 'dob'])
    X_transformed = imputer.fit_transform(dataframe_na)
    assert imputer.variables == ['City', 'Studies', 'Age', 'dob']
    assert imputer.variables_ == ['City', 'Studies', 'Age']
    assert X_transformed.shape == (8, 9)
    assert 'City_na' in X_transformed.columns
    assert 'dob_na' not in X_transformed.columns
    assert X_transformed['City_na'].sum() == 2

    with pytest.raises(ValueError):
        AddMissingIndicator(how='arbitrary')

    with pytest.raises(NotFittedError):
        imputer = AddMissingIndicator()
        imputer.transform(dataframe_na)


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
