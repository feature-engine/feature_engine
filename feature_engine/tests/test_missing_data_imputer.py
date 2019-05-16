# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np
import pytest

import os
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'titanic_test.csv')

from missing_data_imputation import MeanMedianImputer
from missing_data_imputation import RandomSampleImputer
from missing_data_imputation import EndTailImputer
from missing_data_imputation import na_capturer
from missing_data_imputation import CategoricalImputer
from missing_data_imputation import ArbitraryImputer

df_test = pd.read_csv(filename)

def test_MeanMedianImputer():
    imputer = MeanMedianImputer(imputation_method='median')
    imputer.fit(df_test, variables = ['Age'])
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(31.5)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.variables_ == ['Age']
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':31.5}
    assert X['Age'].isnull().sum() == 0
    
    imputer = MeanMedianImputer(imputation_method='mean')
    imputer.fit(df_test, variables = ['Age'])
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(29.946315789473687)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.variables_ == ['Age']
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':29.946315789473687}
    assert X['Age'].isnull().sum() == 0
    
    
def test_RandomSampleImputer():
    imputer = RandomSampleImputer()
    imputer.fit(df_test)
    X = imputer.transform(df_test, random_state=0)
    pd.testing.assert_frame_equal(imputer.X_, df_test)
    assert imputer.input_shape_ == (100,3)
    assert np.sum(X.isnull().sum()) == 0
    
    
def test_EndTailImputer():
    imputer = EndTailImputer(distribution='gaussian', tail='right', fold=3)
    imputer.fit(df_test, variables = ['Age'])
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(73.98313089014289)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.variables_ == ['Age']
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':73.98313089014289}
    assert X['Age'].isnull().sum() == 0
    
    imputer = EndTailImputer(distribution='gaussian', tail='right', fold=1.5)
    imputer.fit(df_test, variables = ['Age'])
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(51.96472333980829)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.variables_ == ['Age']
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':51.96472333980829}
    assert X['Age'].isnull().sum() == 0
    
    imputer = EndTailImputer(distribution='gaussian', tail='left', fold=2)
    imputer.fit(df_test, variables = ['Age'])
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(0.5884390556942236)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.variables_ == ['Age']
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':0.5884390556942236}
    assert X['Age'].isnull().sum() == 0
    
    imputer = EndTailImputer(distribution='skewed', tail='right', fold=3)
    imputer.fit(df_test, variables = ['Age'])
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(95.25)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.variables_ == ['Age']
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':95.25}
    assert X['Age'].isnull().sum() == 0
    
    imputer = EndTailImputer(distribution='skewed', tail='left', fold=3)
    imputer.fit(df_test, variables = ['Age'])
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(-36.0)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.variables_ == ['Age']
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':-36.0}
    assert X['Age'].isnull().sum() == 0
    
    
def test_na_capturer():
    imputer = na_capturer(tol=0.05)
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age_na'] = np.where(df_test['Age'].isnull(),1,0)
    df_transf['Cabin_na'] = np.where(df_test['Cabin'].isnull(),1,0)
    pd.testing.assert_frame_equal(X, df_transf) 
    assert imputer.variables_ == ['Age', 'Cabin']
    assert imputer.input_shape_ == (100,3)
    assert X.shape == (100,5)
    
    
def test_CategoricalImputer():
    imputer = CategoricalImputer(tol=0.05)
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Cabin'] = df_test['Cabin'].fillna('Missing')
    df_transf['Embarked'] = df_test['Embarked'].fillna('S')
    pd.testing.assert_frame_equal(X, df_transf) 
    assert imputer.variables_ == ['Cabin', 'Embarked']
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_['Cabin'] == 'Missing'
    assert imputer.imputer_dict_['Embarked'] == 'S'
    assert np.sum(X[['Cabin', 'Embarked']].isnull().sum()) == 0
    
    
def test_ArbitraryImputer():
    imputer = ArbitraryImputer()
    imputer.fit(df_test, imputation_dictionary = {'Age':100,
                                                  'Cabin':'Missing',
                                                  'Embarked':'Unknown'})
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(100)
    df_transf['Cabin'] = df_test['Cabin'].fillna('Missing')
    df_transf['Embarked'] = df_test['Embarked'].fillna('Unknown')
    pd.testing.assert_frame_equal(X, df_transf) 
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':100,'Cabin':'Missing','Embarked':'Unknown'}
    assert np.sum(X.isnull().sum()) == 0
    
       
#if __name__ == '__main__':
#    test_MeanMedianImputer()
#    test_RandomSampleImputer()
#    test_EndTailImputer()
#    test_na_capturer()
#    test_CategoricalImputer()
#    test_ArbitraryImputer()
