# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import os

import numpy as np
import pandas as pd

filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'titanic_test.csv')

from feature_engine.missing_data_imputers import MeanMedianImputer
from feature_engine.missing_data_imputers import RandomSampleImputer
from feature_engine.missing_data_imputers import EndTailImputer
from feature_engine.missing_data_imputers import CategoricalVariableImputer


df_test = pd.read_csv(filename)

def test_MeanMedianImputer():
    imputer = MeanMedianImputer(imputation_method='median', variables=['Age'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(31.5)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':31.5}
    assert X['Age'].isnull().sum() == 0
    
    imputer = MeanMedianImputer(imputation_method='mean', variables=['Age'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(29.946315789473687)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':29.946315789473687}
    assert X['Age'].isnull().sum() == 0
    
    
def test_RandomSampleImputer():
    imputer = RandomSampleImputer(random_state=0)
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    pd.testing.assert_frame_equal(imputer.X, df_test)
    assert imputer.input_shape_ == (100,3)
    assert np.sum(X.isnull().sum()) == 0
    
    
def test_EndTailImputer():
    imputer = EndTailImputer(distribution='gaussian', tail='right', fold=3, variables=['Age'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(73.98313089014289)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':73.98313089014289}
    assert X['Age'].isnull().sum() == 0
    
    imputer = EndTailImputer(distribution='gaussian', tail='right', fold=1.5, variables=['Age'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(51.96472333980829)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':51.96472333980829}
    assert X['Age'].isnull().sum() == 0
    
    imputer = EndTailImputer(distribution='gaussian', tail='left', fold=2, variables=['Age'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(0.5884390556942236)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':0.5884390556942236}
    assert X['Age'].isnull().sum() == 0
    
    imputer = EndTailImputer(distribution='skewed', tail='right', fold=3, variables=['Age'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(95.25)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':95.25}
    assert X['Age'].isnull().sum() == 0
    
    imputer = EndTailImputer(distribution='skewed', tail='left', fold=3, variables=['Age'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = df_test['Age'].fillna(-36.0)
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.input_shape_ == (100,3)
    assert imputer.imputer_dict_ == {'Age':-36.0}
    assert X['Age'].isnull().sum() == 0
    
    
def test_CategoricalImputer():
    imputer = CategoricalVariableImputer(variables=['Cabin', 'Embarked'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Cabin'] = df_test['Cabin'].fillna('Missing')
    df_transf['Embarked'] = df_test['Embarked'].fillna('Missing')
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.input_shape_ == (100,3)
    assert np.sum(X[['Cabin', 'Embarked']].isnull().sum()) == 0
