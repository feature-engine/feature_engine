# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np
import pytest

from outlier_handler import Windsorizer

import os
filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'titanic_test.csv')
df_test = pd.read_csv(filename)

    
def test_Windsorizer():
    imputer = Windsorizer(distribution='gaussian', end='right', fold=3)
    imputer.fit(df_test, variables = ['Age'])
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = np.where(df_test['Age'] > 73.983130890142888, 73.983130890142888, df_test['Age'])
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.capping_max_ == {'Age': 73.983130890142888}
    assert imputer.capping_min_ == {}
    assert imputer.input_shape_ == (100,3)
    assert X['Age'].max() <= 73.983130890142888
    
    imputer = Windsorizer(distribution='gaussian', end='both', fold=3)
    imputer.fit(df_test, variables = ['Age'])
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = np.where(df_test['Age'] > 73.983130890142888, 73.983130890142888, df_test['Age'])
    df_transf['Age'] = np.where(df_test['Age'] < -14.090499311195508, -14.090499311195508, df_transf['Age'])
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.capping_max_ == {'Age': 73.983130890142888}
    assert imputer.capping_min_ == {'Age': -14.090499311195508}
    assert imputer.input_shape_ == (100,3)
    assert X['Age'].max() <= 73.983130890142888
    assert X['Age'].min() >= -14.090499311195508
    
    imputer = Windsorizer(distribution='gaussian', end='right', fold=0.5)
    imputer.fit(df_test, variables = ['Age'])
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = np.where(df_test['Age'] > 37.285784972918549, 37.285784972918549, df_test['Age'])
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.capping_max_ == {'Age': 37.285784972918549}
    assert imputer.capping_min_ == {}
    assert imputer.input_shape_ == (100,3)
    assert X['Age'].max() <= 37.285784972918549
    
    imputer = Windsorizer(distribution='gaussian', end='both', fold=0.5)
    imputer.fit(df_test, variables = ['Age'])
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = np.where(df_test['Age'] > 37.285784972918549, 37.285784972918549, df_test['Age'])
    df_transf['Age'] = np.where(df_test['Age'] < 22.606846606028821, 22.606846606028821, df_transf['Age'])
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.capping_max_ == {'Age': 37.285784972918549}
    assert imputer.capping_min_ == {'Age': 22.606846606028821}
    assert imputer.input_shape_ == (100,3)
    assert X['Age'].max() <= 37.285784972918549
    assert X['Age'].min() >= 22.606846606028821
        
    imputer = Windsorizer(distribution='skewed', end='left', fold=1)
    imputer.fit(df_test, variables = ['Age'])
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = np.where(df_test['Age'] < 1.5, 1.5, df_test['Age'])
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.capping_max_ == {}
    assert imputer.capping_min_ == {'Age': 1.5}
    assert imputer.input_shape_ == (100,3)
    assert X['Age'].min() >= 1.5
    
    imputer = Windsorizer(distribution='skewed', end='right', user_input=True)
    imputer.fit(df_test, variables = ['Age'], capping_max = {'Age':60})
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = np.where(df_test['Age'] > 60, 60, df_test['Age'])
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.capping_max_ == {'Age':60}
    assert imputer.capping_min_ == {}
    assert imputer.input_shape_ == (100,3)
    assert X['Age'].max() <= 60
    
       
#if __name__ == '__main__':
#    test_Windsorizer()
