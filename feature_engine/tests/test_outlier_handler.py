# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import os

import numpy as np
import pandas as pd

from feature_engine.outlier_removers import Winsorizer

filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'titanic_test.csv')
df_test = pd.read_csv(filename)

    
def test_Windsorizer():
    imputer = Winsorizer(distribution='gaussian', tail='right', fold=3, variables=['Age'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = np.where(df_test['Age'] > 73.983130890142888, 73.983130890142888, df_test['Age'])
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.right_tail_caps_ == {'Age': 73.983130890142888}
    assert imputer.left_tail_caps_ == {}
    assert imputer.input_shape_ == (100,3)
    assert X['Age'].max() <= 73.983130890142888
    
    imputer = Winsorizer(distribution='gaussian', tail='both', fold=3,  variables=['Age'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = np.where(df_test['Age'] > 73.983130890142888, 73.983130890142888, df_test['Age'])
    df_transf['Age'] = np.where(df_test['Age'] < -14.090499311195508, -14.090499311195508, df_transf['Age'])
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.right_tail_caps_ == {'Age': 73.983130890142888}
    assert imputer.left_tail_caps_ == {'Age': -14.090499311195508}
    assert imputer.input_shape_ == (100,3)
    assert X['Age'].max() <= 73.983130890142888
    assert X['Age'].min() >= -14.090499311195508
    
    imputer = Winsorizer(distribution='gaussian', tail='right', fold=0.5, variables=['Age'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = np.where(df_test['Age'] > 37.285784972918549, 37.285784972918549, df_test['Age'])
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.right_tail_caps_ == {'Age': 37.285784972918549}
    assert imputer.left_tail_caps_ == {}
    assert imputer.input_shape_ == (100,3)
    assert X['Age'].max() <= 37.285784972918549
    
    imputer = Winsorizer(distribution='gaussian', tail='both', fold=0.5, variables=['Age'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = np.where(df_test['Age'] > 37.285784972918549, 37.285784972918549, df_test['Age'])
    df_transf['Age'] = np.where(df_test['Age'] < 22.606846606028821, 22.606846606028821, df_transf['Age'])
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.right_tail_caps_ == {'Age': 37.285784972918549}
    assert imputer.left_tail_caps_ == {'Age': 22.606846606028821}
    assert imputer.input_shape_ == (100,3)
    assert X['Age'].max() <= 37.285784972918549
    assert X['Age'].min() >= 22.606846606028821
        
    imputer = Winsorizer(distribution='skewed', tail='left', fold=1, variables=['Age'])
    imputer.fit(df_test)
    X = imputer.transform(df_test)
    df_transf = df_test.copy()
    df_transf['Age'] = np.where(df_test['Age'] < 1.5, 1.5, df_test['Age'])
    pd.testing.assert_frame_equal(X, df_transf)
    assert imputer.right_tail_caps_ == {}
    assert imputer.left_tail_caps_ == {'Age': 1.5}
    assert imputer.input_shape_ == (100,3)
    assert X['Age'].min() >= 1.5
