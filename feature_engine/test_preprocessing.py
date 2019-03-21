# Authors: João Nogueira <joaonogueira@fisica.ufc.br>
# License: BSD 3 clause

import pandas as pd
import pytest

from preprocessers import MinMaxScaler

def test_MinMaxScaler():
    df = {'f1':[0,1,2,3,4,5], 
        'f2': [100,200,300,400,500,600],
        'f3': [1000,2000,3000,4000,5000,6000]}

    df = pd.DataFrame(df)
    
    transf_df = {'f1': [0, 0.2, 0.4, 0.6, 0.8, 1],
    'f2': [0, 0.2, 0.4, 0.6, 0.8, 1],
    'f3': [0, 0.2, 0.4, 0.6, 0.8, 1]}
    transf_df = pd.DataFrame(transf_df)

    # all the variables
    scaler = MinMaxScaler()
    scaler.fit(df)
    X = scaler.transform(df)

    assert list(X.columns) == list(df.columns)
    assert list(X.columns) == list(transf_df.columns)
    assert X.shape == df.shape
    assert X.shape == transf_df.shape

    for var in X.columns:
        assert X[var].min() == 0
        assert X[var].max() == 1
        assert X[var].equals(transf_df[var])

    # only one variable
    vars = ['f1']
    scaler = MinMaxScaler(variables=vars)
    scaler.fit(df)
    X = scaler.transform(df)

    assert list(X.columns) == list(df.columns)
    assert list(X.columns) == list(transf_df.columns)
    assert X.shape == df.shape
    assert X.shape == transf_df.shape

    for var in vars:
        assert X[var].min() == 0
        assert X[var].max() == 1
        assert X[var].equals(transf_df[var])

    # two variables
    vars = ['f1','f2']
    scaler = MinMaxScaler(variables=vars)
    scaler.fit(df)
    X = scaler.transform(df)

    assert list(X.columns) == list(df.columns)
    assert list(X.columns) == list(transf_df.columns)
    assert X.shape == df.shape
    assert X.shape == transf_df.shape

    for var in vars:
        assert X[var].min() == 0
        assert X[var].max() == 1
        assert X[var].equals(transf_df[var])

    # two variables
    vars = ['f3','f2']
    scaler = MinMaxScaler(variables=vars)
    scaler.fit(df)
    X = scaler.transform(df)

    assert list(X.columns) == list(df.columns)
    assert list(X.columns) == list(transf_df.columns)
    assert X.shape == df.shape
    assert X.shape == transf_df.shape

    for var in vars:
        assert X[var].min() == 0
        assert X[var].max() == 1
        assert X[var].equals(transf_df[var])


if __name__ == '__main__':
    test_MinMaxScaler() 