# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import pandas as pd

def _is_dataframe(X):
    # checks if the input is a dataframe. Also creates a copy,
    # important not to transform the original dataset.
    if not isinstance(X, pd.DataFrame):
        raise TypeError("The data set should be a pandas dataframe")
    return X.copy()


def _check_input_matches_training_df(X, reference):
    # check that dataframe to transform has the same number of columns
    # that the dataframe used during fit method
    if X.shape[1] != reference:
        raise ValueError('The number of columns in this data set is different from that of the train set used during'
                         'the fit method')
    return None


def _check_contains_na(X, variables):
    if X[variables].isnull().values.any():
        raise ValueError('Some of the variables to trasnform contain missing values. Check and remove those '
                         'before using this transformer.')