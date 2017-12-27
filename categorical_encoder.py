# Authors: Soledad Galli <solegalli1@gmail.com>

# License: BSD 3 clause

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """ Transforms features by replacing the labels in each feature
    for the counts or frequency.
    
    Encoder can take both objects and numerical variables
    Please note that if no variables are passed, it will only encode 
    categorical variables
    
    This estimator first calculates the frequency / counts for the
    labels of the features.
    
    The transformer replaces the labels by the counts / frequecies
    
    Parameters
    ----------
    encoding_method : str, default=count 
        Desired method of encoding.
        'count': number of observations per label
        'frequency' : percentage of observation per label
        'mean' : target mean per label
        'ratio' : target probability ratio per label
        'woe' : weight of evidence
        
    Attributes
    ----------
    variables_ : list
        The list of variables that need encoding
        passed to the method:`fit`
        
    encoder_dict_: dictionary
        The dictionary containg the count / frequency: label pairs to use
        for each variable to replace missing data
        
    """
    
    def __init__(self, encoding_method='count'):
        self.encoding_method = encoding_method

    def fit(self, X, y=None, variables = None, tol = 0.0001):
        """ Learns thecounts or frequencies that should be used to replace
        the labels in each variable.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        variables: list of columns that should be encoded
        
        Returns
        -------
        self : object
            Returns self.
        """
        if not variables:
            # select all categorical variables
            self.variables_ = [var for var in X.columns if X[var].dtypes=='O']
            
        else:
            # variables indicated by user
            self.variables_ = variables
            # ADD AM ERROR HANDLER FOR NON NUMERICAL VARIABLES
            
        if y is not None:
            temp = pd.concat([X, y], axis=1)
            temp.columns = list(X.columns)+['target']
            
        self.encoder_dict_ = {}
        
        for var in self.variables_:
            if self.encoding_method == 'count':
                self.encoder_dict_[var] = X[var].value_counts().to_dict()
                
            elif self.encoding_method == 'frequency':
                n_obs = np.float(len(X))
                self.encoder_dict_[var] = (X[var].value_counts() / n_obs).to_dict()   
                
            elif self.encoding_method == 'mean':
                self.encoder_dict_[var] = temp.groupby(var)['target'].mean().to_dict()
                
            elif self.encoding_method == 'ratio':
                # add to handle error if y is not categorical
                t = temp.groupby(var)['target'].mean()
                t = pd.concat([t, 1-t], axis=1)
                t.columns = ['p1', 'p0']
                t.loc[t['p0'] == 0, 'p0'] = tol
                self.encoder_dict_[var] = (t.p1/t.p0).to_dict()
                
            elif self.encoding_method == 'woe':
                # add to handle error if y is not categorical
                t = temp.groupby(var)['target'].mean()
                t = pd.concat([t, 1-t], axis=1)
                t.columns = ['p1', 'p0']
                t.loc[t['p0'] == 0, 'p0'] = tol
                t.loc[t['p1'] == 0, 'p1'] = tol
                
                self.encoder_dict_[var] = (np.log(t.p1/t.p0)).to_dict()
                   
        return self

    def transform(self, X):
        """ Replaces labels with the counts or frequencies
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The dataframe containing no missing values for the selected
            variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['encoder_dict_', 'variables_'])
        
        X = X.copy()
        for feature in self.variables_:
            X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X