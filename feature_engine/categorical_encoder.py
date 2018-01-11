"""
Methods to encode categorical variables into numbers and to 
handle rare categories
"""

# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np

import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """ Transforms categories into numbers by multiple methods.
    
    The Encoder can take both categorical and numerical variables if passed
    as an argument. If no variables are passed as argument, it will only encode 
    categorical variables (object type).
    
    The estimator first maps the labels to the numbers for each feature.
    
    The transformer replaces the labels by numbers.
    
    Parameters
    ----------
    encoding_method : str, default=count 
        Desired method of encoding.
        'count': number of observations per label
        'frequency' : percentage of observation per label
        'ordinal' : labels are labelled according to increasing target mean value
        'mean' : target mean per label
        'ratio' : target probability ratio per label
        'woe' : weight of evidence
        
    tol: float, default = 0.0001
        For those cases where the mathematical operation of 0 is not defined
        the transformer adds tol to the value
        
    Attributes
    ----------
    variables_ : list
        The list of categorical variables that need encoding
        passed to the method:'fit'
        
    encoder_dict_: dictionary
        The dictionary containg the count / frequency: label pairs to use
        for each variable to replace missing data
        
    """
    
    def __init__(self, encoding_method='count', tol = 0.0001):
        if encoding_method not in ['count', 'frequency','ordinal','mean','ratio','woe']:
            raise ValueError("encoding_method takes only values 'count', 'frequency','ordinal','mean','ratio','woe'")
        self.encoding_method = encoding_method
        self.tol = tol
           

    def fit(self, X, y=None, variables = None):
        """ Learns the numbers that should be used to replace
        the labels in each variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.
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
            
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            if len(X[self.variables_].select_dtypes(include=numerics).columns) != 0:
                warnings.warn("Some of the selected variables are of dtype numeric; they will be handled as categorical")
            
        # for methods that encode labels using dependent variable (target) information
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
                
            elif self.encoding_method == 'ordinal':
                t = temp.groupby([var])['target'].mean().sort_values().index
                self.encoder_dict_[var] = {k:i for i, k in enumerate(t, 0)} 
                
            elif self.encoding_method == 'mean':
                self.encoder_dict_[var] = temp.groupby(var)['target'].mean().to_dict()
                
            elif self.encoding_method == 'ratio':
                t = temp.groupby(var)['target'].mean()
                t = pd.concat([t, 1-t], axis=1)
                t.columns = ['p1', 'p0']
                t.loc[t['p0'] == 0, 'p0'] = self.tol
                self.encoder_dict_[var] = (t.p1/t.p0).to_dict()
                
            elif self.encoding_method == 'woe':
                t = temp.groupby(var)['target'].mean()
                t = pd.concat([t, 1-t], axis=1)
                t.columns = ['p1', 'p0']
                t.loc[t['p0'] == 0, 'p0'] = self.tol
                t.loc[t['p1'] == 0, 'p1'] = self.tol
                self.encoder_dict_[var] = (np.log(t.p1/t.p0)).to_dict()
                
        if len(self.encoder_dict_)==0:
            raise ValueError('Encoder could not be fitted. Check that correct parameters and dataframe were passed during training')
            
        self.input_shape_ = X.shape
                   
        return self

    def transform(self, X):
        """ Replaces labels with the numbers.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing no missing values for the selected
            variables
        """
        # Check is fit had been called
        check_is_fitted(self, ['encoder_dict_', 'variables_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set '
                             ' used to fit the encoder')
             
        X = X.copy()
        for feature in self.variables_:
            X[feature] = X[feature].map(self.encoder_dict_[feature])
        
        return X
    
    
class RareLabelEncoder(BaseEstimator, TransformerMixin):
    """ Transforms features by grouping rare labels in each feature
    under the separate category "Rare"
    
    The Encoder can take both categorical and numerical variables if passed
    as an argument. If no variables are passed as argument, it will only encode 
    categorical variables (object type).
    
    This estimator first determines the frequent labels for each feature.
    
    The transformer replaces the unusual labels by the selected method
    
    Parameters
    ----------
    tol: float, default = 0.05
        the frequency a label should have to be considered frequent
        and not be removed.
    n_categories: int, default = 10
        the minimum number of categories a variable should have to
        process rare labels.
        
    Attributes
    ----------
    variables_ : list
        The list of variables that need encoding
        passed to the method:`fit`
        
    encoder_dict_: dictionary
        The dictionary containg the usual lanels + the selected label for
        replacement for each variable
        
    """
    
    def __init__(self, tol = 0.05, n_categories = 10):
        if tol <0 or tol >1 :
            raise ValueError("tol takes values between 0 and 1")
            
        if n_categories < 0 :
            raise ValueError("n_categories takes only positive numbers")
        
        self.tol = tol
        self.n_categories = n_categories
        

    def fit(self, X, y=None, variables = None):
        """ Learns the frequent labels.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables
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

            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            if len(X[self.variables_].select_dtypes(include=numerics).columns) != 0:
               warnings.warn("Some of the selected variables are of dtype numeric; they will be handled as categorical")
                        
        self.encoder_dict_ = {}
        
        for var in self.variables_:
            if len(X[var].unique()) > self.n_categories:
                t = pd.Series(X[var].value_counts() / np.float(len(X)))
                # non-rare labels:
                self.encoder_dict_[var] = t[t>=self.tol].index
        
        self.input_shape_ = X.shape
        
        if len(self.encoder_dict_)==0:
            warnings.warn("No rare labels were identified. Please change the encoder parameters")
            
        return self
    

    def transform(self, X):
        """ Groups rare labels under separate group.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing where rare categories have been grouped
        """
        # Check is fit had been called
        check_is_fitted(self, ['encoder_dict_', 'variables_'])
        
        # check that there is an encoder dictionary
        if len(self.encoder_dict_)==0:
            raise ValueError('No rare labels were identified during training. Please change the encoder parameters and re-train')
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set '
                             ' used to fit the encoder')
        
        X = X.copy()
        for feature in self.variables_:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]), X[feature], 'Rare')
            
        return X