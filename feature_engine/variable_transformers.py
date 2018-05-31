# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
#import warnings
import scipy.stats as stats 

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted



class BaseMathTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        if not self.variables:
            # select all numerical variables
            self.variables = list(X.select_dtypes(include=numerics).columns)            
        else:
            # variables indicated by user
            if len(X[self.variables].select_dtypes(exclude=numerics).columns) != 0:
               raise ValueError("Some of the selected variables are NOT numerical. Please cast them as numerical before fitting the imputer")
            
            self.variables = self.variables


class LogTransformer(BaseMathTransformer):
    """ Divides the numerical variable values into equal frequency buckets.
    Equal frequency buckets will contains roughly the same amount of observations.
    
    The buckets are estimated using pandas.qcut. The number of buckets, i.e., 
    quantiles, in which the variable should be divided must be indicated by the
    user.
    
    The Discretiser will binnarise only numerical variables (type 'object'). A 
    list of variables can be passed as an argument. If no variables are indicated,
    the discretiser will only binnarise numerical variables and ignore the rest.
    
    The discretiser first finds the boundaries for the buckets for each variable
    (fit).
    
    The transformer sorts the values into the buckets (transform).
    
    Parameters
    ----------
    q : int, default=10
        Desired number of equal frequency buckets / bins.
    
    variables : list
        The list of numerical variables that will be discretised. If none, it 
        defaults to all numerical type variables.
        
    Attributes
    ----------
    binner_dict_: dictionary
        The dictionary containing the bucket boundaries: variable pairs used
        to binnarise / discretise variable    
    """
    
    def __init__(self, variables = None):
        
        self.variables = variables
    
    
    def fit(self, X, y=None):
        """ Learns the boundaries of the equal frequency buckets / bins for each
        variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this encoder, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
        """

        super().fit(X, y)
        
        if (X[self.variables]<=0).all().all():
            raise ValueError("Some variables contain zero or negative values, can't apply log")

        self.input_shape_ = X.shape             
        return self


    def transform(self, X):
        """ Discretises the variables in the selected bins.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with discrete /  binned variables 
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')

        X = X.copy()
        X[self.variables] = np.log(X[self.variables])
            
        return X
    
    
class ReciprocalTransformer(BaseMathTransformer):
    """ Divides the numerical variable values into equal frequency buckets.
    Equal frequency buckets will contains roughly the same amount of observations.
    
    The buckets are estimated using pandas.qcut. The number of buckets, i.e., 
    quantiles, in which the variable should be divided must be indicated by the
    user.
    
    The Discretiser will binnarise only numerical variables (type 'object'). A 
    list of variables can be passed as an argument. If no variables are indicated,
    the discretiser will only binnarise numerical variables and ignore the rest.
    
    The discretiser first finds the boundaries for the buckets for each variable
    (fit).
    
    The transformer sorts the values into the buckets (transform).
    
    Parameters
    ----------
    q : int, default=10
        Desired number of equal frequency buckets / bins.
    
    variables : list
        The list of numerical variables that will be discretised. If none, it 
        defaults to all numerical type variables.
        
    Attributes
    ----------
    binner_dict_: dictionary
        The dictionary containing the bucket boundaries: variable pairs used
        to binnarise / discretise variable    
    """
    
    def __init__(self, variables = None):
        
        self.variables = variables
    
    
    def fit(self, X, y=None):
        """ Learns the boundaries of the equal frequency buckets / bins for each
        variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this encoder, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
        """

        super().fit(X, y)
        
        if (X[self.variables]==0).all().all():
            raise ValueError("Some variables contain the value zero, can't apply log")

        self.input_shape_ = X.shape             
        return self


    def transform(self, X):
        """ Discretises the variables in the selected bins.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with discrete /  binned variables 
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')

        X = X.copy()
        X[self.variables] = 1 / (X[self.variables])
            
        return X
    
    
class ExponentialTransformer(BaseMathTransformer):
    """ Divides the numerical variable values into equal frequency buckets.
    Equal frequency buckets will contains roughly the same amount of observations.
    
    The buckets are estimated using pandas.qcut. The number of buckets, i.e., 
    quantiles, in which the variable should be divided must be indicated by the
    user.
    
    The Discretiser will binnarise only numerical variables (type 'object'). A 
    list of variables can be passed as an argument. If no variables are indicated,
    the discretiser will only binnarise numerical variables and ignore the rest.
    
    The discretiser first finds the boundaries for the buckets for each variable
    (fit).
    
    The transformer sorts the values into the buckets (transform).
    
    Parameters
    ----------
    q : int, default=10
        Desired number of equal frequency buckets / bins.
    
    variables : list
        The list of numerical variables that will be discretised. If none, it 
        defaults to all numerical type variables.
        
    Attributes
    ----------
    binner_dict_: dictionary
        The dictionary containing the bucket boundaries: variable pairs used
        to binnarise / discretise variable    
    """
    
    def __init__(self, exp = 0.5, variables = None):
        
        if not isinstance(exp, float) and not isinstance(exp, int):
            raise ValueError('exp must be a float or an int')
            
        self.exp = exp
        self.variables = variables
    
    
    def fit(self, X, y=None):
        """ Learns the boundaries of the equal frequency buckets / bins for each
        variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this encoder, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
        """

        super().fit(X, y)
        
        self.input_shape_ = X.shape             
        return self


    def transform(self, X):
        """ Discretises the variables in the selected bins.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with discrete /  binned variables 
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')

        X = X.copy()
        X[self.variables] = X[self.variables]**self.exp
            
        return X
    
    
class BoxCoxTransformer(BaseMathTransformer):
    """ Divides the numerical variable values into equal frequency buckets.
    Equal frequency buckets will contains roughly the same amount of observations.
    
    The buckets are estimated using pandas.qcut. The number of buckets, i.e., 
    quantiles, in which the variable should be divided must be indicated by the
    user.
    
    The Discretiser will binnarise only numerical variables (type 'object'). A 
    list of variables can be passed as an argument. If no variables are indicated,
    the discretiser will only binnarise numerical variables and ignore the rest.
    
    The discretiser first finds the boundaries for the buckets for each variable
    (fit).
    
    The transformer sorts the values into the buckets (transform).
    
    Parameters
    ----------
    q : int, default=10
        Desired number of equal frequency buckets / bins.
    
    variables : list
        The list of numerical variables that will be discretised. If none, it 
        defaults to all numerical type variables.
        
    Attributes
    ----------
    binner_dict_: dictionary
        The dictionary containing the bucket boundaries: variable pairs used
        to binnarise / discretise variable    
    """
    
    def __init__(self, variables = None):
        
        self.variables = variables
    
    
    def fit(self, X, y=None):
        """ Learns the boundaries of the equal frequency buckets / bins for each
        variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this encoder, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
        """

        super().fit(X, y)
        
        self.lambda_dict = {}
        
        for var in self.variables:
            x, self.lambda_dict[var] = stats.boxcox(X[var]) 
            
        self.input_shape_ = X.shape
        
        return self


    def transform(self, X):
        """ Discretises the variables in the selected bins.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with discrete /  binned variables 
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_', 'lambda_dict'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')

        X = X.copy()
        for feature in self.variables:
            X[feature], x = stats.boxcox(X[feature], lamnda=self.lambda_dict[var]) 
            
        return X