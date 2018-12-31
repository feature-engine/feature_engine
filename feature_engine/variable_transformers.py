# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import numpy as np
#import pandas as pd
#import warnings
import scipy.stats as stats 

from sklearn.utils.validation import check_is_fitted
from feature_engine.base_transformers import BaseNumericalTransformer, _define_variables


class LogTransformer(BaseNumericalTransformer):
    """ Applies the logarithmic transformation to the numerical variables.
    
    The transformer only works with numerical non-negative values.
        
    Parameters
    ----------
    variables : list
        The list of numerical variables that will be transformed. If none, it 
        defaults to all numerical type variables.
        
    Attributes
    ----------   
    """
    
    def __init__(self, variables = None):
        
        self.variables = _define_variables(variables)
    
    
    def fit(self, X, y=None):
        """ Selects the numerical variables. Determines whether logarithm can
        be applied on selected variables.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
        """

        super().fit(X, y)
        
        if (X[self.variables]<=0).all().all():
            raise ValueError("Some variables contain zero or negative values, can't apply log")

        self.input_shape_ = X.shape             
        return self


    def transform(self, X):
        """ Transforms the variables using logarithm.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The log transformed dataframe. 
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the transformer')

        X = X.copy()
        X.loc[:, self.variables] = np.log(X.loc[:, self.variables])
            
        return X
    
    
class ReciprocalTransformer(BaseNumericalTransformer):
    """ Applies the reciprocal transformation 1 / x to the numerical variables.
    
    The trasnformer only works ith numerical variables, with non-zero values.
    
    Parameters
    ----------   
    variables : list
        The list of numerical variables that will be discretised. If none, it 
        defaults to all numerical type variables.
        
    Attributes
    ---------- 
    """
    
    def __init__(self, variables = None):
        
        self.variables = _define_variables(variables)
    
    
    def fit(self, X, y=None):
        """        
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
            raise ValueError("Some variables contain the value zero, can't apply reciprocal transformation")

        self.input_shape_ = X.shape             
        return self


    def transform(self, X):
        """ Applies the reciprocal 1 / x transformation.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with reciprocally transformed variables 
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the transformer')

        X = X.copy()
        X.loc[:, self.variables] = 1 / (X.loc[:, self.variables])
            
        return X
    
    
class ExponentialTransformer(BaseNumericalTransformer):
    """ Applies the exponential transformation to numerical variables.
    
    The transformer works only with numerical variables.
    
    Parameters
    ----------
    variables : list
        The list of numerical variables that will be discretised. If none, it 
        defaults to all numerical type variables.
        
    Attributes
    ----------  
    """
    
    def __init__(self, exp = 0.5, variables = None):
        
        if not isinstance(exp, float) and not isinstance(exp, int):
            raise ValueError('exp must be a float or an int')
            
        self.exp = exp
        self.variables = _define_variables(variables)
    
    
    def fit(self, X, y=None):
        """ Learns the numerical variables.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
        """

        super().fit(X, y)
        
        self.input_shape_ = X.shape             
        return self


    def transform(self, X):
        """ Applies the exponential transformation to the variables.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with exponentially transformed variables.
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the transformer')

        X = X.copy()
        X.loc[:, self.variables] = X.loc[:, self.variables]**self.exp
            
        return X
    
    
class BoxCoxTransformer(BaseNumericalTransformer):
    """ Applies the BoxCox transformation to the numerical variables.
    
    Transformer works only with numerical variables.
    
    Parameters
    ----------    
    variables : list
        The list of numerical variables that will be discretised. If none, it 
        defaults to all numerical type variables.
        
    Attributes
    ----------  
    """
    
    def __init__(self, variables = None):
        
        self.variables = _define_variables(variables)
    
    
    def fit(self, X, y=None):
        """ Learns the numerical variables. Captures the optimal lambda for
        the transformation.
        
        More detail on the functionality of BoxCox to come...
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Should be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
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
        """ Applies the BoxCox trasformation.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the transformed variables.
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_', 'lambda_dict'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the transformed')

        X = X.copy()
        for feature in self.variables:
            X[feature] = stats.boxcox(X[feature], lmbda=self.lambda_dict[feature]) 
            
        return X