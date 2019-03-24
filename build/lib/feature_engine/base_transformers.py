# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause


import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _define_variables(variables):
    
    if not variables or isinstance(variables, list):
       variables = variables
    else:
       variables = [variables]
    return variables

            
class BaseNumericalTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        
        #numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        if not self.variables:
            # select all numerical variables
            #self.variables = list(X.select_dtypes(include=numerics).columns)  
            self.variables = list(X.select_dtypes(include='number').columns)
        else:
            # variables indicated by user
            #if len(X[self.variables].select_dtypes(exclude=numerics).columns) != 0:
            if len(X[self.variables].select_dtypes(exclude='number').columns) != 0:
               raise ValueError("Some of the selected variables are NOT numerical. Please cast them as numerical before fitting the imputer")
            
            self.variables = self.variables



class BaseNumericalImputer(BaseNumericalTransformer):
    
    def transform(self, X):
        """ Replaces missing data with the calculated parameters
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        
        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing NO missing values for the selected
            variables
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['imputer_dict_'])
        
        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise TypeError('Number of columns in dataset is different from training set used to fit the imputer')
        
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        
        return X



class BaseDiscretiser(BaseNumericalTransformer):
    
    def transform(self, X):
        """ Discretises the variables in the selected bins.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with discrete / binned variables 
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['binner_dict'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')

        X = X.copy()
        for feature in self.variables:
            X[feature] = pd.cut(X[feature], self.binner_dict[feature], labels=False)
            
        return X



class BaseOutlierRemover(BaseNumericalTransformer):          
    
    def transform(self, X):
        """ Caps variables with the calculated parameters
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with capped values for the selected
            variables
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['left_tail_caps_', 'right_tail_caps_'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')

        X = X.copy()
        for feature in self.right_tail_caps_.keys():
            X[feature] = np.where(X[feature]>self.right_tail_caps_[feature], self.right_tail_caps_[feature], X[feature])
            
        for feature in self.left_tail_caps_.keys():
            X[feature] = np.where(X[feature]<self.left_tail_caps_[feature], self.left_tail_caps_[feature], X[feature])
        
        return X            



class BaseCategoricalTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        
        if not self.variables:
            # select all categorical variables
            self.variables = [var for var in X.columns if X[var].dtypes=='O']
        else:
            # variables indicated by user
            if len(X[self.variables].select_dtypes(exclude='O').columns) != 0:
#            for var in self.variables:
#                if X[var].dtypes != 'O':
                raise TypeError("variable {} is not of type object, check that all indicated variables are of type object")
            self.variables = self.variables
        
        return self.variables



class BaseCategoricalEncoder(BaseCategoricalTransformer):  

    def transform(self, X):
        """ Replaces categories with the estimated numbers.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features].
            The input samples.
        
        Returns
        -------
        X_transformed : pandas dataframe of shape = [n_samples, n_features].
            The dataframe containing categories replaced by numbers.
       """
        # Check that the method fit has been called
        check_is_fitted(self, ['encoder_dict_'])
        
        # Check that the input is of the same shape as the training set passed
        # during fit.
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from train set used to fit the encoder')
        
        # encode labels     
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])
            
            if X[feature].isnull().sum() > 0:
                warnings.warn("NaN values were introduced by the encoder due to labels in variable {} not present in the training set. Try using the RareLabelCategoricalEncoder.".format(feature) )       
        return X

