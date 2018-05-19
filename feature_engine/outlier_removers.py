# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted



class BaseOutlierRemover(BaseEstimator, TransformerMixin):
    
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
            

class Windsorizer(BaseOutlierRemover):
    """ Caps the maximum or minimum values of a variable by finding the values
    according to the variable distribution.
    
    This class works only with numerical variables.
    
    The estimator first calculates the capping values at the end of the
    distribution for the indicated features. The values at the end of the
    distribution can be given by the Gaussian limits or the quantile limits.
    
    Gaussian limits:
        right tail: mean + 3* std
        left tail: mean - 3* std
        
    Quantile limits:
        right tail: 75th Quantile + 3* IQR
        left tail:  25th quantile - 3* IQR
        
        where IQR is the inter-quantal range.
        
    You can select to tune how far out you place your missing values by tuning
    the number by which you multiply the std or the IQR, using the parameter
    fold.
    
    The transformer caps the variables.
    
    Parameters
    ----------
    distribution : str, default=gaussian 
        Desired distribution. 
        Can take 'gaussian' or 'skewed'.
        
    end : str, default=right
        Whether to cap outliers on the right, left or both tails.
        Can take 'left', 'right' or 'both'
        
    fold: int, default=3
        How far out to to placce the capping value. 
        Recommended values, 2 or 3 for Gaussian, 1.5 or 3 for skewed.
        
    variables : list
        The list of variables for which the outliers will be capped.
             
    Attributes
    ----------       
    outlier_capper_dict_: dictionary
        The dictionary containg the values at the end of tails to use
        for each variable to capp the outliers.
        
    """
    
    def __init__(self, distribution='gaussian', tail='right', fold=3, variables = None):
        
        if distribution not in ['gaussian', 'skewed']:
            raise ValueError("distribution takes only values 'gaussian' or 'skewed'")
            
        if tail not in ['right', 'left', 'both']:
            raise ValueError("end takes only values 'right', 'left' or 'both'")
            
        if fold <=0 :
            raise ValueError("fold takes only positive numbers")
            
        self.distribution = distribution
        self.tail = tail
        self.fold = fold
        self.variables = variables

    
    def fit(self, X, y=None):
        """ Learns the values that should be used to replace
        outliers in each variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can contain all the variables, not necessarily only those to remove
            outliers
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.

        """
        super().fit(X, y)
        
        self.right_tail_caps_ = {}
        self.left_tail_caps_ = {}
        
        # estimate the end values
        if self.tail in ['right', 'both']:
            if self.distribution == 'gaussian':
                self.right_tail_caps_ = (X[self.variables].mean()+self.fold*X[self.variables].std()).to_dict()
                
            elif self.distribution == 'skewed':
                IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
                self.right_tail_caps_ = (X[self.variables].quantile(0.75) + (IQR * self.fold)).to_dict()
        
        elif self.tail in ['left', 'both']:
            if self.distribution == 'gaussian':
                self.left_tail_caps_ = (X[self.variables].mean()-self.fold*X[self.variable_].std()).to_dict()
                
            elif self.distribution == 'skewed':
                IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
                self.left_tail_caps_ = (X[self.variables].quantile(0.25) - (IQR * self.fold)).to_dict()        
        
        self.input_shape_ = X.shape  
              
        return self



class ArbitraryOutlierCapper(BaseOutlierRemover):
    """ Caps the maximum or minimum values of a variable by an arbitrary value
    entered by the user.
    
    This class works only with numerical variables.
    
    The user needs to provide the maximum or minimum values that will be used
    to cap the variable in a dictionary {feature:capping value}
       
    The transformer caps the variables.
    
    Parameters
    ----------
    variables: list, default=None
        list of columns that should be transformed
    capping_max : dictionary, default=None
        user specified capping values on right side
    capping_min : dictionary, default=None
        user specified capping values on left side
             
    Attributes
    ----------       
    outlier_capper_dict_: dictionary
        The dictionary containg the values at the end of tails to use
        for each variable to capp the outliers.
        
    """


    def __init__(self, max_capping_dict = None, min_capping_dict = None, variables = None):
        
        if not  max_capping_dict and not min_capping_dict:
            raise ValueError("Please provide at least 1 dictionary with the capping values per variable")
        
        if not max_capping_dict:
            self.right_tail_caps_ = {}
            
        if not min_capping_dict:
            self.left_tail_caps_ = {}
        
        if max_capping_dict and isinstance(max_capping_dict, dict):
            self.right_tail_caps_ = max_capping_dict
        else:
            raise ValueError("max_capping_dict should be a dictionary")
            
        if min_capping_dict and isinstance(min_capping_dict, dict):
            self.left_tail_caps_ = min_capping_dict
        else:
            raise ValueError("min_capping_dict should be a dictionary")    
        
        self.variables = variables
        
    
    def fit(self, X, y=None):
        """ Learns the values that should be used to replace
        outliers in each variable.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can contain all the variables, not necessarily only those to remove
            outliers
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking.

        """
        super().fit(X, y)   

        self.input_shape_ = X.shape  
              
        return self