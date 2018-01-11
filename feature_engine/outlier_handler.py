# Authors: Soledad Galli <solegalli1@gmail.com>

# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Windsorizer(BaseEstimator, TransformerMixin):
    """ Caps the maximum or minimum values of a variable
    
    This class works only with numerical variables
    
    The estimator first calculates the capping values
    for the desired features.
    
    The transformer caps the variables
    
    Parameters
    ----------
    distribution : str, default=gaussian 
        Desired distribution. 
        Can take 'gaussian' or 'skewed'.
        
    end : str, default=right
        Which tail to pick the value from.
        Can take 'left', 'right' or 'both'
        
    fold: int, default=3
        How far out to look for the value.
        Recommended values, 2 or 3 for Gaussian,
        1.5 or 3 for skewed.
        
    user_input : Boolean, default=False
        indicates if user will pass the capping values in a dictionary
        
    Attributes
    ----------
    variables_ : list
        The list of variables that want to be imputed
        passed to the method:`fit`
        
    imputer_dict_: dictionary
        The dictionary containg the values at end of tails to use
        for each variable to replace missing data
        
    """
    
    def __init__(self, distribution='gaussian', end='right', fold=3, user_input=False):
        
        if distribution not in ['gaussian', 'skewed']:
            raise ValueError("distribution takes only values 'gaussian' or 'skewed'")
            
        if end not in ['right', 'left', 'both']:
            raise ValueError("end takes only values 'right', 'left' or 'both'")
            
        if fold <=0 :
            raise ValueError("fold takes only positive numbers")
            
        self.distribution = distribution
        self.end = end
        self.fold = fold
        self.user_input = user_input

    def fit(self, X, y=None, variables = None, capping_max=None, capping_min=None):
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
        variables: list, default=None
            list of columns that should be transformed
        capping_max : dictionary, default=None
            user specified capping values on right side
        capping_min : dictionary, default=None
            user specified capping values on left side
        Returns
        -------
        self : object
            Returns self.
        """
        self.capping_max_ = {}
        self.capping_min_ = {}
        
        if self.user_input:
            if self.end in ['right', 'both'] and capping_max is None:
                raise AssertionError('Please provide the capping dictionary')
            elif self.end in ['left', 'both'] and capping_min is None:
                raise AssertionError('Please provide the capping dictionary')
            else:
                if self.end in ['right', 'both']:
                    self.capping_max_ = capping_max
                if self.end in ['left', 'both']:
                    self.capping_min_ = capping_min
            
        else:
            # Learn the variables to be capped
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            if not variables:
                # select all numerical variables
                variables = list(X.select_dtypes(include=numerics).columns)
            
            if self.end in ['right', 'both']:
                if self.distribution == 'gaussian':
                    self.capping_max_ = (X[variables].mean()+self.fold*X[variables].std()).to_dict()
                elif self.distribution == 'skewed':
                    IQR = X[variables].quantile(0.75) - X[variables].quantile(0.25)
                    self.capping_max_ = (X[variables].quantile(0.75) + (IQR * self.fold)).to_dict()
            
            if self.end in ['left', 'both']:
                if self.distribution == 'gaussian':
                    self.capping_min_ = (X[variables].mean()-self.fold*X[variables].std()).to_dict()
                elif self.distribution == 'skewed':
                    IQR = X[variables].quantile(0.75) - X[variables].quantile(0.25)
                    self.capping_min_ = (X[variables].quantile(0.25) - (IQR * self.fold)).to_dict()        
        
        self.input_shape_ = X.shape 
        return self

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
        check_is_fitted(self, ['capping_max_', 'capping_min_'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set '
                             ' used to fit the encoder')

        X = X.copy()
        for feature in self.capping_max_.keys():
            X[feature] = np.where(X[feature]>self.capping_max_[feature], self.capping_max_[feature], X[feature])
            
        for feature in self.capping_min_.keys():
            X[feature] = np.where(X[feature]<self.capping_min_[feature], self.capping_min_[feature], X[feature])
        
        return X

