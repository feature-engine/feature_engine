# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

#import numpy as np

#from sklearn.utils.validation import check_is_fitted
from feature_engine.base_transformers import BaseOutlierRemover, _define_variables
            

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
        How far out to to place the capping value. 
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
        self.variables = _define_variables(variables)

    
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
        
        if self.tail in ['left', 'both']:
            if self.distribution == 'gaussian':
                self.left_tail_caps_ = (X[self.variables].mean()-self.fold*X[self.variables].std()).to_dict()
                
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


    def __init__(self, max_capping_dict = None, min_capping_dict = None):
        
        self.max_capping_dict = max_capping_dict
        self.min_capping_dict = min_capping_dict
        
        if not  max_capping_dict and not min_capping_dict:
            raise ValueError("Please provide at least 1 dictionary with the capping values per variable")
        
        if not max_capping_dict:
            self.right_tail_caps_ = {}
        else:
            if isinstance(max_capping_dict, dict):
                self.right_tail_caps_ = max_capping_dict
            else:
                raise ValueError("max_capping_dict should be a dictionary")
            
        if not min_capping_dict:
            self.left_tail_caps_ = {}
        else:
            if isinstance(min_capping_dict, dict):
                self.left_tail_caps_ = min_capping_dict
            else:
                raise ValueError("min_capping_dict should be a dictionary")    
        
        self.variables = [x for x in self.right_tail_caps_.keys()]
        self.variables = self.variables + [x for x in self.left_tail_caps_.keys()]
        
        
    def fit(self, X, y=None):
        """ Learns the numerical variables form the dataframe.
        
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