# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

from feature_engine.base_transformers import BaseOutlierRemover, _define_variables
            

class Winsorizer(BaseOutlierRemover):
    """
    The Winsorizer() caps maximum and / or minimum values of a variable.
    
    The Winsorizer() works only with numerical variables. A list of variables can
    be indicated. If no list of variable names is passed, the Winsorizer() will
    find and select all numerical variables seen in the train set.
    
    The Winsorizer() first calculates the capping values at the end of the
    distribution for the indicated features. The values at the end of the
    distribution are calculated wither using a Gaussian approximation or the 
    inter-quantile range proximity rule.
    
    Gaussian limits:
        right tail: mean + 3* std
        left tail: mean - 3* std
        
    IQR limits:
        right tail: 75th Quantile + 3* IQR
        left tail:  25th quantile - 3* IQR
        
    where IQR is the inter-quantal range: 75th Quantile - 25th Quantile.
        
    You can select to tune how far out to cap your maximum or minimum values by
    tuning the number by which you multiply the std or the IQR, using the parameter
    'fold'.
    
    The transformer first finds the values at one or both tails of the distributions
    at which it will cap the variables (fit).
    
    The transformer then caps the variables (transform).
    
    Parameters
    ----------
    
    distribution : str, default=gaussian 
        Desired distribution. Can take 'gaussian' or 'skewed'. If 'gaussian' the
        transformer will find the maximum and / or minimum values to cap the
        variables using the Gaussian approximation. If 'skewed' the transformer
        will find the boundaries using the IQR proximity rule.
        
    end : str, default=right
        Whether to cap outliers on the right, left or both tails of the distribution.
        Can take 'left', 'right' or 'both'.
        
    fold: int, default=3
        How far out to to place the capping value. The number that will multiply
        the std or IQR to calculate the capping values. Recommended values, 2 
        or 3 for the gaussian approximation, or 1.5 or 3 for the IQR proximity 
        rule.
        
    variables : list, default=None
        The list of variables for which the outliers will be capped. If None, 
        the transformer will find and select all numerical variables.
             
    Attributes
    ----------
    
    outlier_capper_dict_: dictionary
        The dictionary containg the values at the end of the distributions to 
        use to cap each variable.
    """
    
    def __init__(self, distribution='gaussian', tail='right', fold=3, variables = None):
        
        if distribution not in ['gaussian', 'skewed', 'quantiles']:
            raise ValueError("distribution takes only values 'gaussian', 'skewed' or 'quantiles'")
            
        if tail not in ['right', 'left', 'both']:
            raise ValueError("tail takes only values 'right', 'left' or 'both'")
            
        if fold <=0 :
            raise ValueError("fold takes only positive numbers")
            
        self.distribution = distribution
        self.tail = tail
        self.fold = fold
        self.variables = _define_variables(variables)

    
    def fit(self, X, y=None):
        """ 
        Learns the values that should be used to replace outliers in each variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can contain all the variables
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
                
            elif self.distribution == 'quantiles':
                self.right_tail_caps_ = X[self.variables].quantile(0.95).to_dict()
        
        if self.tail in ['left', 'both']:
            if self.distribution == 'gaussian':
                self.left_tail_caps_ = (X[self.variables].mean()-self.fold*X[self.variables].std()).to_dict()
                
            elif self.distribution == 'skewed':
                IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
                self.left_tail_caps_ = (X[self.variables].quantile(0.25) - (IQR * self.fold)).to_dict()
                
            elif self.distribution == 'quantiles':
                self.left_tail_caps_ = X[self.variables].quantile(0.05).to_dict()
        
        self.input_shape_ = X.shape  
              
        return self



class ArbitraryOutlierCapper(BaseOutlierRemover):
    """ 
    The ArbitraryOutlierCapper() caps the maximum or minimum values of a variable
    by an arbitrary value indicated by the user.
       
    The user needs to provide the maximum or minimum values that will be used
    to cap each indicated variable in a dictionary {feature:capping value}
       
    The transformer caps the variables.
    
    Parameters
    ----------
    
    capping_max : dictionary, default=None
        user specified capping values on right tail of the distribution (maximum
        values).
    capping_min : dictionary, default=None
        user specified capping values on left tail of the distribution (minimum
        values).
        
    """


    def __init__(self, max_capping_dict = None, min_capping_dict = None):
               
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
        """
        
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