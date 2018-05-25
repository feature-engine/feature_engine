# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
#import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score



class BaseDiscretiser(BaseEstimator, TransformerMixin):
    
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
        check_is_fitted(self, ['binner_dict'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')

        X = X.copy()
        for feature in self.variables:
            X[feature] = pd.cut(X[feature], self.binner_dict[feature], labels=False)
            
        return X
    


class EqualFrequencyDiscretiser(BaseDiscretiser):
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
    
    def __init__(self, q = 10, variables = None):
        
        if not isinstance(q, int):
            raise ValueError('q must be an integer')
            
        self.q = q
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
        
        self.binner_dict = {}
        
        for var in self.variables:
            tmp, bins = pd.qcut(x=X[var], q=self.q, retbins=True, duplicates='drop')
            
            # Prepend/Append infinities to accomodate outliers
            bins = list(bins)
            bins[0]= float("-inf")
            bins[len(bins)-1] = float("inf")
            self.binner_dict[var] = bins
            
        self.input_shape_ = X.shape  
           
        return self
    


class EqualWidthDiscretiser(BaseDiscretiser):
    """ Divides the numerical variable values into equal width buckets.
    Equal width buckets are equi-distant buckets / intervals. All intervals are
    of the same size. Number of observations per bucket may vary.
    
    The buckets are estimated using pandas.cut. The number of buckets / bins
    in which the variable should be divided must be indicated by the
    user.
    
    The Discretiser will binnarise only numerical variables (type 'object'). A 
    list of variables can be passed as an argument. If no variables are indicated,
    the discretiser will only binnarise numerical variables and ignore the rest.
    
    The discretiser first finds the boundaries for the buckets for each variable
    (fit).
    
    The transformer sorts the values into the buckets (transform).
    
    Parameters
    ----------
    bins : int, default=10
        Desired number of equal widht buckets / bins.
    
    variables : list
        The list of numerical variables that will be discretised. If none, it 
        defaults to all numerical type variables.
        
    Attributes
    ----------
    binner_dict_: dictionary
        The dictionary containing the bucket boundaries: variable pairs used
        to binnarise / discretise variable    
    """
    
    def __init__(self, bins = 10, variables = None):
        
        if not isinstance(bins, int):
            raise ValueError('q must be an integer')
            
        self.bins = bins
        self.variables = variables
    
    
    def fit(self, X, y=None):
        """ Learns the boundaries of the equal widht buckets / bins for each
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
        
        self.binner_dict = {}
        
        for var in self.variables:
            tmp, bins = pd.cut(x=X[var], bins=self.bins, retbins=True, duplicates='drop')
            
            # Prepend/Append infinities
            bins = list(bins)
            bins[0]= float("-inf")
            bins[len(bins)-1] = float("inf")
            self.binner_dict[var] = bins
            
        self.input_shape_ = X.shape  
           
        return self       



class DecisionTreeDiscretiser(BaseDiscretiser):
    """ Divides the numerical variable values into equal width buckets.
    Equal width buckets are equi-distant buckets / intervals. All intervals are
    of the same size. Number of observations per bucket may vary.
    
    The buckets are estimated using pandas.cut. The number of buckets / bins
    in which the variable should be divided must be indicated by the
    user.
    
    The Discretiser will binnarise only numerical variables (type 'object'). A 
    list of variables can be passed as an argument. If no variables are indicated,
    the discretiser will only binnarise numerical variables and ignore the rest.
    
    The discretiser first finds the boundaries for the buckets for each variable
    (fit).
    
    The transformer sorts the values into the buckets (transform).
    
    Parameters
    ----------
    bins : int, default=10
        Desired number of equal widht buckets / bins.
    
    variables : list
        The list of numerical variables that will be discretised. If none, it 
        defaults to all numerical type variables.
        
    Attributes
    ----------
    binner_dict_: dictionary
        The dictionary containing the bucket boundaries: variable pairs used
        to binnarise / discretise variable    
    """
    
    def __init__(self, cv = 3, scoring='neg_mean_squared_error', variables = None):
        
        self.cv = cv
        self.scoring = scoring
        self.variables = variables
    
    
    def fit(self, X, y):
        """ Learns the boundaries of the equal widht buckets / bins for each
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

        if y is None:
            raise ValueError('Please provide a target (y) for this encoding method')
            
        super().fit(X, y)
        
        self.binner_dict = {}
        
        for var in self.variables:
            
            score_ls = [] 
            for tree_depth in [1,2,3,4]:
                # call the model
                if len(y)==2:
                    tree_model = DecisionTreeClassifier(max_depth=tree_depth)
                else:
                    tree_model = DecisionTreeRegressor(max_depth=tree_depth)
        
                # train the model using 3 fold cross validation
                scores = cross_val_score(tree_model, X[var].to_frame(), y,
                                         cv=self.cv, scoring=self.scoring)
                
                score_ls.append(np.mean(scores))
        
            # find depth with smallest mse
            depth = [1,2,3,4][np.argmax(score_ls)]
        
            # transform the variable using the tree
            if len(y) == 2:
                tree_model = DecisionTreeClassifier(max_depth=depth)
            else:
                tree_model = DecisionTreeRegressor(max_depth=depth)
                
            tree_model.fit(X[var].to_frame(), y)
            
            self.binner_dict[var] = tree_model

           
        self.input_shape_ = X.shape
        if len(y) == 2:
            self.task = 'Classification'
        else:
            self.task = 'Regression'
           
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
        check_is_fitted(self, ['binner_dict'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the encoder')

        X = X.copy()
        for feature in self.variables:
            if self.task == 'Classification':
                tmp = self.binner_dict[feature].predict_proba(X[feature].to_frame())
                X[feature] = tmp[:,1]
            else:
                X[feature] = self.binner_dict[feature].predict(X[feature].to_frame())
                
        return X    