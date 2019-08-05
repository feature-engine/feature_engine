# Authors: Soledad Galli <solegalli1@gmail.com>
# License: BSD 3 clause

import numpy as np
#import pandas as pd
#import warnings
import scipy.stats as stats 

from sklearn.utils.validation import check_is_fitted
from feature_engine.base_transformers import BaseNumericalTransformer, _define_variables


class LogTransformer(BaseNumericalTransformer):
    """
    The LogTransformer() applies the logarithmic transformation to numerical
    variables.
    
    The LogTransformer() only works with numerical non-negative values.
    
    The LogTransformer() will transform only numerical variables. 
    A list of variables can be passed as an argument. But, if no variables are 
    indicated, the transformer will automatically select and transform numerical 
    variables and ignore the rest.
        
    Parameters
    ----------
    
    variables : list, default=None
        The list of numerical variables to be transformed. If None, the transformer 
        will find and select all numerical type variables.
    """
    
    def __init__(self, variables = None):
        
        self.variables = _define_variables(variables)
    
    
    def fit(self, X, y=None):
        """
        Selects the numerical variables and determines whether the logarithm
        can be applied on selected variables (it checks if the variables are 
        all positive).
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
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
        """
        Transforms the variables using logarithm.
        
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
    """
    The ReciprocalTransformer() applies the reciprocal transformation 1 / x
    to the numerical variables.
    
    The ReciprocalTransformer() only works with numerical variables with non-zero
    values.
    
    The ReciprocalTransformer() will transform only numerical variables. 
    A list of variables can be passed as an argument. But, if no variables are 
    indicated, the transformer will automatically select and transform numerical 
    variables and ignore the rest.
    
    Parameters
    ----------   
    
    variables : list, default=None
        The list of numerical variables that will be transformed. If None, the 
        transformer will automaticalle find and select all numerical type variables.
    """
    
    def __init__(self, variables = None):
        
        self.variables = _define_variables(variables)
    
    
    def fit(self, X, y=None):
        """        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
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
        """
        Applies the reciprocal 1 / x transformation.
        
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
        X.loc[:, self.variables] = np.reciprocal(X.loc[:, self.variables])
            
        return X
    
    
class PowerTransformer(BaseNumericalTransformer):
    """
    The PowerTransformer() applies power or exponential transformations to
    numerical variables.
    
    The PowerTransformer() works only with numerical variables.
    
    The PowerTransformer() will transform only numerical variables. 
    A list of variables can be passed as an argument. But, if no variables are 
    indicated, the transformer will automatically select and transform numerical 
    variables and ignore the rest.
    
    Parameters
    ----------
    
    variables : list, default=None
        The list of numerical variables that will be transformed. If None, the 
        transformer will automatically find and select all numerical type variables.
        
    exp : float, default=0.5
        The power (or exponent).
    """
    
    def __init__(self, exp = 0.5, variables = None):
        
        if not isinstance(exp, float) and not isinstance(exp, int):
            raise ValueError('exp must be a float or an int')
            
        self.exp = exp
        self.variables = _define_variables(variables)
    
    
    def fit(self, X, y=None):
        """
        Learns the numerical variables that should be transformed
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
        """

        super().fit(X, y)
        
        self.input_shape_ = X.shape             
        return self


    def transform(self, X):
        """
        Applies the power transformation to the variables.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with power transformed variables.
        """
        
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the transformer')

        X = X.copy()
        X.loc[:, self.variables] = np.power(X.loc[:, self.variables], self.exp)
            
        return X
    
    
class BoxCoxTransformer(BaseNumericalTransformer):
    """
    The BoxCoxTransformer() applies the BoxCox transformation to the numerical
    variables.
    
    The BoxCox transformation implemented by this transformer is that of
    SciPy.stats:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
    
    The BoxCoxTransformer() works only with numerical positive variables (>=0, 
    the transformer also works for zero values).

    The BoxCoxTransformer() will transform only numerical variables. 
    A list of variables can be passed as an argument. But, if no variables are 
    indicated, the transformer will automatically select and transform numerical 
    variables and ignore the rest.
    
    Parameters
    ----------    
    
    variables : list, default=None
        The list of numerical variables that will be transformed. If None, the 
        transformer will automaticalle find and select all numerical type variables.
        
    Attributes
    ----------
    
    lamda_dict_ : dictionary
        The dictionary containing the {variable: best exponent for the BoxCox
        transfomration} pairs.
    """
    
    def __init__(self, variables = None):
        
        self.variables = _define_variables(variables)
    
    
    def fit(self, X, y=None):
        """
        Learns the numerical variables. Captures the optimal lambda for
        the BoxCox transformation.
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
        """

        super().fit(X, y)
        
        if (X[self.variables]<0).all().all():
            raise ValueError("Some variables contain negative values, try Yeo-Johnson transformation instead")
        
        self.lambda_dict_ = {}
        
        for var in self.variables:
            _, self.lambda_dict_[var] = stats.boxcox(X[var]) 
            
        self.input_shape_ = X.shape
        
        return self


    def transform(self, X):
        """
        Applies the BoxCox transformation.
        
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
        check_is_fitted(self, ['input_shape_', 'lambda_dict_'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the transformer')

        X = X.copy()
        for feature in self.variables:
            X[feature] = stats.boxcox(X[feature], lmbda=self.lambda_dict_[feature]) 
            
        return X


class YeoJohnsonTransformer(BaseNumericalTransformer):
    """
    The YeoJohnsonTransformer() applies the Yeo-Johnson transformation to the
    numerical variables.
    
    The Yeo-Johnson transformation implemented by this transformer is that of
    SciPy.stats:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html
    
    The YeoJohnsonTransformer() works only with numerical variables.

    The YeoJohnsonTransformer() will transform only numerical variables.
    A list of variables can be passed as an argument. But, if no variables are 
    indicated, the transformer will automatically select and transform numerical 
    variables and ignore the rest.
    
    Parameters
    ----------
    
    variables : list, default=None
        The list of numerical variables that will be trasnformed. If None, the 
        transformer will automatically find and select all numerical type variables.
        
    Attributes
    ----------
    
    lamda_dict_ : dictionary
        The dictionary containing the {variable: best lambda for the Yeo-Johnson
        transfomration} pairs.
    """
    
    def __init__(self, variables = None):
        
        self.variables = _define_variables(variables)
    
    
    def fit(self, X, y=None):
        """
        Learns the numerical variables. Captures the optimal lambda for
        the transformation.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this transformer, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
        """

        super().fit(X, y)
        
        self.lambda_dict_ = {}
        
        for var in self.variables:
        	X[var] = X[var].astype('float') # to avoid NumPy error
        	_, self.lambda_dict_[var] = stats.yeojohnson(X[var])
            
        self.input_shape_ = X.shape
        
        return self


    def transform(self, X):
        """
        Applies the BoxCox trasformation.
        
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
        check_is_fitted(self, ['input_shape_', 'lambda_dict_'])
        
        if X.shape[1] != self.input_shape_[1]:
            raise ValueError('Number of columns in dataset is different from training set used to fit the transformer')

        X = X.copy()
        for feature in self.variables:
            X[feature] = stats.yeojohnson(X[feature], lmbda=self.lambda_dict_[feature]) 
            
        return X