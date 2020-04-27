# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
import scipy.stats as stats

from feature_engine.variable_manipulation import _define_variables
from feature_engine.base_transformers import BaseNumericalTransformer


class LogTransformer(BaseNumericalTransformer):
    """
    The LogTransformer() applies the natural logarithm or the base 10
    logarithm to numerical variables. The natural logarithm is logarithm in base e.
    
    The LogTransformer() only works with numerical non-negative values. If the variable
    contains a zero or a negative value, the transformer will return an error.
    
    A list of variables can be passed as an argument. Alternatively, the transformer
    will automatically select and transform all variables of type numeric.
        
    Parameters
    ----------

    base: string, default='e'
        Indicates if the natural or base 10 logarithm should be applied. Can take values
        'e' or '10'.

    variables : list, default=None
        The list of numerical variables to be transformed. If None, the transformer 
        will find and select all numerical variables.
    """

    def __init__(self, base='e', variables=None):

        if base not in ['e', '10']:
            raise ValueError("base can take only '10' or 'e' as values")

        self.variables = _define_variables(variables)
        self.base = base

    def fit(self, X, y=None):
        """
        Selects the numerical variables and determines whether the logarithm
        can be applied on the selected variables (it checks if the variables
        are all positive).
        
        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.

        y : None
            y is not needed in this transformer. You can pass y or None.
        """
        # check input dataframe
        X = super().fit(X)

        # check contains zero or negative values
        if (X[self.variables] <= 0).any().any():
            raise ValueError("Some variables contain zero or negative values, can't apply log")

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Transforms the variables using logarithm.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The log transformed dataframe. 
        """
        # check input dataframe and if class was fitted
        X = super().transform(X)

        # check contains zero or negative values
        if (X[self.variables] <= 0).any().any():
            raise ValueError("Some variables contain zero or negative values, can't apply log")

        # transform
        if self.base == 'e':
            X.loc[:, self.variables] = np.log(X.loc[:, self.variables])
        elif self.base == '10':
            X.loc[:, self.variables] = np.log10(X.loc[:, self.variables])

        return X


class ReciprocalTransformer(BaseNumericalTransformer):
    """
    The ReciprocalTransformer() applies the reciprocal transformation 1 / x
    to numerical variables.
    
    The ReciprocalTransformer() only works with numerical variables with non-zero
    values. If a variable contains the value 0, the transformer will raise an error.
    
    A list of variables can be passed as an argument. Alternatively, the
    transformer will automatically select and transform all numerical
    variables.
    
    Parameters
    ----------   
    
    variables : list, default=None
        The list of numerical variables that will be transformed. If None, the 
        transformer will automatically find and select all numerical variables.
    """

    def __init__(self, variables=None):

        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.

        y : None
            y is not needed in this encoder. You can pass y or None.
        """
        # check input dataframe
        X = super().fit(X)

        # check if the variables contain the value 0
        if (X[self.variables] == 0).any().any():
            raise ValueError("Some variables contain the value zero, can't apply reciprocal transformation")

        self.input_shape_ = X.shape
        return self

    def transform(self, X):
        """
        Applies the reciprocal 1 / x transformation.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with reciprocally transformed variables.
        """
        # check input dataframe and if class was fitted
        X = super().transform(X)

        # check if the variables contain the value 0
        if (X[self.variables] == 0).any().any():
            raise ValueError("Some variables contain the value zero, can't apply reciprocal transformation")

        # transform
        # for some reason reciprocal does not work with integers
        X.loc[:, self.variables] = X.loc[:, self.variables].astype('float')
        X.loc[:, self.variables] = np.reciprocal(X.loc[:, self.variables])

        return X


class PowerTransformer(BaseNumericalTransformer):
    """
    The PowerTransformer() applies power or exponential transformations to
    numerical variables.
    
    The PowerTransformer() works only with numerical variables.
    
    A list of variables can be passed as an argument. Alternatively, the
    transformer will automatically select and transform all numerical
    variables.
    
    Parameters
    ----------
    
    variables : list, default=None
        The list of numerical variables that will be transformed. If None, the 
        transformer will automatically find and select all numerical variables.
        
    exp : float or int, default=0.5
        The power (or exponent).
    """

    def __init__(self, exp=0.5, variables=None):

        if not isinstance(exp, float) and not isinstance(exp, int):
            raise ValueError('exp must be a float or an int')

        self.exp = exp
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.

        y : None
            y is not needed in this transformer. You can pass y or None.
        """
        # check input dataframe
        X = super().fit(X)

        self.input_shape_ = X.shape
        return self

    def transform(self, X):
        """
        Applies the power transformation to the variables.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the power transformed variables.
        """
        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform
        X.loc[:, self.variables] = np.power(X.loc[:, self.variables], self.exp)

        return X


class BoxCoxTransformer(BaseNumericalTransformer):
    """
    The BoxCoxTransformer() applies the BoxCox transformation to numerical
    variables.
    
    The BoxCox transformation implemented by this transformer is that of
    SciPy.stats:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
    
    The BoxCoxTransformer() works only with numerical positive variables (>=0, 
    the transformer also works for zero values).

    A list of variables can be passed as an argument. Alternatively, the
    transformer will automatically select and transform all numerical
    variables.
    
    Parameters
    ----------    
    
    variables : list, default=None
        The list of numerical variables that will be transformed. If None, the 
        transformer will automatically find and select all numerical variables.
        
    Attributes
    ----------
    
    lamda_dict_ : dictionary
        The dictionary containing the {variable: best exponent for the BoxCox
        transfomration} pairs. These are determined automatically.
    """

    def __init__(self, variables=None):

        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Learns the optimal lambda for the BoxCox transformation.
        
        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.

        y : None
            y is not needed in this transformer. You can pass y or None.
        """
        # check input dataframe
        X = super().fit(X)

        if (X[self.variables] < 0).any().any():
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
            The data to be transformed.

        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the transformed variables.
        """
        # check input dataframe and if class was fitted
        X = super().transform(X)

        # check if variable contains negative numbers
        if (X[self.variables] < 0).any().any():
            raise ValueError("Some variables contain negative values, try Yeo-Johnson transformation instead")

        # transform
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

    A list of variables can be passed as an argument. Alternatively, the
    transformer will automatically select and transform all numerical
    variables.
    
    Parameters
    ----------
    
    variables : list, default=None
        The list of numerical variables that will be transformed. If None, the
        transformer will automatically find and select all numerical variables.
        
    Attributes
    ----------
    
    lamda_dict_ : dictionary
        The dictionary containing the {variable: best lambda for the Yeo-Johnson
        transformation} pairs.
    """

    def __init__(self, variables=None):

        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Learns the optimal lambda for the Yeo-Johnson transformation.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.
        y : None
            y is not needed in this transformer. You can pass y or None.
        """
        # check input dataframe
        X = super().fit(X)

        self.lambda_dict_ = {}

        # to avoid NumPy error
        X[self.variables] = X[self.variables].astype('float')

        for var in self.variables:
            _, self.lambda_dict_[var] = stats.yeojohnson(X[var])

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Applies the Yeo-Johnson transformation.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the transformed variables.
        """
        # check input dataframe and if class was fitted

        X = super().transform(X)
        for feature in self.variables:
            X[feature] = stats.yeojohnson(X[feature], lmbda=self.lambda_dict_[feature])

        return X
