# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np

from feature_engine.variable_manipulation import _define_variables
from feature_engine.base_transformers import BaseNumericalTransformer


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


