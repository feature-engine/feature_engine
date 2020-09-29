# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
import scipy.stats as stats

from feature_engine.variable_manipulation import _define_variables
from feature_engine.base_transformers import BaseNumericalTransformer


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
