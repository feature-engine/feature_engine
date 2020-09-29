# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df, _check_contains_na
from feature_engine.variable_manipulation import _find_categorical_variables, _define_variables
from feature_engine.base_transformers import BaseCategoricalTransformer


def _check_encoding_dictionary(dictionary):
    # check that there is a dictionary with category to number pairs
    if len(dictionary) == 0:
        raise ValueError('Encoder could not be fitted. Check the parameters and the variables '
                         'in your dataframe.')
    return dictionary


class OrdinalEncoder(BaseCategoricalTransformer):
    """ 
    The OrdinalCategoricalEncoder() replaces categories by ordinal numbers 
    (0, 1, 2, 3, etc). The numbers can be ordered based on the mean of the target
    per category, or assigned arbitrarily.
    
    Ordered ordinal encoding:  for the variable colour, if the mean of the target
    for blue, red and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 1,
    red by 2 and grey by 0.
    
    Arbitrary ordinal encoding: the numbers will be assigned arbitrarily to the
    categories, on a first seen first served basis.
    
    The encoder will encode only categorical variables (type 'object'). A list
    of variables can be passed as an argument. If no variables are passed, the
    encoder will find and encode all categorical variables (type 'object').
    
    The encoder first maps the categories to the numbers for each variable (fit).

    The encoder then transforms the categories to the mapped numbers (transform).
    
    Parameters
    ----------
    
    encoding_method : str, default='ordered' 
        Desired method of encoding.

        'ordered': the categories are numbered in ascending order according to
        the target mean value per category.

        'arbitrary' : categories are numbered arbitrarily.
        
    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and select all object type variables.
        
    Attributes
    ----------
    
    encoder_dict_: dictionary
        The dictionary containing the {category: ordinal number} pairs for
        every variable.
    """

    def __init__(self, encoding_method='ordered', variables=None):

        if encoding_method not in ['ordered', 'arbitrary']:
            raise ValueError("encoding_method takes only values 'ordered' and 'arbitrary'")

        self.encoding_method = encoding_method
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """ Learns the numbers to be used to replace the categories in each
        variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to be
            encoded.

        y : pandas series, default=None
            The Target. Can be None if encoding_method = 'arbitrary'.
            Otherwise, y needs to be passed when fitting the transformer.
       
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find categorical variables or check that those entered by the user
        # are of type object
        self.variables = _find_categorical_variables(X, self.variables)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        # join target to predictor variables
        if self.encoding_method == 'ordered':
            if y is None:
                raise ValueError('Please provide a target y for this encoding method')

            temp = pd.concat([X, y], axis=1)
            temp.columns = list(X.columns) + ['target']

        # find mappings
        self.encoder_dict_ = {}

        for var in self.variables:

            if self.encoding_method == 'ordered':
                t = temp.groupby([var])['target'].mean().sort_values(ascending=True).index

            elif self.encoding_method == 'arbitrary':
                t = X[var].unique()

            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        self.encoder_dict_ = _check_encoding_dictionary(self.encoder_dict_)

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseCategoricalTransformer.transform.__doc__

    def inverse_transform(self, X):
        X = super().inverse_transform(X)
        return X

    inverse_transform.__doc__ = BaseCategoricalTransformer.inverse_transform.__doc__


