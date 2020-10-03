# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import pandas as pd
from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.variable_manipulation import _define_variables


class MeanEncoder(BaseCategoricalTransformer):
    """ 
    The MeanCategoricalEncoder() replaces categories by the mean value of the
    target for each category.
    
    For example in the variable colour, if the mean of the target for blue, red
    and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 0.5, red by 0.8
    and grey by 0.1.
    
    The encoder will encode only categorical variables (type 'object'). A list
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will find and encode all categorical variables
    (object type).
    
    The encoder first maps the categories to the numbers for each variable (fit).

    The encoder then transforms the categories to the mapped numbers (transform).
    
    Parameters
    ----------  
    
    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and select all object type variables.
    """

    def __init__(self, variables=None):
        self.variables = _define_variables(variables)

    def fit(self, X, y):
        """
        Learns the mean value of the target for each category of the variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to be encoded.

        y : pandas series
            The target.

        Attributes
        ----------

        encoder_dict_: dictionary
            The dictionary containing the {category: target mean} pairs used
            to replace categories in every variable.
        """

        X = self._check_fit_input_and_variables(X)

        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']

        self.encoder_dict_ = {}

        for var in self.variables:
            self.encoder_dict_[var] = temp.groupby(var)['target'].mean().to_dict()

        self._check_encoding_dictionary()

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
