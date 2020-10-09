# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd

from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.variable_manipulation import _define_variables


class WoEEncoder(BaseCategoricalTransformer):
    """ 
    The WoERatioCategoricalEncoder() replaces categories by the weight of evidence.
    
    The weight of evidence is given by: np.log(P(X=xj|Y = 1)/P(X=xj|Y=0))
        
    Note: This categorical encoding is exclusive for binary classification.
    
    For details on the calculation of the weight of evidence visit:
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    
    Note: the division by 0 is not defined and the log(0) is not defined.
    Thus, if p(0) = 0 for the ratio encoder, or either p(0) = 0 or p(1) = 0 for
    woe or log_ratio, in any of the variables, the encoder will return an error.
       
    The encoder will encode only categorical variables (type 'object'). A list
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will find and encode all categorical variables
    (object type).
    
    The encoder first maps the categories to the weight of evidence for variable (fit).

    The encoder then transforms the categories into the mapped numbers (transform).
        
    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and select all object type variables.
    """

    def __init__(self, variables=None):

        self.variables = _define_variables(variables)

    def fit(self, X, y):
        """
        Learns the numbers that should be used to replace the categories in each
        variable. That is the WoE.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y : pandas series.
            Target, must be binary [0,1].

        Attributes
        ----------

        encoder_dict_: dictionary
            The dictionary containing the {category: WoE} pairs per variable.
        """

        X = self._check_fit_input_and_variables(X)

        # check that y is binary
        if any(x for x in y.unique() if x not in [0, 1]):
            raise ValueError("This encoder is only designed for binary classification, values of y can be only 0 or 1")

        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']

        self.encoder_dict_ = {}

        total_pos = temp['target'].sum()
        total_neg = len(temp) - total_pos
        temp['non_target'] = np.where(temp['target'] == 1, 0, 1)

        for var in self.variables:
            pos = temp.groupby([var])['target'].sum() / total_pos
            neg = temp.groupby([var])['non_target'].sum() / total_neg

            t = pd.concat([pos, neg], axis=1)
            t['woe'] = np.log(t['target'] / t['non_target'])

            if not t.loc[t['target'] == 0, :].empty or not t.loc[t['non_target'] == 0, :].empty:
                raise ValueError(
                    "The proportion of 1 of the classes for a category in variable {} is zero, and log of zero is "
                    "not defined".format(var))

            self.encoder_dict_[var] = t['woe'].to_dict()

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


