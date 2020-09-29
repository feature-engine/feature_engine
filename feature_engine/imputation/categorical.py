# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df
from feature_engine.variable_manipulation import (
    _find_categorical_variables,
    _define_variables,
    _find_numerical_variables,
    _define_numerical_dict
)
from feature_engine.base_transformers import BaseImputer


class CategoricalImputer(BaseImputer):
    """
    The CategoricalVariableImputer() replaces missing data in categorical variables
    by the string 'Missing' or by the most frequent category.

    The CategoricalVariableImputer() works only with categorical variables.

    The user can pass a list with the variables to be imputed. Alternatively,
    the CategoricalVariableImputer() will automatically find and select all
    variables of type object.

    Parameters
    ----------

    imputation_method : str, default=missing
        Desired method of imputation. Can be 'frequent' or 'missing'.
        
    fill_value : str, default='Missing'
        Only used when imputation_method='missing'. Can be used to set a 
        user-defined value to replace the missing data.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all object type variables.

    return_object: bool, default=False
        If working with numerical variables cast as object, decide
        whether to return the variables as numeric or re-cast them as object.
        Note that pandas will re-cast them automatically as numeric after the
        transformation with the mode.

        Tip: return the variables as object if planning to do categorical encoding
        with feature-engine.
    """

    def __init__(self, imputation_method='missing', fill_value='Missing', variables=None, return_object=False):
        
        if imputation_method not in ['missing', 'frequent']:
            raise ValueError("imputation_method takes only values 'missing' or 'frequent'")
        
        if not isinstance(fill_value, str):
            raise ValueError("parameter 'fill_value' should be string")
        
        self.imputation_method = imputation_method
        self.fill_value = fill_value
        self.variables = _define_variables(variables)
        self.return_object = return_object

    def fit(self, X, y=None):
        """
        Learns the most frequent category if the imputation method is set to frequent.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the selected variables.

        y : None
            y is not needed in this imputation. You can pass None or y.

        Attributes
        ----------

        imputer_dict_: dictionary
            The dictionary mapping each variable to the most frequent category, or to
            the value 'Missing' depending on the imputation_method. The most frequent
            category is calculated when fitting the transformer.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for categorical variables
        self.variables = _find_categorical_variables(X, self.variables)
        
        if self.imputation_method == 'missing':
            self.imputer_dict_ = {var: self.fill_value for var in self.variables}

        elif self.imputation_method == 'frequent':
            self.imputer_dict_ = {}

            for var in self.variables:
                mode_vals = X[var].mode()

                # careful: some variables contain multiple modes
                if len(mode_vals) == 1:
                    self.imputer_dict_[var] = mode_vals[0]
                else:
                    raise ValueError('The variable {} contains multiple frequent categories.'.format(var))

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        # bring functionality from the BaseImputer
        X = super().transform(X)

        # add additional step to return variables cast as object
        if self.return_object:
            X[self.variables] = X[self.variables].astype('O')
        return X

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    transform.__doc__ = BaseImputer.transform.__doc__


