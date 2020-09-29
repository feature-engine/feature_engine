# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.variable_manipulation import (
    _define_variables,
    _find_numerical_variables
)

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
    _check_contains_na,
)


class ArbitraryOutlierCapper(BaseEstimator, TransformerMixin):
    """ 
    The ArbitraryOutlierCapper() caps the maximum or minimum values of a variable
    by an arbitrary value indicated by the user.
       
    The user must provide the maximum or minimum values that will be used
    to cap each variable in a dictionary {feature:capping value}

    Parameters
    ----------
    
    capping_max : dictionary, default=None
        user specified capping values on right tail of the distribution (maximum
        values).

    capping_min : dictionary, default=None
        user specified capping values on left tail of the distribution (minimum
        values).

    missing_values : string, default='raise'
    	Indicates if missing values should be ignored or raised. If 
    	missing_values='raise' the transformer will return an error if the
    	training or other datasets contain missing values.        
    """

    def __init__(self, max_capping_dict=None, min_capping_dict=None, missing_values='raise'):

        if not max_capping_dict and not min_capping_dict:
            raise ValueError("Please provide at least 1 dictionary with the capping values per variable")

        if max_capping_dict is None or isinstance(max_capping_dict, dict):
            self.max_capping_dict = max_capping_dict
        else:
            raise ValueError("max_capping_dict should be a dictionary")

        if min_capping_dict is None or isinstance(min_capping_dict, dict):
            self.min_capping_dict = min_capping_dict
        else:
            raise ValueError("min_capping_dict should be a dictionary")

        if min_capping_dict is None:
            self.variables = [x for x in max_capping_dict.keys()]
        elif max_capping_dict is None:
            self.variables = [x for x in min_capping_dict.keys()]
        else:
            tmp = min_capping_dict.copy()
            tmp.update(max_capping_dict)
            self.variables = [x for x in tmp.keys()]

        if missing_values not in ['raise', 'ignore']:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'")

        self.missing_values = missing_values

    def fit(self, X, y=None):
        """
        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : None
            y is not needed in this transformer. You can pass y or None.

        Attributes
        ----------

        right_tail_caps_: dictionary
            The dictionary containing the maximum values at which variables
            will be capped.

        left_tail_caps_ : dictionary
            The dictionary containing the minimum values at which variables
            will be capped.
        """
        X = _is_dataframe(X)

        if self.missing_values == 'raise':
            # check if dataset contains na
            _check_contains_na(X, self.variables)

        # find or check for numerical variables
        self.variables = _find_numerical_variables(X, self.variables)

        if self.max_capping_dict is not None:
            self.right_tail_caps_ = self.max_capping_dict
        else:
            self.right_tail_caps_ = {}

        if self.min_capping_dict is not None:
            self.left_tail_caps_ = self.min_capping_dict
        else:
            self.left_tail_caps_ = {}

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Caps the variable values, that is, censors outliers.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the capped variables.
        """

        # check if class was fitted
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        if self.missing_values == 'raise':
            # check if dataset contains na
            _check_contains_na(X, self.variables)

        # Check that the dataframe contains the same number of columns
        # than the dataframe used to fit the imputer.
        _check_input_matches_training_df(X, self.input_shape_[1])

        # replace outliers
        for feature in self.right_tail_caps_.keys():
            X[feature] = np.where(X[feature] > self.right_tail_caps_[feature], self.right_tail_caps_[feature],
                                  X[feature])

        for feature in self.left_tail_caps_.keys():
            X[feature] = np.where(X[feature] < self.left_tail_caps_[feature], self.left_tail_caps_[feature], X[feature])

        return X


