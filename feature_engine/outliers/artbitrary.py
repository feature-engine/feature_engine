# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from feature_engine.dataframe_checks import _is_dataframe, _check_contains_na
from feature_engine.outliers.base_outlier import BaseOutlier
from feature_engine.variable_manipulation import _find_numerical_variables


class ArbitraryOutlierCapper(BaseOutlier):
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

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseOutlier.transform.__doc__


