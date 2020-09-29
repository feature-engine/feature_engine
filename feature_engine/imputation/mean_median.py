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


class MeanMedianImputer(BaseImputer):
    """
    The MeanMedianImputer() transforms features by replacing missing data by the mean
    or median value of the variable.

    The MeanMedianImputer() works only with numerical variables.

    Users can pass a list of variables to be imputed as argument. Alternatively, the
    MeanMedianImputer() will automatically find and select all variables of type numeric.

    The imputer first calculates the mean / median values of the variables (fit).

    The imputer then replaces the missing data with the estimated mean / median (transform).

    Parameters
    ----------

    imputation_method : str, default=median
        Desired method of imputation. Can take 'mean' or 'median'.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will select
        all variables of type numeric.
    """

    def __init__(self, imputation_method='median', variables=None):

        if imputation_method not in ['median', 'mean']:
            raise ValueError("imputation_method takes only values 'median' or 'mean'")

        self.imputation_method = imputation_method
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Learns the mean or median values.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            User can pass the entire dataframe, not just the variables that need imputation.

        y : pandas series or None, default=None
            y is not needed in this imputation. You can pass None or y.

        Attributes
        ----------

        imputer_dict_ : dictionary
            The dictionary containing the mean / median values per variable. These
            values will be used by the imputer to replace missing data.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables = _find_numerical_variables(X, self.variables)

        # find imputation parameters: mean or median
        if self.imputation_method == 'mean':
            self.imputer_dict_ = X[self.variables].mean().to_dict()

        elif self.imputation_method == 'median':
            self.imputer_dict_ = X[self.variables].median().to_dict()

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseImputer.transform.__doc__

