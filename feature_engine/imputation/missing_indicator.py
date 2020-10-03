# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.imputation.base_imputer import BaseImputer
from feature_engine.variable_manipulation import _define_variables


class AddMissingIndicator(BaseImputer):
    """
    The AddMissingIndicator() adds an additional column or binary variable that
    indicates if data is missing.

    AddMissingIndicator() will add as many missing indicators as variables
    indicated by the user, or variables with missing data in the train set.

    The AddMissingIndicator() works for both numerical and categorical variables.
    The user can pass a list with the variables for which the missing indicators
    should be added as a list. Alternatively, the imputer will select and add missing
    indicators to all variables in the training set that show missing data.

    Parameters
    ----------

    missing_only : bool, defatult=True
        Indicates if missing indicators should be added to variables with missing
        data or to all variables.

        True: indicators will be created only for those variables that showed
        missing data during fit.

        False: indicators will be created for all variables

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all variables with missing data.
        Note: the transformer will first select all variables or all user entered variables
        and if how=missing_only, it will re-select from the original group only those
        that show missing data in during fit.
    """

    def __init__(self, missing_only=True, variables=None):

        if not isinstance(missing_only, bool):
            raise ValueError("missing_only takes values True or False")

        self.variables = _define_variables(variables)
        self.missing_only = missing_only

    def fit(self, X, y=None):
        """
        Learns the variables for which the missing indicators will be created.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : None
            y is not needed in this imputation. You can pass None or y.

        Attributes
        ----------

        variables_: list
            the lit of variables for which the missing indicator will be created.
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find variables for which indicator should be added
        if self.missing_only:
            if not self.variables:
                self.variables_ = [var for var in X.columns if X[var].isnull().sum() > 0]
            else:
                self.variables_ = [var for var in self.variables if X[var].isnull().sum() > 0]

        else:
            if not self.variables:
                self.variables_ = [var for var in X.columns]
            else:
                self.variables_ = self.variables

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Adds the binary missing indicators.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe containing the additional binary variables.
            Binary variables are named with the original variable name plus
            '_na'.
        """

        X = self._check_transform_input_and_state(X)

        X = X.copy()
        for feature in self.variables_:
            X[feature + '_na'] = np.where(X[feature].isnull(), 1, 0)

        return X

