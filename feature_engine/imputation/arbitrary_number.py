# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from feature_engine.dataframe_checks import _is_dataframe
from feature_engine.variable_manipulation import (
    _define_variables,
    _find_numerical_variables
)
from feature_engine.parameter_checks import _define_numerical_dict
from feature_engine.imputation.base_imputer import BaseImputer


class ArbitraryNumberImputer(BaseImputer):
    """
    The ArbitraryNumberImputer() replaces missing data in each variable
    by an arbitrary value determined by the user.

    Parameters
    ----------

    arbitrary_number : int or float, default=999
        the number to be used to replace missing data.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all numerical type variables. Attribute is used only if `imputer_dict`
        attribute is None.

    imputer_dict: dict, default=None
        The dictionary of variables and their arbitrary numbers. If imputer_dict is not None,
        it has to be dictionary with all values of integer or float type.
        If None, `variables` attribute is used for imputation.
    """

    def __init__(self, arbitrary_number=999, variables=None, imputer_dict=None):

        if isinstance(arbitrary_number, int) or isinstance(arbitrary_number, float):
            self.arbitrary_number = arbitrary_number
        else:
            raise ValueError('arbitrary_number must be numeric of type int or float')

        self.variables = _define_variables(variables)

        self.imputer_dict = _define_numerical_dict(imputer_dict)

    def fit(self, X, y=None):
        """
        Checks that the variables are numerical.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            User can pass the entire dataframe, not just the variables to impute.

        y : None
            y is not needed in this imputation. You can pass None or y.


        Attributes
        ----------

        imputer_dict_: dictionary
            The dictionary containing the values that will be used to replace each variable.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        if self.imputer_dict:
            self.variables = _find_numerical_variables(X, self.imputer_dict.keys())
        else:
            self.variables = _find_numerical_variables(X, self.variables)

        # create the imputer dictionary
        if self.imputer_dict:
            self.imputer_dict_ = self.imputer_dict
        else:
            self.imputer_dict_ = {var: self.arbitrary_number for var in self.variables}

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseImputer.transform.__doc__
