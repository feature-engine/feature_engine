# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import scipy.stats as stats

from feature_engine.variable_manipulation import _define_variables
from feature_engine.base_transformers import BaseNumericalTransformer


class BoxCoxTransformer(BaseNumericalTransformer):
    """
    The BoxCoxTransformer() applies the BoxCox transformation to numerical
    variables.

    The BoxCox transformation implemented by this transformer is that of
    SciPy.stats:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html

    The BoxCoxTransformer() works only with numerical positive variables (>=0,
    the transformer also works for zero values).

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
        The dictionary containing the {variable: best exponent for the BoxCox
        transfomration} pairs. These are determined automatically.
    """

    def __init__(self, variables=None):

        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Learns the optimal lambda for the BoxCox transformation.

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

        if (X[self.variables] < 0).any().any():
            raise ValueError(
                "Some variables contain negative values, try Yeo-Johnson transformation instead"
            )

        self.lambda_dict_ = {}

        for var in self.variables:
            _, self.lambda_dict_[var] = stats.boxcox(X[var])

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Applies the BoxCox transformation.

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

        # check if variable contains negative numbers
        if (X[self.variables] < 0).any().any():
            raise ValueError(
                "Some variables contain negative values, try Yeo-Johnson transformation instead"
            )

        # transform
        for feature in self.variables:
            X[feature] = stats.boxcox(X[feature], lmbda=self.lambda_dict_[feature])

        return X
