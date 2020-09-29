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


class EndTailImputer(BaseImputer):
    """
    The EndTailImputer() transforms features by replacing missing data by a
    value at either tail of the distribution.

    The EndTailImputer() works only with numerical variables.

    The user can indicate the variables to be imputed in a list. Alternatively, the
    EndTailImputer() will automatically find and select all variables of type numeric.

    The imputer first calculates the values at the end of the distribution for each variable
    (fit). The values at the end of the distribution are determined using the Gaussian limits,
    the the IQR proximity rule limits, or a factor of the maximum value:

    Gaussian limits:
        right tail: mean + 3*std

        left tail: mean - 3*std

    IQR limits:
        right tail: 75th quantile + 3*IQR

        left tail:  25th quantile - 3*IQR

    where IQR is the inter-quartile range = 75th quantile - 25th quantile

    Maximum value:
        right tail: max * 3

        left tail: not applicable

    You can change the factor that multiplies the std, IQR or the maximum value
    using the parameter 'fold'.

    The imputer then replaces the missing data with the estimated values (transform).

    Parameters
    ----------

    distribution : str, default=gaussian
        Method to be used to find the replacement values. Can take 'gaussian',
        'skewed' or 'max'.

        gaussian: the imputer will use the Gaussian limits to find the values
        to replace missing data.

        skewed: the imputer will use the IQR limits to find the values to replace
        missing data.

        max: the imputer will use the maximum values to replace missing data. Note
        that if 'max' is passed, the parameter 'tail' is ignored.

    tail : str, default=right
        Indicates if the values to replace missing data should be selected from the right
        or left tail of the variable distribution. Can take values 'left' or 'right'.

    fold: int, default=3
        Factor to multiply the std, the IQR or the Max values. Recommended values
        are 2 or 3 for Gaussian, or 1.5 or 3 for skewed.

    variables : list, default=None
        The list of variables to be imputed. If None, the imputer will find and
        select all variables of type numeric.
    """

    def __init__(self, distribution='gaussian', tail='right', fold=3, variables=None):

        if distribution not in ['gaussian', 'skewed', 'max']:
            raise ValueError("distribution takes only values 'gaussian', 'skewed' or 'max'")

        if tail not in ['right', 'left']:
            raise ValueError("tail takes only values 'right' or 'left'")

        if fold <= 0:
            raise ValueError("fold takes only positive numbers")

        self.distribution = distribution
        self.tail = tail
        self.fold = fold
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Learns the values at the end of the variable distribution.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            The user can pass the entire dataframe, not just the variables that need imputation.

        y : None
            y is not needed in this imputation. You can pass None or y.

        Attributes
        ----------

        imputer_dict_: dictionary
            The dictionary containing the values at the end of the distribution
            per variable. These values will be used by the imputer to replace missing
            data.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables = _find_numerical_variables(X, self.variables)

        # estimate imputation values
        if self.distribution == 'max':
            self.imputer_dict_ = (X[self.variables].max() * self.fold).to_dict()

        elif self.distribution == 'gaussian':
            if self.tail == 'right':
                self.imputer_dict_ = (X[self.variables].mean() + self.fold * X[self.variables].std()).to_dict()
            elif self.tail == 'left':
                self.imputer_dict_ = (X[self.variables].mean() - self.fold * X[self.variables].std()).to_dict()

        elif self.distribution == 'skewed':
            IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
            if self.tail == 'right':
                self.imputer_dict_ = (X[self.variables].quantile(0.75) + (IQR * self.fold)).to_dict()
            elif self.tail == 'left':
                self.imputer_dict_ = (X[self.variables].quantile(0.25) - (IQR * self.fold)).to_dict()

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseImputer.transform.__doc__
