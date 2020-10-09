# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
from feature_engine.variable_manipulation import _define_variables
from feature_engine.encoding.base_encoder import BaseCategoricalTransformer


class CountFrequencyEncoder(BaseCategoricalTransformer):
    """
    The CountFrequencyCategoricalEncoder() replaces categories by the count of
    observations per category or by the percentage of observations per category.

    For example in the variable colour, if 10 observations are blue, blue will
    be replaced by 10. Alternatively, if 10% of the observations are blue, blue
    will be replaced by 0.1.

    The CountFrequencyCategoricalEncoder() will encode only categorical variables
    (type 'object'). A list of variables to be encoded can be passed as argument.
    Alternatively, the encoder will find and encode all categorical variables
    (object type).

    The encoder first maps the categories to the numbers (counts or frequencies)
    for each variable (fit).

    The encoder then transforms the categories to those mapped numbers (transform).

    Parameters
    ----------

    encoding_method : str, default='count'
        Desired method of encoding.

        'count': number of observations per category

        'frequency': percentage of observations per category

    variables : list
        The list of categorical variables that will be encoded. If None, the
        encoder will find and transform all object type variables.
    """

    def __init__(self, encoding_method="count", variables=None):

        if encoding_method not in ["count", "frequency"]:
            raise ValueError(
                "encoding_method takes only values 'count' and 'frequency'"
            )

        self.encoding_method = encoding_method
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Learns the counts or frequencies which will be used to replace the categories.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            The user can pass the entire dataframe.

        y : None
            y is not needed in this encoder. You can pass y or None.

        Attributes
        ----------

        encoder_dict_: dictionary
            Dictionary containing the {category: count / frequency} pairs for
            each variable.
        """

        X = self._check_fit_input_and_variables(X)

        self.encoder_dict_ = {}

        # learn encoding maps
        for var in self.variables:
            if self.encoding_method == "count":
                self.encoder_dict_[var] = X[var].value_counts().to_dict()

            elif self.encoding_method == "frequency":
                n_obs = np.float(len(X))
                self.encoder_dict_[var] = (X[var].value_counts() / n_obs).to_dict()

        self._check_encoding_dictionary()

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseCategoricalTransformer.transform.__doc__

    def inverse_transform(self, X):
        X = super().inverse_transform(X)
        return X

    inverse_transform.__doc__ = BaseCategoricalTransformer.inverse_transform.__doc__
