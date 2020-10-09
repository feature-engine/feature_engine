# Authors: Nicolas Galli <nicolas.galli@yahoo.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd

from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.variable_manipulation import _define_variables


class PRatioEncoder(BaseCategoricalTransformer):
    """
    The PRatioEncoder() replaces categories by the ratio of the probability of the
    target = 1 and the probability of the target = 0.

    The target probability ratio is given by: p(1) / p(0)

    The log of the target probability ratio is: np.log( p(1) / p(0) )

    Note: This categorical encoding is exclusive for binary classification.

    For example in the variable colour, if the mean of the target = 1 for blue
    is 0.8 and the mean of the target = 0  is 0.2, blue will be replaced by:
    0.8 / 0.2 = 4 if ratio is selected, or log(0.8/0.2) = 1.386 if log_ratio
    is selected.

    Note: the division by 0 is not defined and the log(0) is not defined.
    Thus, if p(0) = 0 for the ratio encoder, or either p(0) = 0 or p(1) = 0 for
    log_ratio, in any of the variables, the encoder will return an error.

    The encoder will encode only categorical variables (type 'object'). A list
    of variables can be passed as an argument. If no variables are passed as
    argument, the encoder will find and encode all categorical variables
    (object type).

    The encoder first maps the categories to the numbers for each variable (fit).

    The encoder then transforms the categories into the mapped numbers (transform).

    Parameters
    ----------

    encoding_method : str, default=woe
        Desired method of encoding.

        'ratio' : probability ratio

        'log_ratio' : log probability ratio

    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the
        encoder will find and select all object type variables.
    """

    def __init__(self, encoding_method="ratio", variables=None):

        if encoding_method not in ["ratio", "log_ratio"]:
            raise ValueError(
                "encoding_method takes only values 'ratio' and 'log_ratio'"
            )

        self.encoding_method = encoding_method
        self.variables = _define_variables(variables)

    def fit(self, X, y):
        """
        Learns the numbers that should be used to replace the categories in each
        variable. That is the ratio of probability.

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
            The dictionary containing the {category: ratio} pairs per variable.
        """

        X = self._check_fit_input_and_variables(X)

        # check that y is binary
        if any(x for x in y.unique() if x not in [0, 1]):

            raise ValueError(
                "This encoder is only designed for binary classification, values of y "
                "can be only 0 or 1."
            )

        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        self.encoder_dict_ = {}

        for var in self.variables:
          
            t = temp.groupby(var)["target"].mean()
            t = pd.concat([t, 1 - t], axis=1)
            t.columns = ["p1", "p0"]

            if self.encoding_method == "log_ratio":
                if not t.loc[t["p0"] == 0, :].empty or not t.loc[t["p1"] == 0, :].empty:
                    raise ValueError(
                        "p(0) or p(1) for a category in variable {} is zero, log of "
                        "zero is not defined".format(var)
                    )
                else:
                    self.encoder_dict_[var] = (np.log(t.p1 / t.p0)).to_dict()

            elif self.encoding_method == "ratio":
                if not t.loc[t["p0"] == 0, :].empty:
                    raise ValueError(
                        "p(0) for a category in variable {} is zero, division by 0 is "
                        "not defined".format(var)
                    )

                else:
                    self.encoder_dict_[var] = (t.p1 / t.p0).to_dict()

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

