# Authors: Nicolas Galli <nicolas.galli@yahoo.com>
# License: BSD 3 clause

from typing import List, Union

import numpy as np
import pandas as pd

from feature_engine.encoding.base_encoder import BaseCategorical
from feature_engine.validation import _return_tags


class PRatioEncoder(BaseCategorical):
    """
    The PRatioEncoder() replaces categories by the ratio of the probability of the
    target = 1 and the probability of the target = 0.

    The target probability ratio is given by:

    .. math::

        p(1) / p(0)

    The log of the target probability ratio is:

    .. math::

        log( p(1) / p(0) )

    **Note**

    This categorical encoding is exclusive for binary classification.

    The division by 0 is not defined and the log(0) is not defined.
    Thus, if p(0) = 0 for the ratio encoder, or either p(0) = 0 or p(1) = 0 for
    log_ratio, in any of the variables, the encoder will return an error.

    The encoder will encode only categorical variables by default (type 'object' or
    'categorical'). You can pass a list of variables to encode. Alternatively, the
    encoder will find and encode all categorical variables (type 'object' or
    'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as well.
    The procedure is identical, you can either enter the list of variables to encode, or
    the transformer will automatically select all variables.

    The encoder first maps the categories to the numbers for each variable (fit). The
    encoder then transforms the categories into the mapped numbers (transform).

    More details in the :ref:`User Guide <pratio_encoder>`.

    Parameters
    ----------
    encoding_method: str, default='ratio'
        Desired method of encoding.

        **'ratio'**: probability ratio

        **'log_ratio'**: log probability ratio

    variables: list, default=None
        The list of categorical variables that will be encoded. If None, the
        encoder will find and transform all variables of type object or categorical by
        default. You can also make the transformer accept numerical variables, see the
        next parameter.

    ignore_format: bool, default=False
        Whether the format in which the categorical variables are cast should be
        ignored. If False, the encoder will automatically select variables of type
        object or categorical, or check that the variables entered by the user are of
        type object or categorical. If True, the encoder will select all variables or
        accept all variables entered by the user, including those cast as numeric.

    errors: string, default='ignore'
        Indicates what to do when categories not present in the train set are
        encountered during transform. If 'raise', then rare categories will raise an
        error. If 'ignore', then rare categories will be set as NaN and a warning will
        be raised instead.

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the probability ratio per category per variable.

    variables_:
        The group of variables that will be transformed.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Learn probability ratio per category, per variable.
    transform:
        Encode categories into numbers.
    fit_transform:
        Fit to the data, then transform it.
    inverse_transform:
        Encode the numbers into the original categories.

    Notes
    -----
    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    See Also
    --------
    feature_engine.encoding.RareLabelEncoder
    """

    def __init__(
        self,
        encoding_method: str = "ratio",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
        errors: str = "ignore"
    ) -> None:

        if encoding_method not in ["ratio", "log_ratio"]:
            raise ValueError(
                "encoding_method takes only values 'ratio' and 'log_ratio'"
            )

        super().__init__(variables, ignore_format, errors)

        self.encoding_method = encoding_method

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Learn the numbers that should be used to replace the categories in each
        variable. That is the ratio of probability.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            categorical variables.

        y: pandas series.
            Target, must be binary.
        """

        X = self._check_fit_input_and_variables(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # check that y is binary
        if y.nunique() != 2:
            raise ValueError(
                "This encoder is designed for binary classification. The target "
                "used has more than 2 unique values."
            )

        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        # if target does not have values 0 and 1, we need to remap, to be able to
        # compute the averages.
        if any(x for x in y.unique() if x not in [0, 1]):
            temp["target"] = np.where(temp["target"] == y.unique()[0], 0, 1)

        self.encoder_dict_ = {}

        for var in self.variables_:

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

        self.n_features_in_ = X.shape[1]

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseCategorical.transform.__doc__

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().inverse_transform(X)

        return X

    inverse_transform.__doc__ = BaseCategorical.inverse_transform.__doc__

    def _more_tags(self):
        tags_dict = _return_tags()
        # in the current format, the tests are performed using continuous np.arrays
        # this means that when we encode some of the values, the denominator is 0
        # and this the transformer raises an error, and the test fails.
        # For this reason, most sklearn transformers will fail. And it has nothing to
        # do with the class not being compatible, it is just that the inputs passed
        # are not suitable
        tags_dict["_skip_test"] = True
        return tags_dict
