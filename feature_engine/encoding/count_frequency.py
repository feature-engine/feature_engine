# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class CountFrequencyEncoder(BaseCategoricalTransformer):
    """
    The CountFrequencyEncoder() replaces categories by either the count or the
    percentage of observations per category.

    For example in the variable colour, if 10 observations are blue, blue will
    be replaced by 10. Alternatively, if 10% of the observations are blue, blue
    will be replaced by 0.1.

    The CountFrequencyEncoder() will encode only categorical variables by default
    (type 'object' or 'categorical'). You can pass a list of variables to encode.
    Alternatively, the encoder will find and encode all categorical variables
    (type 'object' or 'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as well.
    The procedure is identical, you can either enter the list of variables to encode, or
    the transformer will automatically select all variables.

    The encoder first maps the categories to the counts or frequencies for each
    variable (fit). The encoder then replaces the categories with those numbers
    (transform).

    More details in the :ref:`User Guide <count_freq_encoder>`.

    See BaseCategoricalTransformer() docstring for the explanations of the inherited init params,
    i.e., 'ignore_format' and 'variables'.

    Parameters
    ----------
    encoding_method: str, default='count'
        Desired method of encoding.

        **'count'**: number of observations per category

        **'frequency'**: percentage of observations per category

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the count or frequency per category, per variable.

    variables_:
        The group of variables that will be transformed.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Learn the count or frequency per category, per variable.
    transform:
        Encode the categories to numbers.
    fit_transform:
        Fit to the data, then transform it.
    inverse_transform:
        Encode the numbers into the original categories.

    Notes
    -----
    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    There is a similar implementation in the the open-source package
    `Category encoders <https://contrib.scikit-learn.org/category_encoders/>`_

    See Also
    --------
    feature_engine.encoding.RareLabelEncoder
    category_encoders.count.CountEncoder
    """

    def __init__(
        self,
        encoding_method: str = "count",
        rare_labels: str = "ignore"
    ) -> None:

        if encoding_method not in ["count", "frequency"]:
            raise ValueError(
                "encoding_method takes only values 'count' and 'frequency'"
            )

        self.encoding_method = encoding_method
        self.rare_labels = rare_labels

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the counts or frequencies which will be used to replace the categories.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y: pandas Series, default = None
            y is not needed in this encoder. You can pass y or None.
        """

        X = self._check_fit_input_and_variables(X)

        self.encoder_dict_ = {}

        # learn encoding maps
        for var in self.variables_:
            if self.encoding_method == "count":
                self.encoder_dict_[var] = X[var].value_counts().to_dict()

            elif self.encoding_method == "frequency":
                n_obs = float(len(X))
                self.encoder_dict_[var] = (X[var].value_counts() / n_obs).to_dict()

        self._check_encoding_dictionary()

        self.n_features_in_ = X.shape[1]

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise not necessary
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        return X

    transform.__doc__ = BaseCategoricalTransformer.transform.__doc__

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().inverse_transform(X)

        return X

    inverse_transform.__doc__ = BaseCategoricalTransformer.inverse_transform.__doc__
