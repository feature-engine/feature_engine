# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _missing_values_docstring,
    _variables_categorical_docstring,
)
from feature_engine._docstrings.init_parameters.encoders import (
    _ignore_format_docstring,
    _unseen_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _inverse_transform_docstring,
    _transform_encoders_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import check_X
from feature_engine.encoding._helper_functions import check_parameter_unseen
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixinNA,
    CategoricalMethodsMixin,
)

_unseen_docstring = (
    _unseen_docstring
    + """ If `'encode'`, unseen categories will be encoded as 0 (zero)."""
)


@Substitution(
    ignore_format=_ignore_format_docstring,
    missing_values=_missing_values_docstring,
    variables=_variables_categorical_docstring,
    unseen=_unseen_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    transform=_transform_encoders_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class CountFrequencyEncoder(CategoricalMethodsMixin, CategoricalInitMixinNA):
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

    Parameters
    ----------
    encoding_method: str, default='count'
        Desired method of encoding.

        **'count'**: number of observations per category

        **'frequency'**: percentage of observations per category

    {variables}

    {missing_values}

    {ignore_format}

    {unseen}

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the count or frequency per category, per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the count or frequency per category, per variable.

    {fit_transform}

    {inverse_transform}

    {transform}

    Notes
    -----
    NAN will be introduced when encoding categories that were not present in the
    training set. If this happens, try grouping infrequent categories using the
    RareLabelEncoder(), or set `unseen='encode'`.

    There is a similar implementation in the open-source package
    `Category encoders <https://contrib.scikit-learn.org/category_encoders/>`_

    See Also
    --------
    feature_engine.encoding.RareLabelEncoder
    category_encoders.count.CountEncoder

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.encoding import CountFrequencyEncoder
    >>> X = pd.DataFrame(dict(x1 = [1,2,3,4], x2 = ["c", "a", "b", "c"]))
    >>> cf = CountFrequencyEncoder(encoding_method='count')
    >>> cf.fit(X)
    >>> cf.transform(X)
       x1  x2
    0   1   2
    1   2   1
    2   3   1
    3   4   2

    >>> cf = CountFrequencyEncoder(encoding_method='frequency')
    >>> cf.fit(X)
    >>> cf.transform(X)
       x1    x2
    0   1  0.50
    1   2  0.25
    2   3  0.25
    3   4  0.50
    """

    def __init__(
        self,
        encoding_method: str = "count",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        missing_values: str = "raise",
        ignore_format: bool = False,
        unseen: str = "ignore",
    ) -> None:

        if encoding_method not in ["count", "frequency"]:
            raise ValueError(
                "encoding_method takes only values 'count' and 'frequency'. "
                f"Got {encoding_method} instead."
            )

        check_parameter_unseen(unseen, ["ignore", "raise", "encode"])
        super().__init__(variables, missing_values, ignore_format)
        self.encoding_method = encoding_method
        self.unseen = unseen

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
        X = check_X(X)
        variables_ = self._check_or_select_variables(X)
        self._check_na(X, variables_)

        self.encoder_dict_ = {}

        # learn encoding maps
        for var in variables_:
            if self.encoding_method == "count":
                self.encoder_dict_[var] = X[var].value_counts().to_dict()

            elif self.encoding_method == "frequency":
                self.encoder_dict_[var] = X[var].value_counts(normalize=True).to_dict()
            else:
                raise ValueError(
                    "Unrecognized value for encoding_method. It should be 'count' or "
                    f"'frequency'. Got {self.encoding_method} instead."
                )

        # unseen categories are replaced by 0
        if self.unseen == "encode":
            self._unseen = 0

        # assign underscore parameters at the end in case code above fails
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self
