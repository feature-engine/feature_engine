# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _inverse_transform_docstring,
)
from feature_engine._docstrings.fit_attributes import (
    _variables_attribute_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.encoding._docstrings import (
    _errors_docstring,
    _ignore_format_docstring,
    _transform_docstring,
    _variables_docstring,
)
from feature_engine.encoding.base_encoder import BaseCategorical


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_docstring,
    errors=_errors_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    transform=_transform_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class OrdinalEncoder(BaseCategorical):
    """
    The OrdinalEncoder() replaces categories by ordinal numbers
    (0, 1, 2, 3, etc). The numbers can be ordered based on the mean of the target
    per category, or assigned arbitrarily.

    The encoder will encode only categorical variables by default (type 'object' or
    'categorical'). You can pass a list of variables to encode. Alternatively, the
    encoder will find and encode all categorical variables (type 'object' or
    'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as well.
    The procedure is identical, you can either enter the list of variables to encode, or
    the transformer will automatically select all variables.

    The encoder first maps the categories to the numbers for each variable (fit). The
    encoder then transforms the categories to the mapped numbers (transform).

    More details in the :ref:`User Guide <ordinal_encoder>`.

    Parameters
    ----------
    encoding_method: str, default='ordered'
        Desired method of encoding.

        **'ordered'**: the categories are numbered in ascending order according to
        the target mean value per category.

        **'arbitrary'**: categories are numbered arbitrarily.

    {variables}

    {ignore_format}

    {errors}

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the ordinal number per category, per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find the integer to replace each category in each variable.

    {fit_transform}

    {inverse_transform}

    {transform}

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
    category_encoders.ordinal.OrdinalEncoder

    References
    ----------
    Encoding into integers ordered following target mean was discussed in the following
    talk at PyData London 2017:

    .. [1] Galli S. "Machine Learning in Financial Risk Assessment".
        https://www.youtube.com/watch?v=KHGGlozsRtA
    """

    def __init__(
        self,
        encoding_method: str = "ordered",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
        errors: str = "ignore",
    ) -> None:

        if encoding_method not in ["ordered", "arbitrary"]:
            raise ValueError(
                "encoding_method takes only values 'ordered' and 'arbitrary'"
            )

        super().__init__(variables, ignore_format, errors)

        self.encoding_method = encoding_method

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learn the numbers to be used to replace the categories in each
        variable.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            variables to be encoded.

        y: pandas series, default=None
            The Target. Can be None if `encoding_method='arbitrary'`.
            Otherwise, y needs to be passed when fitting the transformer.
        """

        if self.encoding_method == "ordered":
            X, y = self._check_X_y(X, y)
        else:
            X = self._check_X(X)

        self._check_or_select_variables(X)
        self._get_feature_names_in(X)

        if self.encoding_method == "ordered":
            temp = pd.concat([X, y], axis=1)
            temp.columns = list(X.columns) + ["target"]

        # find mappings
        self.encoder_dict_ = {}

        for var in self.variables_:

            if self.encoding_method == "ordered":
                t = (
                    temp.groupby([var])["target"]
                    .mean()
                    .sort_values(ascending=True)
                    .index
                )

            elif self.encoding_method == "arbitrary":
                t = X[var].unique()

            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        self._check_encoding_dictionary()

        return self
