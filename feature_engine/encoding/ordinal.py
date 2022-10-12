# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters import (
    _ignore_format_docstring,
    _unseen_docstring,
    _variables_categorical_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _inverse_transform_docstring,
    _transform_encoders_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.encoding._helper_functions import check_parameter_unseen
from feature_engine.dataframe_checks import check_X, check_X_y
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixin,
    CategoricalMethodsMixin,
)

_unseen_docstring = (
    _unseen_docstring + """ If `'encode'`, unseen categories will be encoded as -1."""
)


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_categorical_docstring,
    unseen=_unseen_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    transform=_transform_encoders_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class OrdinalEncoder(CategoricalInitMixin, CategoricalMethodsMixin):
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

    {unseen}

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
        unseen: str = "ignore",
    ) -> None:

        if encoding_method not in ["ordered", "arbitrary"]:
            raise ValueError(
                "encoding_method takes only values 'ordered' and 'arbitrary'"
            )

        check_parameter_unseen(unseen, ["ignore", "raise", "encode"])
        super().__init__(variables, ignore_format)
        self.encoding_method = encoding_method
        self.unseen = unseen

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
            X, y = check_X_y(X, y)
        else:
            X = check_X(X)

        self._fit(X)
        self._get_feature_names_in(X)

        # find mappings
        self.encoder_dict_ = {}

        for var in self.variables_:

            if self.encoding_method == "ordered":
                t = y.groupby(X[var]).mean()  # type: ignore
                t = t.sort_values(ascending=True).index

            elif self.encoding_method == "arbitrary":
                t = X[var].unique()

            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        if self.unseen == "encode":
            self._unseen = -1

        return self
