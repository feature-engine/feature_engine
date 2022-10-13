# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters import (
    _ignore_format_docstring,
    _variables_categorical_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import _check_contains_na, check_X
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixin,
    CategoricalMethodsMixin,
)


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_categorical_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class RareLabelEncoder(CategoricalInitMixin, CategoricalMethodsMixin):
    """
    The RareLabelEncoder() groups rare or infrequent categories in
    a new category called "Rare", or any other name entered by the user.

    For example in the variable colour, if the percentage of observations
    for the categories magenta, cyan and burgundy are < 5 %, all those
    categories will be replaced by the new label "Rare".

    **Note**

    Infrequent labels can also be grouped under a user defined name, for
    example 'Other'. The name to replace infrequent categories is defined
    with the parameter `replace_with`.

    The encoder will encode only categorical variables by default (type 'object' or
    'categorical'). You can pass a list of variables to encode.Alternatively, the
    encoder will find and encode all categorical variables (type 'object' or
    'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as well.
    The procedure is identical, you can either enter the list of variables to encode, or
    the transformer will automatically select all variables.

    The encoder first finds the frequent labels for each variable (fit). The encoder
    then groups the infrequent labels under the new label 'Rare' or by another user
    defined string (transform).

    More details in the :ref:`User Guide <rarelabel_encoder>`.


    Parameters
    ----------
    tol: float, default=0.05
        The minimum frequency a label should have to be considered frequent.
        Categories with frequencies lower than tol will be grouped.

    n_categories: int, default=10
        The minimum number of categories a variable should have for the encoder
        to find frequent labels. If the variable contains less categories, all
        of them will be considered frequent.

    max_n_categories: int, default=None
        The maximum number of categories that should be considered frequent.
        If None, all categories with frequency above the tolerance (tol) will be
        considered frequent. If you enter 5, only the 5 most frequent categories will
        be retained and the rest grouped.

    replace_with: string, intege or float, default='Rare'
        The value that will be used to replace infrequent categories.

    {variables}

    {ignore_format}

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the frequent categories, i.e., those that will be kept, per
        variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Find frequent categories.

    {fit_transform}

    transform:
        Group rare categories
    """

    def __init__(
        self,
        tol: float = 0.05,
        n_categories: int = 10,
        max_n_categories: Optional[int] = None,
        replace_with: Union[str, int, float] = "Rare",
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
    ) -> None:

        if tol < 0 or tol > 1:
            raise ValueError("tol takes values between 0 and 1")

        if n_categories < 0 or not isinstance(n_categories, int):
            raise ValueError("n_categories takes only positive integer numbers")

        if max_n_categories is not None:
            if (
                not isinstance(max_n_categories, int)
                or isinstance(max_n_categories, int)
                and max_n_categories < 0
            ):
                raise ValueError("max_n_categories takes only positive integer numbers")

        super().__init__(variables, ignore_format)
        self.tol = tol
        self.n_categories = n_categories
        self.max_n_categories = max_n_categories
        self.replace_with = replace_with

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the frequent categories for each variable.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just selected
            variables

        y: None
            y is not required. You can pass y or None.
        """

        X = check_X(X)
        self._fit(X)
        self._get_feature_names_in(X)

        self.encoder_dict_ = {}

        for var in self.variables_:
            if len(X[var].unique()) > self.n_categories:

                # if the variable has more than the indicated number of categories
                # the encoder will learn the most frequent categories
                t = pd.Series(X[var].value_counts() / float(len(X)))

                # non-rare labels:
                freq_idx = t[t >= self.tol].index

                if self.max_n_categories:
                    self.encoder_dict_[var] = freq_idx[: self.max_n_categories]
                else:
                    self.encoder_dict_[var] = freq_idx

            else:
                # if the total number of categories is smaller than the indicated
                # the encoder will consider all categories as frequent.
                warnings.warn(
                    "The number of unique categories for variable {} is less than that "
                    "indicated in n_categories. Thus, all categories will be "
                    "considered frequent".format(var)
                )
                self.encoder_dict_[var] = X[var].unique()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Group infrequent categories. Replace infrequent categories by the string 'Rare'
        or any other name provided by the user.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataframe where rare categories have been grouped.
        """

        X = self._check_transform_input_and_state(X)

        # check if dataset contains na
        _check_contains_na(X, self.variables_)

        for feature in self.variables_:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]),
                X[feature],
                self.replace_with,
            )

        return X

    def inverse_transform(self, X: pd.DataFrame):
        """inverse_transform is not implemented for this transformer."""
        raise NotImplementedError(
            "inverse_transform is not implemented for this transformer."
        )
