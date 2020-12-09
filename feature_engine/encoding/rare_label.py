# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Optional, List, Union
import warnings

import numpy as np
import pandas as pd

from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class RareLabelEncoder(BaseCategoricalTransformer):
    """
    The RareLabelCategoricalEncoder() groups rare / infrequent categories in
    a new category called "Rare", or any other name entered by the user.

    For example in the variable colour, if the percentage of observations
    for the categories magenta, cyan and burgundy are < 5 %, all those
    categories will be replaced by the new label "Rare".

    **Note**

    Infrequent labels can also be grouped under a user defined name, for
    example 'Other'. The name to replace infrequent categories is defined
    with the parameter `replace_with`.

    The encoder will encode only categorical variables (type 'object'). A list
    of variables can be passed as an argument. If no variables are passed as
    argument, the encoder will find and encode all categorical variables
    (object type).

    The encoder first finds the frequent labels for each variable (fit). The encoder
    then groups the infrequent labels under the new label 'Rare' or by another user
    defined string (transform).

    Parameters
    ----------
    tol : float, default=0.05
        The minimum frequency a label should have to be considered frequent.
        Categories with frequencies lower than tol will be grouped.

    n_categories: int, default=10
        The minimum number of categories a variable should have for the encoder
        to find frequent labels. If the variable contains less categories, all
        of them will be considered frequent.

    max_n_categories: int, default=None
        The maximum number of categories that should be considered frequent.
        If None, all categories with frequency above the tolerance (tol) will be
        considered frequent.

    variables : list, default=None
        The list of categorical variables to encode. If None, the encoder will
        find and select all object type variables.

    replace_with : string, default='Rare'
        The category name that will be used to replace infrequent categories.

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the frequent categories, i.e.., those that will be
        kept, per variable.

    Methods
    -------
    fit:
        Find frequent categories.
    transform:
        Group rare categories
    fit_transform:
        Fit to data, then transform it.
    """

    def __init__(
        self,
        tol: float = 0.05,
        n_categories: int = 10,
        max_n_categories: Optional[int] = None,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        replace_with: str = "Rare",
    ) -> None:

        if tol < 0 or tol > 1:
            raise ValueError("tol takes values between 0 and 1")

        if n_categories < 0 or not isinstance(n_categories, int):
            raise ValueError("n_categories takes only positive integer numbers")

        if max_n_categories is not None:
            if max_n_categories < 0 or not isinstance(max_n_categories, int):
                raise ValueError("max_n_categories takes only positive integer numbers")

        if not isinstance(replace_with, str):
            raise ValueError("replace_with takes only strings as values.")

        self.tol = tol
        self.n_categories = n_categories
        self.max_n_categories = max_n_categories
        self.variables = _check_input_parameter_variables(variables)
        self.replace_with = replace_with

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learn the frequent categories for each variable.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just selected
            variables

        y : None
            y is not required. You can pass y or None.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame.
            - If any user provided variable is not categorical
        ValueError
            - If there are no categorical variables in the df or the df is empty
            - If the variable(s) contain null values
        Warning
            If the number of categories in any one variable is less than the indicated
            in `n_categories`.

        Returns
        -------
        self
        """

        X = self._check_fit_input_and_variables(X)

        self.encoder_dict_ = {}

        for var in self.variables:
            if len(X[var].unique()) > self.n_categories:

                # if the variable has more than the indicated number of categories
                # the encoder will learn the most frequent categories
                t = pd.Series(X[var].value_counts() / np.float(len(X)))

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

        self._check_encoding_dictionary()

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Group infrequent categories. Replace infrequent categories by the string 'Rare'
        or any other name provided by the user.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values
            - If dataframe is not of same size as that used in fit()

        Returns
        -------
        X : pandas dataframe of shape = [n_samples, n_features]
            The dataframe where rare categories have been grouped.
        """

        X = self._check_transform_input_and_state(X)

        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]),
                X[feature],
                self.replace_with,
            )

        return X

    def inverse_transform(self, X: pd.DataFrame):
        """inverse_transform is not implemented for this transformer yet."""
        return self
