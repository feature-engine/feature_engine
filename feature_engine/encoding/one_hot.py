# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.fit_attributes import (
    _variables_attribute_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.encoding._docstrings import (
    _ignore_format_docstring,
    _variables_docstring,
)
from feature_engine.encoding.base_encoder import BaseCategoricalTransformer


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class OneHotEncoder(BaseCategoricalTransformer):
    """
    The OneHotEncoder() replaces categorical variables by a set of binary variables
    representing each one of the unique categories in the variable.

    The encoder has the option to create k or k-1 binary variables, where k is the
    number of unique categories.

    The encoder has the additional option to generate binary variables only for the
    most popular categories, that is, the categories that are shared by the
    majority of the observations in the dataset. This behaviour can be specified with
    the parameter `top_categories`.

    The encoder will encode only categorical variables by default (type 'object' or
    'categorical'). You can pass a list of variables to encode. Alternatively, the
    encoder will find and encode all categorical variables (type 'object' or
    'categorical').

    With `ignore_format=True` you have the option to encode numerical variables as well.
    The procedure is identical, you can either enter the list of variables to encode, or
    the transformer will automatically select all variables.

    The encoder first finds the categories to be encoded for each variable (fit). The
    encoder then creates one dummy variable per category for each variable
    (transform).

    **Note**

    New categories in the data to transform, that is, those that did not appear
    in the training set, will be ignored (no binary variable will be created for them).
    This means that observations with categories not present in the train set, will be
    encoded as 0 in all the binary variables.

    **Also Note**

    The original categorical variables are removed from the returned dataset when we
    apply the transform() method. In their place, the binary variables are returned.

    More details in the :ref:`User Guide <onehot_encoder>`.

    Parameters
    ----------
    top_categories: int, default=None
        If None, dummy variables will be created for each unique category of the
        variable. Alternatively, we can indicate in the number of most frequent
        categories to encode. In this case, dummy variables will be created only for
        those popular categories and the rest will be ignored, i.e., they will show the
        value 0 in all the binary variables. Note that if `top_categories` is not None,
        the parameter `drop_last` is ignored.

    drop_last: boolean, default=False
        Only used if `top_categories = None`. It indicates whether to create dummy
        variables for all the categories (k dummies), or if set to `True`, it will
        ignore the last binary variable and return k-1 dummies.

    drop_last_binary: boolean, default=False
        Whether to return 1 or 2 dummy variables for binary categorical variables. When
        a categorical variable has only 2 categories, then the second dummy variable
        created by one hot encoding can be completely redundant. Setting this parameter
        to `True`, will ensure that for every binary variable in the dataset, only 1
        dummy is created.

    {variables}

    {ignore_format}

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the categories for which dummy variables will be created.

    {variables_}

    variables_binary_:
        List with binary variables identified in the data. That is, variables with
        only 2 categories.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Learn the unique categories per variable

    {fit_transform}

    transform:
        Replace the categorical variables by the binary variables.

    Notes
    -----
    If the variables are intended for linear models, it is recommended to encode into
    k-1 or top categories. If the variables are intended for tree based algorithms,
    it is recommended to encode into k or top n categories. If feature selection
    will be performed, then also encode into k or top n categories. Linear models
    evaluate all features during fit, while tree based models and many feature
    selection algorithms evaluate variables or groups of variables separately. Thus, if
    encoding into k-1, the last variable / category will not be examined.

    References
    ----------
    One hot encoding of top categories was described in the following article:

    .. [1] Niculescu-Mizil, et al. "Winning the KDD Cup Orange Challenge with Ensemble
        Selection". JMLR: Workshop and Conference Proceedings 7: 23-34. KDD 2009
        http://proceedings.mlr.press/v7/niculescu09/niculescu09.pdf
    """

    def __init__(
        self,
        top_categories: Optional[int] = None,
        drop_last: bool = False,
        drop_last_binary: bool = False,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        ignore_format: bool = False,
    ) -> None:

        if top_categories and not isinstance(top_categories, int):
            raise ValueError("top_categories takes only integer numbers, 1, 2, 3, etc.")

        if not isinstance(drop_last, bool):
            raise ValueError("drop_last takes only True or False")

        if not isinstance(drop_last_binary, bool):
            raise ValueError("drop_last_binary takes only True or False")

        super().__init__(variables, ignore_format)
        self.top_categories = top_categories
        self.drop_last = drop_last
        self.drop_last_binary = drop_last_binary

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Learns the unique categories per variable. If top_categories is indicated,
        it will learn the most popular categories. Alternatively, it learns all
        unique categories per variable.

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.

        y: pandas series, default=None
            Target. It is not needed in this encoded. You can pass y or
            None.
        """

        X = self._check_X(X)
        self._check_or_select_variables(X)
        self._get_feature_names_in(X)

        self.encoder_dict_ = {}

        # make dummies only for the most popular categories
        if self.top_categories:
            for var in self.variables_:
                self.encoder_dict_[var] = [
                    x
                    for x in X[var]
                    .value_counts()
                    .sort_values(ascending=False)
                    .head(self.top_categories)
                    .index
                ]

        else:
            # return k-1 dummies
            if self.drop_last:
                for var in self.variables_:
                    category_ls = [x for x in X[var].unique()]
                    self.encoder_dict_[var] = category_ls[:-1]

            # return k dummies
            else:
                for var in self.variables_:
                    self.encoder_dict_[var] = [x for x in X[var].unique()]

        self.variables_binary_ = [
            var for var in self.variables_ if X[var].nunique() == 2
        ]

        # automatically encode binary variables as 1 dummy
        if self.drop_last_binary:
            for var in self.variables_binary_:
                category = X[var].unique()[0]
                self.encoder_dict_[var] = [category]

        self._check_encoding_dictionary()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces the categorical variables by the binary variables.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------
        X_new: pandas dataframe.
            The transformed dataframe. The shape of the dataframe will be different from
            the original as it includes the dummy variables in place of the of the
            original categorical ones.
        """

        X = self._check_transform_input_and_state(X)

        for feature in self.variables_:
            for category in self.encoder_dict_[feature]:
                X[str(feature) + "_" + str(category)] = np.where(
                    X[feature] == category, 1, 0
                )

        # drop the original non-encoded variables.
        X.drop(labels=self.variables_, axis=1, inplace=True)

        return X

    def inverse_transform(self, X: pd.DataFrame):
        """inverse_transform is not implemented for this transformer."""
        return self

    def get_feature_names_out(self, input_features: Optional[List] = None) -> List:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features: list, default=None
            Input features. If `input_features` is `None`, then the names of all the
            variables in the transformed dataset (original + new variables) is returned.
            Alternatively, only the names for the binary variables derived from
            input_features will be returned.

        Returns
        -------
        feature_names_out: list
            The feature names.
        """
        check_is_fitted(self)

        if input_features is None:
            input_features_ = self.feature_names_in_
        else:
            if not isinstance(input_features, list):
                raise ValueError(
                    f"input_features must be a list. Got {input_features} instead."
                )
            if any(f for f in input_features if f not in self.feature_names_in_):
                raise ValueError(
                    "Some of the features requested were not seen during training."
                )
            input_features_ = input_features

        # the features not encoded
        feature_names = [f for f in input_features_ if f not in self.variables_]

        # the encoded features
        encoded = [f for f in input_features_ if f in self.variables_]

        for feature in encoded:
            for category in self.encoder_dict_[feature]:
                feature_names.append(str(feature) + "_" + str(category))

        return feature_names
