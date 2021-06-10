# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.variable_manipulation import _check_input_parameter_variables


class OneHotEncoder(BaseCategoricalTransformer):
    """
    One hot encoding consists in replacing the categorical variable by a
    combination of binary variables which take value 0 or 1, to indicate if
    a certain category is present in an observation. The binary variables are also
    known as dummy variables.

    For example, from the categorical variable "Gender" with categories "female" and
    "male", we can generate the boolean variable "female", which takes 1 if the
    observation is female or 0 otherwise. We can also generate the variable "male",
    which takes 1 if the observation is "male" and 0 otherwise.

    The encoder can create k binary variables per categorical variable, k being the
    number of unique categories, or alternatively k-1 to avoid redundant information.
    This behaviour can be specified using the parameter `drop_last`.

    The encoder has the additional option to generate binary variables only for the
    top n most popular categories, that is, the categories that are shared by the
    majority of the observations in the dataset. This behaviour can be specified with
    the parameter `top_categories`.

    **Note**

    Only when creating binary variables for all categories of the variable, we
    can specify if we want to encode into k or k-1 binary variables, where k is the
    number if unique categories. If we encode only the top n most popular categories,
    the encoder will create only n binary variables per categorical variable.
    Observations that do not show any of these popular categories, will have 0 in all
    the binary variables.

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

    Parameters
    ----------
    top_categories: int, default=None
        If None, a dummy variable will be created for each category of the variable.
        Alternatively, we can indicate in `top_categories` the number of most frequent
        categories to encode. In this case, dummy variables will be created only for
        those popular categories and the rest will be ignored, i.e., they will show the
        value 0 in all the binary variables.

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

    variables: list, default=None
        The list of categorical variables that will be encoded. If None, the
        encoder will find and transform all variables of type object or categorical by
        default. You can also make the transformer accept numerical variables, see the
        next parameter.

    ignore_format: bool, default=False
        Whether the format in which the categorical variables are cast should be
        ignored. If false, the encoder will automatically select variables of type
        object or categorical, or check that the variables entered by the user are of
        type object or categorical. If True, the encoder will select all variables or
        accept all variables entered by the user, including those cast as numeric.

    Attributes
    ----------
    encoder_dict_:
        Dictionary with the categories for which dummy variables will be created.

    variables_:
        The group of variables that will be transformed.

    variables_binary_:
        A list with binary variables identified from the data. That is, variables with
        only 2 categories.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Learn the unique categories per variable
    transform:
        Replace the categorical variables by the binary variables.
    fit_transform:
        Fit to the data, then transform it.

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

        if not isinstance(ignore_format, bool):
            raise ValueError("ignore_format takes only booleans True and False")

        self.top_categories = top_categories
        self.drop_last = drop_last
        self.drop_last_binary = drop_last_binary
        self.variables = _check_input_parameter_variables(variables)
        self.ignore_format = ignore_format

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

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame.
            - f user enters non-categorical variables (unless ignore_format is True)
        ValueError
            - If there are no categorical variables in the df or the df is empty
            - If the variable(s) contain null values

        Returns
        -------
        self
        """

        X = self._check_fit_input_and_variables(X)

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
                    self.encoder_dict_[var] = X[var].unique()

        self.variables_binary_ = [
            var for var in self.variables_ if X[var].nunique() == 2
        ]

        # automatically encode binary variables as 1 dummy
        if self.drop_last_binary:
            for var in self.variables_binary_:
                self.encoder_dict_[var] = X[var].unique()[0]

        self._check_encoding_dictionary()

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces the categorical variables by the binary variables.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values.
            - If dataframe has different number of features than the df used in fit()

        Returns
        -------
        X: pandas dataframe.
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
