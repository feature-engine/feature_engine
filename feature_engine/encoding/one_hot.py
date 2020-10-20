# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np

from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.variable_manipulation import _define_variables


class OneHotEncoder(BaseCategoricalTransformer):
    """
    One hot encoding consists in replacing the categorical variable by a
    combination of binary variables which take value 0 or 1, to indicate if
    a certain category is present in an observation.

    Each one of the binary variables are also known as dummy variables. For
    example, from the categorical variable "Gender" with categories 'female'
    and 'male', we can generate the boolean variable "female", which takes 1
    if the person is female or 0 otherwise. We can also generate the variable
    male, which takes 1 if the person is "male" and 0 otherwise.

    The encoder has the option to generate one dummy variable per category, or
    to create dummy variables only for the top n most popular categories, that is,
    the categories that are shown by the majority of the observations.

    If dummy variables are created for all the categories of a variable, you have
    the option to drop one category not to create information redundancy. That is,
    encoding into k-1 variables, where k is the number if unique categories.

    The encoder will encode only categorical variables (type 'object'). A list
    of variables can be passed as an argument. If no variables are passed as
    argument, the encoder will find and encode categorical variables (object type).

    The encoder first finds the categories to be encoded for each variable (fit).

    The encoder then creates one dummy variable per category for each variable
    (transform).

    Note: new categories in the data to transform, that is, those that did not appear
    in the training set, will be ignored (no binary variable will be created for them).

    Parameters
    ----------

    top_categories: int, default=None
        If None, a dummy variable will be created for each category of the variable.
        Alternatively, top_categories indicates the number of most frequent categories
        to encode. Dummy variables will be created only for those popular categories
        and the rest will be ignored. Note that this is equivalent to grouping all the
        remaining categories in one group.

    variables : list
        The list of categorical variables that will be encoded. If None, the
        encoder will find and select all object type variables.

    drop_last: boolean, default=False
        Only used if top_categories = None. It indicates whether to create dummy
        variables for all the categories (k dummies), or if set to True, it will
        ignore the last variable of the list (k-1 dummies).
    """

    def __init__(self, top_categories=None, variables=None, drop_last=False):

        if top_categories and not isinstance(top_categories, int):
            raise ValueError("top_categories takes only integer numbers, 1, 2, 3, etc.")

        if not isinstance(drop_last, bool):
            raise ValueError("drop_last takes only True or False")

        self.top_categories = top_categories
        self.drop_last = drop_last
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Learns the unique categories per variable. If top_categories is indicated,
        it will learn the most popular categories. Alternatively, it learns all
        unique categories per variable.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.

        y : pandas series, default=None
            Target. It is not needed in this encoded. You can pass y or
            None.

        Attributes
        ----------

        encoder_dict_: dictionary
            The dictionary containing the categories for which dummy variables
            will be created.
        """

        X = self._check_fit_input_and_variables(X)

        self.encoder_dict_ = {}

        for var in self.variables:
            if not self.top_categories:
                if self.drop_last:
                    category_ls = [x for x in X[var].unique()]
                    self.encoder_dict_[var] = category_ls[:-1]
                else:
                    self.encoder_dict_[var] = X[var].unique()

            else:
                self.encoder_dict_[var] = [
                    x
                    for x in X[var]
                    .value_counts()
                    .sort_values(ascending=False)
                    .head(self.top_categories)
                    .index
                ]

        self._check_encoding_dictionary()

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Creates the dummy / binary variables.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The data to transform.

        Returns
        -------

        X_transformed : pandas dataframe.
            The shape of the dataframe will be different from the original as it
            includes the dummy variables.
        """
        X = self._check_transform_input_and_state(X)

        for feature in self.variables:
            for category in self.encoder_dict_[feature]:
                X[str(feature) + "_" + str(category)] = np.where(
                    X[feature] == category, 1, 0
                )

        # drop the original non-encoded variables.
        X.drop(labels=self.variables, axis=1, inplace=True)

        return X
