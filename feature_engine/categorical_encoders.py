# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df, _check_contains_na
from feature_engine.variable_manipulation import _find_categorical_variables, _define_variables
from feature_engine.base_transformers import BaseCategoricalTransformer


def _check_encoding_dictionary(dictionary):
    # check that there is a dictionary with category to number pairs
    if len(dictionary) == 0:
        raise ValueError('Encoder could not be fitted. Check the parameters and the variables '
                         'in your dataframe.')
    return dictionary


class CountFrequencyCategoricalEncoder(BaseCategoricalTransformer):
    """ 
    The CountFrequencyCategoricalEncoder() replaces categories by the count of
    observations per category or by the percentage of observations per category.
    
    For example in the variable colour, if 10 observations are blue, blue will
    be replaced by 10. Alternatively, if 10% of the observations are blue, blue
    will be replaced by 0.1.
    
    The CountFrequencyCategoricalEncoder() will encode only categorical variables
    (type 'object'). A list of variables to be encoded can be passed as argument.
    Alternatively, the encoder will find and encode all categorical variables
    (object type).
    
    The encoder first maps the categories to the numbers (counts or frequencies)
    for each variable (fit).

    The encoder then transforms the categories to those mapped numbers (transform).
    
    Parameters
    ----------
    
    encoding_method : str, default='count'
        Desired method of encoding.

        'count': number of observations per category

        'frequency': percentage of observations per category
    
    variables : list
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and transform all object type variables.
    """

    def __init__(self, encoding_method='count', variables=None):

        if encoding_method not in ['count', 'frequency']:
            raise ValueError("encoding_method takes only values 'count' and 'frequency'")

        self.encoding_method = encoding_method
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Learns the counts or frequencies which will be used to replace the categories.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            The user can pass the entire dataframe.

        y : None
            y is not needed in this encoder. You can pass y or None.

        Attributes
        ----------

        encoder_dict_: dictionary
            Dictionary containing the {category: count / frequency} pairs for
            each variable.
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find categorical variables or check that those entered by the user
        # are of type object
        self.variables = _find_categorical_variables(X, self.variables)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        self.encoder_dict_ = {}

        # learn encoding maps
        for var in self.variables:
            if self.encoding_method == 'count':
                self.encoder_dict_[var] = X[var].value_counts().to_dict()

            elif self.encoding_method == 'frequency':
                n_obs = np.float(len(X))
                self.encoder_dict_[var] = (X[var].value_counts() / n_obs).to_dict()

        self.encoder_dict_ = _check_encoding_dictionary(self.encoder_dict_)

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseCategoricalTransformer.transform.__doc__

    def inverse_transform(self, X):
        X = super().inverse_transform(X)
        return X

    inverse_transform.__doc__ = BaseCategoricalTransformer.inverse_transform.__doc__


class OrdinalCategoricalEncoder(BaseCategoricalTransformer):
    """ 
    The OrdinalCategoricalEncoder() replaces categories by ordinal numbers 
    (0, 1, 2, 3, etc). The numbers can be ordered based on the mean of the target
    per category, or assigned arbitrarily.
    
    Ordered ordinal encoding:  for the variable colour, if the mean of the target
    for blue, red and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 1,
    red by 2 and grey by 0.
    
    Arbitrary ordinal encoding: the numbers will be assigned arbitrarily to the
    categories, on a first seen first served basis.
    
    The encoder will encode only categorical variables (type 'object'). A list
    of variables can be passed as an argument. If no variables are passed, the
    encoder will find and encode all categorical variables (type 'object').
    
    The encoder first maps the categories to the numbers for each variable (fit).

    The encoder then transforms the categories to the mapped numbers (transform).
    
    Parameters
    ----------
    
    encoding_method : str, default='ordered' 
        Desired method of encoding.

        'ordered': the categories are numbered in ascending order according to
        the target mean value per category.

        'arbitrary' : categories are numbered arbitrarily.
        
    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and select all object type variables.
        
    Attributes
    ----------
    
    encoder_dict_: dictionary
        The dictionary containing the {category: ordinal number} pairs for
        every variable.
    """

    def __init__(self, encoding_method='ordered', variables=None):

        if encoding_method not in ['ordered', 'arbitrary']:
            raise ValueError("encoding_method takes only values 'ordered' and 'arbitrary'")

        self.encoding_method = encoding_method
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """ Learns the numbers to be used to replace the categories in each
        variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to be
            encoded.

        y : pandas series, default=None
            The Target. Can be None if encoding_method = 'arbitrary'.
            Otherwise, y needs to be passed when fitting the transformer.
       
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find categorical variables or check that those entered by the user
        # are of type object
        self.variables = _find_categorical_variables(X, self.variables)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        # join target to predictor variables
        if self.encoding_method == 'ordered':
            if y is None:
                raise ValueError('Please provide a target y for this encoding method')

            temp = pd.concat([X, y], axis=1)
            temp.columns = list(X.columns) + ['target']

        # find mappings
        self.encoder_dict_ = {}

        for var in self.variables:

            if self.encoding_method == 'ordered':
                t = temp.groupby([var])['target'].mean().sort_values(ascending=True).index

            elif self.encoding_method == 'arbitrary':
                t = X[var].unique()

            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        self.encoder_dict_ = _check_encoding_dictionary(self.encoder_dict_)

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseCategoricalTransformer.transform.__doc__

    def inverse_transform(self, X):
        X = super().inverse_transform(X)
        return X

    inverse_transform.__doc__ = BaseCategoricalTransformer.inverse_transform.__doc__


class MeanCategoricalEncoder(BaseCategoricalTransformer):
    """ 
    The MeanCategoricalEncoder() replaces categories by the mean value of the
    target for each category.
    
    For example in the variable colour, if the mean of the target for blue, red
    and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 0.5, red by 0.8
    and grey by 0.1.
    
    The encoder will encode only categorical variables (type 'object'). A list
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will find and encode all categorical variables
    (object type).
    
    The encoder first maps the categories to the numbers for each variable (fit).

    The encoder then transforms the categories to the mapped numbers (transform).
    
    Parameters
    ----------  
    
    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and select all object type variables.
    """

    def __init__(self, variables=None):
        self.variables = _define_variables(variables)

    def fit(self, X, y):
        """
        Learns the mean value of the target for each category of the variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to be encoded.

        y : pandas series
            The target.

        Attributes
        ----------

        encoder_dict_: dictionary
            The dictionary containing the {category: target mean} pairs used
            to replace categories in every variable.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find categorical variables or check that those entered by the user
        # are of type object
        self.variables = _find_categorical_variables(X, self.variables)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        if y is None:
            raise ValueError('Please provide a target y for this encoding method')

        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']

        self.encoder_dict_ = {}

        for var in self.variables:
            self.encoder_dict_[var] = temp.groupby(var)['target'].mean().to_dict()

        self.encoder_dict_ = _check_encoding_dictionary(self.encoder_dict_)

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseCategoricalTransformer.transform.__doc__

    def inverse_transform(self, X):
        X = super().inverse_transform(X)
        return X

    inverse_transform.__doc__ = BaseCategoricalTransformer.inverse_transform.__doc__


class WoERatioCategoricalEncoder(BaseCategoricalTransformer):
    """ 
    The WoERatioCategoricalEncoder() replaces categories by the weight of evidence
    or by the ratio between the probability of the target = 1 and the probability
    of the  target = 0.

    The weight of evidence is given by: np.log(P(X=xj|Y = 1)/P(X=xj|Y=0))
    
    The target probability ratio is given by: p(1) / p(0)

    And the log of the target probability ratio is: np.log( p(1) / p(0) )
    
    Note: This categorical encoding is exclusive for binary classification.
    
    For example in the variable colour, if the mean of the target = 1 for blue
    is 0.8 and the mean of the target = 0  is 0.2, blue will be replaced by:
    np.log(0.8/0.2) = 1.386 if log_ratio is selected. Alternatively, blue will be
    replaced by 0.8 / 0.2 = 4 if ratio is selected.

    For details on the calculation of the weight of evidence visit:
    https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    
    Note: the division by 0 is not defined and the log(0) is not defined.
    Thus, if p(0) = 0 for the ratio encoder, or either p(0) = 0 or p(1) = 0 for
    woe or log_ratio, in any of the variables, the encoder will return an error.
       
    The encoder will encode only categorical variables (type 'object'). A list
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will find and encode all categorical variables
    (object type).
    
    The encoder first maps the categories to the numbers for each variable (fit).

    The encoder then transforms the categories into the mapped numbers (transform).
    
    Parameters
    ----------
    
    encoding_method : str, default=woe
        Desired method of encoding.

        'woe': weight of evidence

        'ratio' : probability ratio
        
    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and select all object type variables.
    """

    def __init__(self, encoding_method='woe', variables=None):

        if encoding_method not in ['woe', 'ratio', 'log_ratio']:
            raise ValueError("encoding_method takes only values 'woe', 'ratio' and 'log_ratio'")

        self.encoding_method = encoding_method
        self.variables = _define_variables(variables)

    def fit(self, X, y):
        """
        Learns the numbers that should be used to replace the categories in each
        variable. That is the WoE or ratio of probability.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y : pandas series.
            Target, must be binary [0,1].

        Attributes
        ----------

        encoder_dict_: dictionary
            The dictionary containing the {category: WoE / ratio} pairs per variable.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find categorical variables or check that those entered by the user
        # are of type object
        self.variables = _find_categorical_variables(X, self.variables)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        if y is None:
            raise ValueError('Please provide a target y for this encoding method')

        # check that y is binary
        if len([x for x in y.unique() if x not in [0, 1]]) > 0:
            raise ValueError("This encoder is only designed for binary classification, values of y can be only 0 or 1")

        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ['target']

        self.encoder_dict_ = {}

        if self.encoding_method == 'woe':
            total_pos = temp['target'].sum()
            total_neg = len(temp) - total_pos
            temp['non_target'] = np.where(temp['target'] == 1, 0, 1)

            for var in self.variables:
                pos = temp.groupby([var])['target'].sum() / total_pos
                neg = temp.groupby([var])['non_target'].sum() / total_neg

                t = pd.concat([pos, neg], axis=1)
                t['woe'] = np.log(t['target'] / t['non_target'])

                if not t.loc[t['target'] == 0, :].empty or not t.loc[t['non_target'] == 0, :].empty:
                    raise ValueError(
                        "The proportion of 1 of the classes for a category in variable {} is zero, and log of zero is "
                        "not defined".format(var))

                self.encoder_dict_[var] = t['woe'].to_dict()

        else:
            for var in self.variables:
                t = temp.groupby(var)['target'].mean()
                t = pd.concat([t, 1 - t], axis=1)
                t.columns = ['p1', 'p0']

                if self.encoding_method == 'log_ratio':
                    if not t.loc[t['p0'] == 0, :].empty or not t.loc[t['p1'] == 0, :].empty:
                        raise ValueError(
                            "p(0) or p(1) for a category in variable {} is zero, log of zero is not defined".format(var))
                    else:
                        self.encoder_dict_[var] = (np.log(t.p1 / t.p0)).to_dict()

                elif self.encoding_method == 'ratio':
                    if not t.loc[t['p0'] == 0, :].empty:
                        raise ValueError(
                            "p(0) for a category in variable {} is zero, division by 0 is not defined".format(var))
                    else:
                        self.encoder_dict_[var] = (t.p1 / t.p0).to_dict()

        self.encoder_dict_ = _check_encoding_dictionary(self.encoder_dict_)

        self.input_shape_ = X.shape

        return self

    # Ugly work around to import the docstring for Sphinx, otherwise none of this is necessary
    def transform(self, X):
        X = super().transform(X)
        return X

    transform.__doc__ = BaseCategoricalTransformer.transform.__doc__

    def inverse_transform(self, X):
        X = super().inverse_transform(X)
        return X

    inverse_transform.__doc__ = BaseCategoricalTransformer.inverse_transform.__doc__


class OneHotCategoricalEncoder(BaseEstimator, TransformerMixin):
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

        if top_categories:
            if not isinstance(top_categories, int):
                raise ValueError("top_categories takes only integer numbers, 1, 2, 3, etc.")

        if drop_last not in [True, False]:
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
        # check input dataframe
        X = _is_dataframe(X)

        # find categorical variables or check that those entered by the user
        # are of type object
        self.variables = _find_categorical_variables(X, self.variables)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        self.encoder_dict_ = {}

        for var in self.variables:
            if not self.top_categories:
                if self.drop_last:
                    category_ls = [x for x in X[var].unique()]
                    self.encoder_dict_[var] = category_ls[:-1]
                else:
                    self.encoder_dict_[var] = X[var].unique()

            else:
                self.encoder_dict_[var] = [x for x in X[var].value_counts().sort_values(ascending=False).head(
                    self.top_categories).index]

        self.encoder_dict_ = _check_encoding_dictionary(self.encoder_dict_)

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
            The shape of the dataframe will be different from the original as it includes the dummy variables.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        # Check that the dataframe contains the same number of columns than the dataframe
        # used to fit the imputer.
        _check_input_matches_training_df(X, self.input_shape_[1])

        for feature in self.variables:
            for category in self.encoder_dict_[feature]:
                X[str(feature) + '_' + str(category)] = np.where(X[feature] == category, 1, 0)

        # drop the original non-encoded variables.
        X.drop(labels=self.variables, axis=1, inplace=True)

        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    The RareLabelCategoricalEncoder() groups rare / infrequent categories in
    a new category called "Rare", or any other name entered by the user.
    
    For example in the variable colour, if the percentage of observations
    for the categories magenta, cyan and burgundy are < 5 %, all those
    categories will be replaced by the new label "Rare".

    Note, infrequent labels can also be grouped under a user defined name, for
    example 'Other'. The name to replace infrequent categories is defined
    with the parameter replace_with.
       
    The encoder will encode only categorical variables (type 'object'). A list
    of variables can be passed as an argument. If no variables are passed as 
    argument, the encoder will find and encode all categorical variables
    (object type).
    
    The encoder first finds the frequent labels for each variable (fit).

    The encoder then groups the infrequent labels under the new label 'Rare'
    or by another user defined string (transform).
    
    Parameters
    ----------
    
    tol: float, default=0.05
        the minimum frequency a label should have to be considered frequent.
        Categories with frequencies lower than tol will be grouped.

    n_categories: int, default=10
        the minimum number of categories a variable should have for the encoder
        to find frequent labels. If the variable contains less categories, all
        of them will be considered frequent.

    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the 
        encoder will find and select all object type variables.

    replace_with : string, default='Rare'
        The category name that will be used to replace infrequent categories.

    return_object: bool, default=False
        Whether the variables should be re-cast as object, in case they have numerical values.
    """

    def __init__(self, tol=0.05, n_categories=10, variables=None, replace_with='Rare', return_object=False):

        if tol < 0 or tol > 1:
            raise ValueError("tol takes values between 0 and 1")

        if n_categories < 0 or not isinstance(n_categories, int):
            raise ValueError("n_categories takes only positive integer numbers")

        if not isinstance(replace_with, str):
            raise ValueError("replace_with takes only strings as values.")

        self.tol = tol
        self.n_categories = n_categories
        self.variables = _define_variables(variables)
        self.replace_with = replace_with
        self.return_object = return_object

    def fit(self, X, y=None):
        """
        Learns the frequent categories for each variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just selected variables

        y : None
            y is not required. You can pass y or None.

        Attributes
        ----------

        encoder_dict_: dictionary
            The dictionary containing the frequent categories (that will be kept)
            for each variable. Categories not present in this list will be replaced
            by 'Rare' or by the user defined value.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find categorical variables or check that those entered by the user
        # are of type object
        self.variables = _find_categorical_variables(X, self.variables)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        self.encoder_dict_ = {}

        for var in self.variables:
            if len(X[var].unique()) > self.n_categories:

                # if the variable has more than the indicated number of categories
                # the encoder will learn the most frequent categories
                t = pd.Series(X[var].value_counts() / np.float(len(X)))

                # non-rare labels:
                self.encoder_dict_[var] = t[t >= self.tol].index

            else:
                # if the total number of categories is smaller than the indicated
                # the encoder will consider all categories as frequent.
                warnings.warn("The number of unique categories for variable {} is less than that indicated in "
                              "n_categories. Thus, all categories will be considered frequent".format(var))
                self.encoder_dict_[var] = X[var].unique()

        self.encoder_dict_ = _check_encoding_dictionary(self.encoder_dict_)

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Groups rare labels under separate group 'Rare' or any other name provided
        by the user.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        
        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe where rare categories have been grouped.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        # Check that the dataframe contains the same number of columns than the dataframe
        # used to fit the imputer.
        _check_input_matches_training_df(X, self.input_shape_[1])

        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]), X[feature], self.replace_with)

        # add additional step to return variables cast as object
        if self.return_object:
            X[self.variables] = X[self.variables].astype('O')

        return X
