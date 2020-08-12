# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engine.variable_manipulation import (
    _define_variables,
    _find_numerical_variables
)

from feature_engine.dataframe_checks import (
    _is_dataframe,
    _check_input_matches_training_df,
    _check_contains_na,
)


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    The Winsorizer() caps maximum and / or minimum values of a variable.
    
    The Winsorizer() works only with numerical variables. A list of variables can
    be indicated. Alternatively, the Winsorizer() will select all numerical
    variables in the train set.
    
    The Winsorizer() first calculates the capping values at the end of the
    distribution. The values are determined using 1) a Gaussian approximation,
    2) the inter-quantile range proximity rule or 3) percentiles.
    
    Gaussian limits:

        right tail: mean + 3* std

        left tail: mean - 3* std
        
    IQR limits:

        right tail: 75th quantile + 3* IQR

        left tail:  25th quantile - 3* IQR

    where IQR is the inter-quartile range: 75th quantile - 25th quantile.

    percentiles or quantiles:

        right tail: 95th percentile

        left tail:  5th percentile

    You can select how far out to cap the maximum or minimum values with the
    parameter 'fold'.

    If distribution='gaussian' fold gives the value to multiply the std.

    If distribution='skewed' fold is the value to multiply the IQR.

    If distribution='quantile', fold is the percentile on each tail that should
    be censored. For example, if fold=0.05, the limits will be the 5th and 95th
    percentiles. If fold=0.1, the limits will be the 10th and 90th percentiles.
    
    The transformer first finds the values at one or both tails of the distributions
    (fit).
    
    The transformer then caps the variables (transform).
    
    Parameters
    ----------
    
    distribution : str, default=gaussian
        Desired distribution. Can take 'gaussian', 'skewed' or 'quantiles'.

        gaussian: the transformer will find the maximum and / or minimum values to
        cap the variables using the Gaussian approximation.

        skewed: the transformer will find the boundaries using the IQR proximity rule.

        quantiles: the limits are given by the percentiles.
        
    tail : str, default=right
        Whether to cap outliers on the right, left or both tails of the distribution.
        Can take 'left', 'right' or 'both'.
        
    fold: int or float, default=3
        How far out to to place the capping values. The number that will multiply
        the std or IQR to calculate the capping values. Recommended values, 2 
        or 3 for the gaussian approximation, or 1.5 or 3 for the IQR proximity 
        rule.

        If distribution='quantile', then 'fold' indicates the percentile. So if
        fold=0.05, the limits will be the 95th and 5th percentiles.
        Note: Outliers will be removed up to a maximum of the 20th percentiles on both
        sides. Thus, when distribution='quantile', then 'fold' takes values between 0
        and 0.20.

    variables: list, default=None
        The list of variables for which the outliers will be capped. If None, 
        the transformer will find and select all numerical variables.

    missing_values: string, default='raise'
    	Indicates if missing values should be ignored or raised. Sometimes we want to remove
    	outliers in the raw, original data, sometimes, we may want to remove outliers in the
    	already pre-transformed data. If missing_values='ignore', the transformer will ignore
    	missing data when learning the capping parameters or transforming the data. If 
    	missing_values='raise' the transformer will return an error if the training or other
    	datasets contain missing values.
    """

    def __init__(self, distribution='gaussian', tail='right', fold=3, variables=None, missing_values='raise'):

        if distribution not in ['gaussian', 'skewed', 'quantiles']:
            raise ValueError("distribution takes only values 'gaussian', 'skewed' or 'quantiles'")

        if tail not in ['right', 'left', 'both']:
            raise ValueError("tail takes only values 'right', 'left' or 'both'")

        if fold <= 0:
            raise ValueError("fold takes only positive numbers")

        if distribution == 'quantiles' and fold > 0.2:
            raise ValueError("with distribution='quantiles', fold takes values between 0 and 0.20 only.")

        if missing_values not in ['raise', 'ignore']:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'")

        self.distribution = distribution
        self.tail = tail
        self.fold = fold
        self.variables = _define_variables(variables)
        self.missing_values = missing_values

    def fit(self, X, y=None):
        """ 
        Learns the values that should be used to replace outliers.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : None
            y is not needed in this transformer. You can pass y or None.

        Attributes
        ----------

        right_tail_caps_: dictionary
            The dictionary containing the maximum values at which variables
            will be capped.

        left_tail_caps_ : dictionary
            The dictionary containing the minimum values at which variables
            will be capped.
        """

        # check input dataframe
        X = _is_dataframe(X)

        # find or check for numerical variables
        self.variables = _find_numerical_variables(X, self.variables)

        if self.missing_values == 'raise':
            # check if dataset contains na
            _check_contains_na(X, self.variables)

        self.right_tail_caps_ = {}
        self.left_tail_caps_ = {}

        # estimate the end values
        if self.tail in ['right', 'both']:
            if self.distribution == 'gaussian':
                self.right_tail_caps_ = (X[self.variables].mean() + self.fold * X[self.variables].std()).to_dict()

            elif self.distribution == 'skewed':
                IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
                self.right_tail_caps_ = (X[self.variables].quantile(0.75) + (IQR * self.fold)).to_dict()

            elif self.distribution == 'quantiles':
                self.right_tail_caps_ = X[self.variables].quantile(1 - self.fold).to_dict()

        if self.tail in ['left', 'both']:
            if self.distribution == 'gaussian':
                self.left_tail_caps_ = (X[self.variables].mean() - self.fold * X[self.variables].std()).to_dict()

            elif self.distribution == 'skewed':
                IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
                self.left_tail_caps_ = (X[self.variables].quantile(0.25) - (IQR * self.fold)).to_dict()

            elif self.distribution == 'quantiles':
                self.left_tail_caps_ = X[self.variables].quantile(self.fold).to_dict()

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Caps the variable values, that is, censors outliers.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the capped variables.
        """

        # check input dataframe an if class was fitted
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        if self.missing_values == 'raise':
            # check if dataset contains na
            _check_contains_na(X, self.variables)

        # Check that the dataframe contains the same number of columns
        # than the dataframe used to fit the imputer.
        _check_input_matches_training_df(X, self.input_shape_[1])

        for feature in self.right_tail_caps_.keys():
            X[feature] = np.where(X[feature] > self.right_tail_caps_[feature], self.right_tail_caps_[feature],
                                  X[feature])

        for feature in self.left_tail_caps_.keys():
            X[feature] = np.where(X[feature] < self.left_tail_caps_[feature], self.left_tail_caps_[feature], X[feature])

        return X


class ArbitraryOutlierCapper(BaseEstimator, TransformerMixin):
    """ 
    The ArbitraryOutlierCapper() caps the maximum or minimum values of a variable
    by an arbitrary value indicated by the user.
       
    The user must provide the maximum or minimum values that will be used
    to cap each variable in a dictionary {feature:capping value}

    Parameters
    ----------
    
    capping_max : dictionary, default=None
        user specified capping values on right tail of the distribution (maximum
        values).

    capping_min : dictionary, default=None
        user specified capping values on left tail of the distribution (minimum
        values).

    missing_values : string, default='raise'
    	Indicates if missing values should be ignored or raised. If 
    	missing_values='raise' the transformer will return an error if the
    	training or other datasets contain missing values.        
    """

    def __init__(self, max_capping_dict=None, min_capping_dict=None, missing_values='raise'):

        if not max_capping_dict and not min_capping_dict:
            raise ValueError("Please provide at least 1 dictionary with the capping values per variable")

        if max_capping_dict is None or isinstance(max_capping_dict, dict):
            self.max_capping_dict = max_capping_dict
        else:
            raise ValueError("max_capping_dict should be a dictionary")

        if min_capping_dict is None or isinstance(min_capping_dict, dict):
            self.min_capping_dict = min_capping_dict
        else:
            raise ValueError("min_capping_dict should be a dictionary")

        if min_capping_dict is None:
            self.variables = [x for x in max_capping_dict.keys()]
        elif max_capping_dict is None:
            self.variables = [x for x in min_capping_dict.keys()]
        else:
            tmp = min_capping_dict.copy()
            tmp.update(max_capping_dict)
            self.variables = [x for x in tmp.keys()]

        if missing_values not in ['raise', 'ignore']:
            raise ValueError("missing_values takes only values 'raise' or 'ignore'")

        self.missing_values = missing_values

    def fit(self, X, y=None):
        """
        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : None
            y is not needed in this transformer. You can pass y or None.

        Attributes
        ----------

        right_tail_caps_: dictionary
            The dictionary containing the maximum values at which variables
            will be capped.

        left_tail_caps_ : dictionary
            The dictionary containing the minimum values at which variables
            will be capped.
        """
        X = _is_dataframe(X)

        if self.missing_values == 'raise':
            # check if dataset contains na
            _check_contains_na(X, self.variables)

        # find or check for numerical variables
        self.variables = _find_numerical_variables(X, self.variables)

        if self.max_capping_dict is not None:
            self.right_tail_caps_ = self.max_capping_dict
        else:
            self.right_tail_caps_ = {}

        if self.min_capping_dict is not None:
            self.left_tail_caps_ = self.min_capping_dict
        else:
            self.left_tail_caps_ = {}

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Caps the variable values, that is, censors outliers.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the capped variables.
        """

        # check if class was fitted
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        if self.missing_values == 'raise':
            # check if dataset contains na
            _check_contains_na(X, self.variables)

        # Check that the dataframe contains the same number of columns
        # than the dataframe used to fit the imputer.
        _check_input_matches_training_df(X, self.input_shape_[1])

        # replace outliers
        for feature in self.right_tail_caps_.keys():
            X[feature] = np.where(X[feature] > self.right_tail_caps_[feature], self.right_tail_caps_[feature],
                                  X[feature])

        for feature in self.left_tail_caps_.keys():
            X[feature] = np.where(X[feature] < self.left_tail_caps_[feature], self.left_tail_caps_[feature], X[feature])

        return X


class OutlierTrimmer(Winsorizer):
    """ The OutlierTrimmer() removes observations with outliers from the dataset.

    It works only with numerical variables. A list of variables can be indicated.
    Alternatively, the OutlierTrimmer() will select all numerical variables.

    The OutlierTrimmer() first calculates the maximum and /or minimum values
    beyond which a value will be considered an outlier, and thus removed.

    Limits are determined using 1) a Gaussian approximation, 2) the inter-quantile
    range proximity rule or 3) percentiles.

    Gaussian limits:

        right tail: mean + 3* std

        left tail: mean - 3* std

    IQR limits:

        right tail: 75th quantile + 3* IQR

        left tail:  25th quantile - 3* IQR

    where IQR is the inter-quartile range: 75th quantile - 25th quantile.

    percentiles or quantiles:

        right tail: 95th percentile

        left tail:  5th percentile

    You can select how far out to allow the maximum or minimum values with the
    parameter 'fold'.

    If distribution='gaussian' fold gives the value to multiply the std.

    If distribution='skewed' fold is the value to multiply the IQR.

    If distribution='quantile', fold is the percentile on each tail that should
    be censored. For example, if fold=0.05, the limits will be the 5th and 95th
    percentiles. If fold=0.1, the limits will be the 10th and 90th percentiles.

    The transformer first finds the values at one or both tails of the distributions
    (fit).

    The transformer then removes observations with outliers from the dataframe
    (transform).

    Parameters
    ----------

    distribution : str, default=gaussian
        Desired distribution. Can take 'gaussian', 'skewed' or 'quantiles'.

        gaussian: the transformer will find the maximum and / or minimum values to
        cap the variables using the Gaussian approximation.

        skewed: the transformer will find the boundaries using the IQR proximity rule.

        quantiles: the limits are given by the percentiles.

    tail : str, default=right
        Whether to cap outliers on the right, left or both tails of the distribution.
        Can take 'left', 'right' or 'both'.

    fold: int or float, default=3
        How far out to to place the capping values. The number that will multiply
        the std or IQR to calculate the capping values. Recommended values, 2
        or 3 for the gaussian approximation, or 1.5 or 3 for the IQR proximity
        rule.

        If distribution='quantile', then 'fold' indicates the percentile. So if
        fold=0.05, the limits will be the 95th and 5th percentiles.
        Note: Outliers will be removed up to a maximum of the 20th percentiles on both
        sides. Thus, when distribution='quantile', then 'fold' takes values between 0
        and 0.20.

    variables : list, default=None
        The list of variables for which the outliers will be capped. If None,
        the transformer will find and select all numerical variables.

    missing_values: string, default='raise'
    	Indicates if missing values should be ignored or raised. Sometimes we want to remove
    	outliers in the raw, original data, sometimes, we may want to remove outliers in the
    	already pre-transformed data. If missing_values='ignore', the transformer will ignore
    	missing data when learning the capping parameters or transforming the data. If 
    	missing_values='raise' the transformer will return an error if the training or other
    	datasets contain missing values.
    	
    """

    def transform(self, X):
        """
        Removes observations with outliers from the dataframe.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe without outlier observations.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = _is_dataframe(X)

        if self.missing_values == 'raise':
            # check if dataset contains na
            _check_contains_na(X, self.variables)

        # Check that the dataframe contains the same number of columns than the dataframe
        # used to fit the imputer.
        _check_input_matches_training_df(X, self.input_shape_[1])

        for feature in self.right_tail_caps_.keys():
            outliers = np.where(X[feature] > self.right_tail_caps_[feature], True, False)
            X = X.loc[~outliers]

        for feature in self.left_tail_caps_.keys():
            outliers = np.where(X[feature] < self.left_tail_caps_[feature], True, False)
            X = X.loc[~outliers]

        return X
