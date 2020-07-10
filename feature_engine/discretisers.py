# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import pandas as pd

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from feature_engine.variable_manipulation import _define_variables
from feature_engine.base_transformers import BaseNumericalTransformer


class EqualFrequencyDiscretiser(BaseNumericalTransformer):
    """
    The EqualFrequencyDiscretiser() divides continuous numerical variables
    into contiguous equal frequency intervals, that is, intervals that contain
    approximately the same proportion of observations.
    
    The interval limits are determined using pandas.qcut(), in other words,
    the interval limits are determined by the quantiles. The number of intervals,  
    i.e., the number of quantiles in which the variable should be divided is
    determined by the user.
    
    The EqualFrequencyDiscretiser() works only with numerical variables.
    A list of variables can be passed as argument. Alternatively, the discretiser
    will automatically select and transform all numerical variables.
    
    The EqualFrequencyDiscretiser() first finds the boundaries for the intervals or
    quantiles for each variable, fit.
    
    Then it transforms the variables, that is, it sorts the values into the intervals,
    transform.
    
    Parameters
    ----------
    
    q : int, default=10
        Desired number of equal frequency intervals / bins. In other words the
        number of quantiles in which the variables should be divided.
    
    variables : list
        The list of numerical variables that will be discretised. If None, the 
        EqualFrequencyDiscretiser() will select all numerical variables.
        
    return_object : bool, default=False
        Whether the numbers in the discrete variable should be returned as
        numeric or as object. The decision is made by the user based on
        whether they would like to proceed the engineering of the variable as  
        if it was numerical or categorical.

    return_boundaries: bool, default=False
        whether the output should be the interval boundaries. If True, it returns
        the interval boundaries. If False, it returns integers.
    """

    def __init__(self, q=10, variables=None, return_object=False, return_boundaries=False):

        if not isinstance(q, int):
            raise ValueError('q must be an integer')

        if not isinstance(return_object, bool):
            raise ValueError('return_object must be True or False')

        if not isinstance(return_boundaries, bool):
            raise ValueError('return_boundaries must be True or False')

        self.q = q
        self.variables = _define_variables(variables)
        self.return_object = return_object
        self.return_boundaries = return_boundaries

    def fit(self, X, y=None):
        """
        Learns the limits of the equal frequency intervals, that is the 
        quantiles for each variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to be transformed.
        y : None
            y is not needed in this encoder. You can pass y or None.

        Attributes
        ----------

        binner_dict_: dictionary
            The dictionary containing the {variable: interval limits} pairs used
            to sort the values into discrete intervals.
        """
        # check input dataframe
        X = super().fit(X, y)

        self.binner_dict_ = {}

        for var in self.variables:
            tmp, bins = pd.qcut(x=X[var], q=self.q, retbins=True, duplicates='drop')

            # Prepend/Append infinities to accommodate outliers
            bins = list(bins)
            bins[0] = float("-inf")
            bins[len(bins) - 1] = float("inf")
            self.binner_dict_[var] = bins

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """ Sorts the variable values into the intervals.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the discrete variables.
        """
        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform variables
        if self.return_boundaries:
            for feature in self.variables:
                X[feature] = pd.cut(X[feature], self.binner_dict_[feature])

        else:
            for feature in self.variables:
                X[feature] = pd.cut(X[feature], self.binner_dict_[feature], labels=False)

            # return object
            if self.return_object:
                X[self.variables] = X[self.variables].astype('O')

        return X


class EqualWidthDiscretiser(BaseNumericalTransformer):
    """
    The EqualWidthDiscretiser() divides continuous numerical variables into
    intervals of the same width, that is, equidistant intervals. Note that the
    proportion of observations per interval may vary.
    
    The interval limits are determined using pandas.cut(). The number of intervals
    in which the variable should be divided must be indicated by the user.
    
    The EqualWidthDiscretiser() works only with numerical variables.
    A list of variables can be passed as argument. Alternatively, the discretiser
    will automatically select all numerical variables.
    
    The EqualWidthDiscretiser() first finds the boundaries for the intervals for
    each variable, fit.
    
    Then, it transforms the variables, that is, sorts the values into the intervals,
    transform.
    
    Parameters
    ----------
    
    bins : int, default=10
        Desired number of equal width intervals / bins.
    
    variables : list
        The list of numerical variables to transform. If None, the
        discretiser will automatically select all numerical type variables.
        
    return_object : bool, default=False
        Whether the numbers in the discrete variable should be returned as
        numeric or as object. The decision should be made by the user based on 
        whether they would like to proceed the engineering of the variable as  
        if it was numerical or categorical.

    return_boundaries: bool, default=False
        whether the output should be the interval boundaries. If True, it returns
        the interval boundaries. If False, it returns integers.
    """

    def __init__(self, bins=10, variables=None, return_object=False, return_boundaries=False):

        if not isinstance(bins, int):
            raise ValueError('q must be an integer')

        if not isinstance(return_object, bool):
            raise ValueError('return_object must be True or False')

        if not isinstance(return_boundaries, bool):
            raise ValueError('return_boundaries must be True or False')

        self.bins = bins
        self.variables = _define_variables(variables)
        self.return_object = return_object
        self.return_boundaries = return_boundaries

    def fit(self, X, y=None):
        """
        Learns the boundaries of the equal width intervals / bins for each
        variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.
        y : None
            y is not needed in this encoder. You can pass y or None.

        Attributes
        ----------

        binner_dict_: dictionary
            The dictionary containing the {variable: interval boundaries} pairs used
            to transform each variable.
            """
        # check input dataframe
        X = super().fit(X, y)

        # fit
        self.binner_dict_ = {}

        for var in self.variables:
            tmp, bins = pd.cut(x=X[var], bins=self.bins, retbins=True, duplicates='drop')

            # Prepend/Append infinities
            bins = list(bins)
            bins[0] = float("-inf")
            bins[len(bins) - 1] = float("inf")
            self.binner_dict_[var] = bins

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Sorts the variable values into the intervals.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the discrete variables.
        """
        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform variables
        if self.return_boundaries:
            for feature in self.variables:
                X[feature] = pd.cut(X[feature], self.binner_dict_[feature])

        else:
            for feature in self.variables:
                X[feature] = pd.cut(X[feature], self.binner_dict_[feature], labels=False)

            # return object
            if self.return_object:
                X[self.variables] = X[self.variables].astype('O')

        return X


class DecisionTreeDiscretiser(BaseNumericalTransformer):
    """
    The DecisionTreeDiscretiser() divides continuous numerical variables into discrete,
    finite, values estimated by a decision tree.
    
    The methods is inspired by the following article from the winners of the KDD
    2009 competition:
    http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf
        
    At the moment, this transformer only works for binary classification or
    regression. Multi-class classification is not supported.
    
    The DecisionTreeDiscretiser() works only with numerical variables.
    A list of variables can be passed as an argument. Alternatively, the
    discretiser will automatically select all numerical variables.
    
    The DecisionTreeDiscretiser() first trains a decision tree for each variable,
    fit.
    
    The DecisionTreeDiscretiser() then transforms the variables, that is,
    makes predictions based on the variable values, using the trained decision
    tree, transform.
    
    Parameters
    ----------
    
    cv : int, default=3
        Desired number of cross-validation fold to be used to fit the decision
        tree.
        
    scoring: str, default='neg_mean_squared_error'
        Desired metric to optimise the performance for the tree. Comes from
        sklearn metrics. See DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html
    
    variables : list
        The list of numerical variables that will be transformed. If None, the
        discretiser will automatically select all numerical type variables.
        
    regression : boolean, default=True
        Indicates whether the discretiser should train a regression or a classification
        decision tree.
        
    param_grid : dictionary, default={'max_depth': [1,2,3,4]}
        The list of parameters over which the decision tree should be optimised
        during the grid search. The param_grid can contain any of the permitted
        parameters for Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier().
        
    random_state : int, default=None
        The random_state to initialise the training of the decision tree. It is one
        of the parameters of the Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier(). For reproducibility it is recommended to set
        the random_state to an integer.
    """

    def __init__(self, cv=3, scoring='neg_mean_squared_error',
                 variables=None, param_grid={'max_depth': [1, 2, 3, 4]},
                 regression=True, random_state=None):

        if not isinstance(cv, int) or cv < 0:
            raise ValueError('cv can only take only positive integers')

        if not isinstance(regression, bool):
            raise ValueError('regression can only take True or False')

        self.cv = cv
        self.scoring = scoring
        self.regression = regression
        self.variables = _define_variables(variables)
        self.param_grid = param_grid
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fits the decision trees. One tree per variable to be transformed.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to transform.
        y : pandas series.
            Target variable. Required to train the decision tree.

        Attributes
        ----------

        binner_dict_: dictionary
            The dictionary containing the {variable: fitted tree} pairs.

        scores_dict_ : dictionary
            The score of the best decision tree, over the train set.
            Provided in case the user wishes to understand the performance of the
            decision tree.
            """

        # check input dataframe
        X = super().fit(X, y)

        self.binner_dict_ = {}
        self.scores_dict_ = {}

        for var in self.variables:
            # call the model
            if not self.regression:
                tree_model = GridSearchCV(DecisionTreeClassifier(random_state=self.random_state),
                                          cv=self.cv,
                                          scoring=self.scoring,
                                          param_grid=self.param_grid)
            else:
                tree_model = GridSearchCV(DecisionTreeRegressor(random_state=self.random_state),
                                          cv=self.cv,
                                          scoring=self.scoring,
                                          param_grid=self.param_grid)
            # fit the model to the variable
            tree_model.fit(X[var].to_frame(), y)

            self.binner_dict_[var] = tree_model
            self.scores_dict_[var] = tree_model.score(X[var].to_frame(), y)

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Returns the predictions of the tree, based of the variable original
        values. The tree outcome is finite, aka, discrete.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The dataframe with transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        for feature in self.variables:
            if not self.regression:
                tmp = self.binner_dict_[feature].predict_proba(X[feature].to_frame())
                X[feature] = tmp[:, 1]
            else:
                X[feature] = self.binner_dict_[feature].predict(X[feature].to_frame())

        return X


class UserInputDiscretiser(BaseNumericalTransformer):
    """
    The UserInputDiscretiser() divides continuous numerical variables
    into contiguous intervals are arbitrarily entered by the user.

    The user needs to enter a dictionary with variable names as keys, and a list of
    the limits of the intervals as values. For example {'var1':[0, 10, 100, 1000],
    'var2':[5, 10, 15, 20]}.

    The UserInputDiscretiser() works only with numerical variables. The discretiser will
    check if the dictionary entered by the user contains variables present in the training
    set, and if these variables are cast as numerical, before doing any transformation.

    Then it transforms the variables, that is, it sorts the values into the intervals,
    transform.

    Parameters
    ----------

    binning_dict : dict
        The dictionary with the variable : interval limits pairs, provided by the user. A
        valid dictionary looks like this: {'var1':[0, 10, 100, 1000], 'var2':[5, 10, 15, 20]}.

    return_object : bool, default=False
        Whether the numbers in the discrete variable should be returned as
        numeric or as object. The decision is made by the user based on
        whether they would like to proceed the engineering of the variable as
        if it was numerical or categorical.

    return_boundaries: bool, default=False
        whether the output should be the interval boundaries. If True, it returns
        the interval boundaries. If False, it returns integers.
    """

    def __init__(self, binning_dict, return_object=False, return_boundaries=False):

        if not isinstance(binning_dict, dict):
            raise ValueError("Please provide at a dictionary with the interval limits per variable")

        if not isinstance(return_object, bool):
            raise ValueError('return_object must be True or False')

        self.binning_dict = binning_dict
        self.variables = [x for x in binning_dict.keys()]
        self.return_object = return_object
        self.return_boundaries = return_boundaries

    def fit(self, X, y=None):
        """
        Checks that the user entered variables are in the train set and cast as numerical.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the variables to be transformed.

        y : None
            y is not needed in this encoder. You can pass y or None.

        Attributes
        ----------

        binner_dict_: dictionary
            The dictionary containing the {variable: interval limits} pairs used
            to sort the values into discrete intervals.
        """
        # check input dataframe
        X = super().fit(X, y)

        # check that all variables in the dictionary are present in the df
        tmp = [x for x in self.variables if x not in X.columns]
        if len(tmp) == 0:
            self.binner_dict_ = self.binning_dict
        else:
            raise ValueError('There are variables in the provided dictionary which are not present in the train set '
                             'or not cast as numerical')

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """ Sorts the variable values into the intervals.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------

        X_transformed : pandas dataframe of shape = [n_samples, n_features]
            The transformed data with the discrete variables.
        """
        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform variables
        if self.return_boundaries:
            for feature in self.variables:
                X[feature] = pd.cut(X[feature], self.binner_dict_[feature])

        else:
            for feature in self.variables:
                X[feature] = pd.cut(X[feature], self.binner_dict_[feature], labels=False)

            # return object
            if self.return_object:
                X[self.variables] = X[self.variables].astype('O')

        return X
