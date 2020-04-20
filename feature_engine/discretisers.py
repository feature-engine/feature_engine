# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

import pandas as pd

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from feature_engine.utils import _define_variables
from feature_engine.base_transformers import BaseNumericalTransformer


class EqualFrequencyDiscretiser(BaseNumericalTransformer):
    """
    The EqualFrequencyDiscretiser() divides the numerical variable values 
    into contiguous equal frequency intervals, that is, intervals that contain
    approximately the same proportion of observations.
    
    The interval limits are determined using pandas.qcut(), in other words,
    the interval limits are determined by the quantiles. The number of intervals,  
    i.e., the number of quantiles in which the variable should be divided is
    determined by the user.
    
    The EqualFrequencyDiscretiser() will binnarise only numerical variables. 
    A list of variables can be passed as an argument. But, if no variables are 
    indicated, the discretiser will automatically select and binnarise numerical 
    variables and ignore the rest.
    
    The EqualFrequencyDiscretiser() must be fit to a training set, so that it
    can first find the boundaries for the intervals / quantiles for each variable
    ( fit() ).
    
    The EqualFrequencyDiscretiser() can then transform the variables, that is,
    sort the values into the intervals ( transform() ).
    
    Parameters
    ----------
    
    q : int, default=10
        Desired number of equal frequency intervals / bins. In other words the
        number of quantiles in which the variable should be divided.
    
    variables : list
        The list of numerical variables that will be discretised. If None, the 
        EqualFrequencyDiscretiser() will select all numerical variables.
        
    return_object : bool, default=False
        Whether the numbers in the discretised variable should be returned as
        numeric or as object. The decision should be made by the user based on 
        whether they would like to proceed the engineering of the variable as  
        if it was numerical or categorical.
        
    Attributes
    ----------
    
    binner_dict_: dictionary
        The dictionary containing the {interval limits: variable} pairs used
        to binnarise / discretise variable.
    """

    def __init__(self, q=10, variables=None, return_object=False):

        if not isinstance(q, int):
            raise ValueError('q must be an integer')

        if not isinstance(return_object, bool):
            raise ValueError('return_object must be True or False')

        self.q = q
        self.variables = _define_variables(variables)
        self.return_object = return_object

    def fit(self, X, y=None):
        """
        Learns the limits of the equal frequency intervals, that is the 
        quantiles for each variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this encoder, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
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
        '''

        :param X:
        :return:
        '''
        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform variables
        for feature in self.variables:
            X[feature] = pd.cut(X[feature], self.binner_dict_[feature], labels=False)

        # return object
        if self.return_object:
            X[self.variables] = X[self.variables].astype('O')

        return X


class EqualWidthDiscretiser(BaseNumericalTransformer):
    """
    The EqualWidthDiscretiser() divides the numerical variable values into 
    intervals of the same width, that is equi-distant intervals. Note that the 
    proportion of observations per interval may vary.
    
    The interval limits are determined using pandas.cut(). The number of intervals
    in which the variable should be divided must be indicated by the user.
    
    The EqualWidthDiscretiser() will binnarise only numerical variables. 
    A list of variables can be passed as an argument. But, if no variables are 
    indicated, the discretiser will automatically select and binnarise numerical 
    variables and ignore the rest.
    
    The EqualWidthDiscretiser() must be fit to a training set, so that it
    can first find the boundaries for the intervals for each variable
    ( fit() ).
    
    The EqualWidthDiscretiser() can then transform the variables, that is,
    sort the values into the intervals ( transform() ).
    
    Parameters
    ----------
    
    bins : int, default=10
        Desired number of equal widht buckets / bins.
    
    variables : list
        The list of numerical variables that will be discretised. If None, the
        discretiser will automatically select all numerical type variables.
        
    return_object : bool, default=False
        Whether the numbers in the discretised variable should be returned as
        numeric or as object. The decision should be made by the user based on 
        whether they would like to proceed the engineering of the variable as  
        if it was numerical or categorical.
        
    Attributes
    ----------
    
    binner_dict_: dictionary
        The dictionary containing the {interval boundaries: variable} pairs used
        to binnarise / discretise each variable.
    """

    def __init__(self, bins=10, variables=None, return_object=False):

        if not isinstance(bins, int):
            raise ValueError('q must be an integer')

        if not isinstance(return_object, bool):
            raise ValueError('return_object must be True or False')

        self.bins = bins
        self.variables = _define_variables(variables)
        self.return_object = return_object

    def fit(self, X, y=None):
        """
        Learns the boundaries of the equal width intervals / bins for each
        variable.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
        y : None
            y is not needed in this encoder, yet the sklearn pipeline API
            requires this parameter for checking. You can either leave it as None
            or pass y.
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
        '''

        :param X:
        :return:
        '''
        # check input dataframe and if class was fitted
        X = super().transform(X)

        # transform variables
        for feature in self.variables:
            X[feature] = pd.cut(X[feature], self.binner_dict_[feature], labels=False)

        # return object
        if self.return_object:
            X[self.variables] = X[self.variables].astype('O')

        return X


class DecisionTreeDiscretiser(BaseNumericalTransformer):
    """
    The DecisionTreeDiscretiser() divides the numerical variable into groups
    estimated by a decision tree. In other words, the intervals are the predictions
    made by a decision tree.
    
    The methods is inspired by the following article from the winners of the KDD
    2009 competition:
    
    http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf
        
    At the moment, the discretiser only works for binary classification or
    regression. Multiclass classification is not supported.
    
    The DecisionTreeDiscretiser() will binnarise only numerical variables. 
    A list of variables can be passed as an argument. But, if no variables are 
    indicated, the discretiser will automatically select and binnarise numerical 
    variables and ignore the rest.
    
    The DecisionTreeDiscretiser() must be fit to a training set, so that it
    can first train a decision tree for each variable ( fit() ).
    
    The DecisionTreeDiscretiser() can then transform the variables, that is,
    make predictions based on the variable values, using the trained decision
    tree ( transform() ).
    
    Parameters
    ----------
    
    cv : int, default=3
        Desired number of cross-validation fold to be used to fit the decision
        tree for each variable.
        
    scoring: str, default='neg_mean_squared_error'
        Desired metric to optimise the performance for the tree. Comes from
        sklearn metrics. See DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation  for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html
    
    variables : list
        The list of numerical variables that will be discretised. If None, the 
        discretiser will automatically select all numerical type variables.
        
    regression : boolean, default=True
        Indicates whether the discretiser should train a regression or a classification
        decision tree.
        
    param_grid : dictionary, default={'max_depth': [1,2,3,4]}
        The list of parameters over which the decision tree should be optimised
        during the grid search for the best tree. The param_grid can contain any
        of the permitted parameters for Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier().
        
    random_state : int, default=None
        The random_state to initilise the training of the decision tree. It is one
        of the normal parameters of the Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier(). For reproducibility it is recommended to set
        the random_state to an integer.
        
    Attributes
    ----------
    
    binner_dict_: dictionary
        The dictionary containing the {fitted tree: variable} pairs, used
        to transform each variable.
        
    scores_dict_ : dictionary
        The score of the best decision tree, over the train set.
        Provided in case the user wishes to understand the performance of the 
        decision tree.
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
        Fits the decision tree.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just seleted variables.
        y : target variable. Required for this transformer to train the decision 
            tree.
        """

        # if y is None:
        #     raise ValueError('Please provide a target (y) for this discretiser')

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
        Discretises the variables using the trained tree. That is, returns 
        the predictions of the tree, based of the variable values.
        
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
