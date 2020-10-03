# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from feature_engine.variable_manipulation import _define_variables
from feature_engine.base_transformers import BaseNumericalTransformer


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
        
    param_grid : dictionary, default=None
        The list of parameters over which the decision tree should be optimised
        during the grid search. The param_grid can contain any of the permitted
        parameters for Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier().

        If None, then param_grid = {'max_depth': [1, 2, 3, 4]}
        
    random_state : int, default=None
        The random_state to initialise the training of the decision tree. It is one
        of the parameters of the Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier(). For reproducibility it is recommended to set
        the random_state to an integer.
    """

    def __init__(self, cv=3, scoring='neg_mean_squared_error',
                 variables=None, param_grid=None,
                 regression=True, random_state=None):

        if param_grid is None:
            param_grid = {'max_depth': [1, 2, 3, 4]}

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

            if self.regression:
                model = DecisionTreeRegressor(random_state=self.random_state)
            else:
                model = DecisionTreeClassifier(random_state=self.random_state)

            tree_model = GridSearchCV(model,
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
            if self.regression:
                X[feature] = self.binner_dict_[feature].predict(X[feature].to_frame())
            else:
                tmp = self.binner_dict_[feature].predict_proba(X[feature].to_frame())
                X[feature] = tmp[:, 1]

        return X
