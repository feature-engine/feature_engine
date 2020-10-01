# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from feature_engine.dataframe_checks import _is_dataframe, _check_input_matches_training_df, _check_contains_na
from feature_engine.variable_manipulation import _find_categorical_variables, _define_variables
from feature_engine.encoding.ordinal import OrdinalEncoder
from feature_engine.discretisation import DecisionTreeDiscretiser


class DecisionTreeEncoder(BaseEstimator, TransformerMixin):
    """
    The DecisionTreeCategoricalEncoder() encodes categorical variables with predictions of a decision tree model.

    The categorical variable will be first encoded into integers with the OrdinalCategoricalEncoder(). The
    integers can be assigned arbitrarily to the categories or following the mean value of the target in each category.

    Then a decision tree will be fit using the resulting numerical variable to predict the target  variable.
    Finally, the original categorical variable values will be replaced by the predictions of the decision
    tree.

    Parameters
    ----------

    encoding_method: str, default='arbitrary'
        The categorical encoding method that will be used to encode the original
        categories to numerical values.

        'ordered': the categories are numbered in ascending order according to
        the target mean value per category.

        'arbitrary' : categories are numbered arbitrarily.

    cv : int, default=3
        Desired number of cross-validation fold to be used to fit the decision
        tree.
        
    scoring: str, default='neg_mean_squared_error'
        Desired metric to optimise the performance for the tree. Comes from
        sklearn metrics. See the DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html
    
    regression : boolean, default=True
        Indicates whether the encoder should train a regression or a classification
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

    variables : list, default=None
        The list of categorical variables that will be encoded. If None, the
        encoder will find and select all object type variables.

    Attributes
    ----------

    encoder_ : sklearn Pipeline
        Encoder pipeline containing the ordinal encoder and decision
        tree discretiser.
    """

    def __init__(self, encoding_method='arbitrary', cv=3, scoring='neg_mean_squared_error',
                 param_grid={'max_depth': [1, 2, 3, 4]}, regression=True,
                 random_state=None, variables=None):
      
        self.encoding_method = encoding_method
        self.cv = cv
        self.scoring = scoring
        self.regression = regression
        self.param_grid = param_grid
        self.random_state = random_state
        self.variables = _define_variables(variables)

    def fit(self, X, y=None):
        """
        Learns the numbers that should be used to replace the categories in each
        variable.

        Parameters
        ----------

        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.
            Can be the entire dataframe, not just the categorical variables.

        y : pandas series.
            The target variable. Required to train the decision tree and for
            ordered ordinal encoding.
        """
        # check input dataframe
        X = _is_dataframe(X)

        # find categorical variables or check that variables entered by user are of type object
        self.variables = _find_categorical_variables(X, self.variables)

        # check if dataset contains na
        _check_contains_na(X, self.variables)

        # initialize categorical encoder
        cat_encoder = OrdinalEncoder(encoding_method=self.encoding_method,
                                                variables=self.variables)

        # initialize decision tree discretiser
        tree_discretiser = DecisionTreeDiscretiser(cv=self.cv, scoring=self.scoring,
                                                   variables=self.variables, param_grid=self.param_grid,
                                                   regression=self.regression, random_state=self.random_state)

        # pipeline for the encoder
        self.encoder_ = Pipeline([('categorical_encoder', cat_encoder),
                                  ('tree_discretiser', tree_discretiser)])

        self.encoder_.fit(X, y)

        self.input_shape_ = X.shape

        return self

    def transform(self, X):
        """
        Returns the predictions of the decision tree based of the variable's original value.
        
        Parameters
        ----------
        
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.
        
        Returns
        -------
        
        X_transformed : pandas dataframe of shape = [n_samples, n_features].
                        Dataframe with variables encoded with decision tree predictions.
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

        X = self.encoder_.transform(X)

        return X
