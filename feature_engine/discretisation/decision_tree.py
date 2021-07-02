# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Dict, List, Optional, Union

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from feature_engine.base_transformers import BaseNumericalTransformer
from feature_engine.variable_manipulation import (
    _check_input_parameter_variables,
    _find_or_check_numerical_variables,
)


class DecisionTreeDiscretiser(BaseNumericalTransformer):
    """
    The DecisionTreeDiscretiser() replaces continuous numerical variables by discrete,
    finite, values estimated by a decision tree.

    The methods is inspired by the following article from the winners of the KDD
    2009 competition:
    http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf

    The DecisionTreeDiscretiser() works only with numerical variables.
    A list of variables can be passed as an argument. Alternatively, the
    discretiser will automatically select all numerical variables.

    The DecisionTreeDiscretiser() first trains a decision tree for each variable.

    The DecisionTreeDiscretiser() then transforms the variables, that is,
    makes predictions based on the variable values, using the trained decision
    tree.

    Parameters
    ----------
    variables: list, default=None
        The list of numerical variables to transform. If None, the discretiser will
        automatically select all numerical variables.

    cv: int, default=3
        Desired number of cross-validation fold to be used to fit the decision
        tree.

    scoring: str, default='neg_mean_squared_error'
        Desired metric to optimise the performance for the tree. Comes from
        sklearn.metrics. See DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    param_grid: dictionary, default=None
        The list of parameters over which the decision tree should be optimised
        during the grid search. The param_grid can contain any of the permitted
        parameters for Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier().

        If None, then param_grid = {'max_depth': [1, 2, 3, 4]}

    regression: boolean, default=True
        Indicates whether the discretiser should train a regression or a classification
        decision tree.

    random_state : int, default=None
        The random_state to initialise the training of the decision tree. It is one
        of the parameters of the Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier(). For reproducibility it is recommended to set
        the random_state to an integer.

    Attributes
    ----------
    binner_dict_:
        Dictionary containing the fitted tree per variable.

    scores_dict_:
        Dictionary with the score of the best decision tree, over the train set.

    variables_:
         The variables to discretise.

    n_features_in_:
        The number of features in the train set used in fit.

    Methods
    -------
    fit:
        Fit a decision tree per variable.
    transform:
        Replace continuous values by the predictions of the decision tree.
    fit_transform:
        Fit to the data, then transform it.

    See Also
    --------
    sklearn.tree.DecisionTreeClassifier
    sklearn.tree.DecisionTreeRegressor

    References
    ----------
    .. [1] Niculescu-Mizil, et al. "Winning the KDD Cup Orange Challenge with Ensemble
        Selection". JMLR: Workshop and Conference Proceedings 7: 23-34. KDD 2009
        http://proceedings.mlr.press/v7/niculescu09/niculescu09.pdf

    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        cv: int = 3,
        scoring: str = "neg_mean_squared_error",
        param_grid: Optional[Dict[str, Union[str, int, float, List[int]]]] = None,
        regression: bool = True,
        random_state: Optional[int] = None,
    ) -> None:

        if not isinstance(cv, int) or cv < 0:
            raise ValueError("cv can only take only positive integers")

        if not isinstance(regression, bool):
            raise ValueError("regression can only take True or False")

        self.cv = cv
        self.scoring = scoring
        self.regression = regression
        self.variables = _check_input_parameter_variables(variables)
        self.param_grid = param_grid
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):  # type: ignore
        """
        Fit the decision trees. One tree per variable to be transformed.

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.
        y: pandas series.
            Target variable. Required to train the decision tree.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame
            - If any of the user provided variables are not numerical
        ValueError
            - If there are no numerical variables in the df or the df is empty
            - If the variable(s) contain null values

        Returns
        -------
        self
        """

        # check input dataframe
        X = super().fit(X, y)

        self.variables_ = _find_or_check_numerical_variables(X, self.variables)

        if self.param_grid:
            param_grid = self.param_grid
        else:
            param_grid = {"max_depth": [1, 2, 3, 4]}

        self.binner_dict_ = {}
        self.scores_dict_ = {}

        for var in self.variables_:

            if self.regression:
                model = DecisionTreeRegressor(random_state=self.random_state)
            else:
                model = DecisionTreeClassifier(random_state=self.random_state)

            tree_model = GridSearchCV(
                model, cv=self.cv, scoring=self.scoring, param_grid=param_grid
            )

            # fit the model to the variable
            tree_model.fit(X[var].to_frame(), y)

            self.binner_dict_[var] = tree_model
            self.scores_dict_[var] = tree_model.score(X[var].to_frame(), y)

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces original variable with the predictions of the tree. The tree outcome
        is finite, aka, discrete.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Raises
        ------
        TypeError
           If the input is not a Pandas DataFrame
        ValueError
           - If the variable(s) contain null values
           - If the dataframe is not of the same size as the one used in fit()

        Returns
        -------
        X_transformed: pandas dataframe of shape = [n_samples, n_features]
            The dataframe with transformed variables.
        """

        # check input dataframe and if class was fitted
        X = super().transform(X)

        for feature in self.variables_:
            if self.regression:
                X[feature] = self.binner_dict_[feature].predict(X[feature].to_frame())
            else:
                tmp = self.binner_dict_[feature].predict_proba(X[feature].to_frame())
                X[feature] = tmp[:, 1]

        return X
