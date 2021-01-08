# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Optional, List, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.encoding.ordinal import OrdinalEncoder
from feature_engine.discretisation import DecisionTreeDiscretiser
from feature_engine.variable_manipulation import _check_input_parameter_variables


class DecisionTreeEncoder(BaseCategoricalTransformer):
    """
    The DecisionTreeEncoder() encodes categorical variables with predictions
    of a decision tree model.

    Each categorical feature is recoded by training a decision tree, typically of
    limited depth (2, 3 or 4) using that feature alone, and let the tree directly
    predict the target. The probabilistic predictions of this decision tree are used as
    the new values of the original categorical feature, that now is linearly (or at
    least monotonically) correlated with the target.

    In practice, the categorical variable will be first encoded into integers with the
    OrdinalCategoricalEncoder(). The integers can be assigned arbitrarily to the
    categories or following the mean value of the target in each category. Then a
    decision tree will fit the resulting numerical variable to predict the target
    variable. Finally, the original categorical variable values will be replaced by the
    predictions of the decision tree.

    Note that a decision tree is fit per every single categorical variable to encode.

    Parameters
    ----------
    encoding_method : str, default='arbitrary'
        The categorical encoding method that will be used to encode the original
        categories to numerical values.

        'ordered': the categories are numbered in ascending order according to
        the target mean value per category.

        'arbitrary' : categories are numbered arbitrarily.

    cv : int, default=3
        Desired number of cross-validation fold to be used to fit the decision
        tree.

    scoring : str, default='neg_mean_squared_error'
        Desired metric to optimise the performance for the decision tree. Comes from
        sklearn.metrics. See the DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    regression : boolean, default=True
        Indicates whether the encoder should train a regression or a classification
        decision tree.

    param_grid : dictionary, default=None
        The list of parameters over which the decision tree should be optimised
        during the grid search. The param_grid can contain any of the permitted
        parameters for Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier().

        If None, then param_grid = {'max_depth': [1, 2, 3, 4]}.

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
    encoder_ :
        sklearn Pipeline containing the ordinal encoder and the decision tree.

    Methods
    -------
    fit:
        Fit a decision tree per variable.
    transform:
        Replace categorical variable by the predictions of the decision tree.
    fit_transform:
        Fit to the data, then transform it.

    Notes
    -----
    The authors designed this method originally, to work with numerical variables. We
    can replace numerical variables by the preditions of a decision tree utilising the
    DecisionTreeDiscretiser().

    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    See Also
    --------
    sklearn.ensemble.DecisionTreeRegressor
    sklearn.ensemble.DecisionTreeClassifier
    feature_engine.discretisation.DecisionTreeDiscretiser
    feature_engine.encoding.RareLabelEncoder

    References
    ----------
    .. [1] Niculescu-Mizil, et al. "Winning the KDD Cup Orange Challenge with Ensemble
        Selection". JMLR: Workshop and Conference Proceedings 7: 23-34. KDD 2009
        http://proceedings.mlr.press/v7/niculescu09/niculescu09.pdf
    """

    def __init__(
        self,
        encoding_method: str = "arbitrary",
        cv: int = 3,
        scoring: str = "neg_mean_squared_error",
        param_grid: Optional[dict] = None,
        regression: bool = True,
        random_state: Optional[int] = None,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if param_grid is None:
            param_grid = {"max_depth": [1, 2, 3, 4]}

        self.encoding_method = encoding_method
        self.cv = cv
        self.scoring = scoring
        self.regression = regression
        self.param_grid = param_grid
        self.random_state = random_state
        self.variables = _check_input_parameter_variables(variables)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit a decision tree per variable.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            categorical variables.

        y : pandas series.
            The target variable. Required to train the decision tree and for
            ordered ordinal encoding.

        Raises
        ------
        TypeError
            - If the input is not a Pandas DataFrame.
            - If any user provided variable is not categorical
        ValueError
            - If there are no categorical variables in the df or the df is empty
            - If the variable(s) contain null values

        Returns
        -------
        self
        """

        # check input dataframe
        X = self._check_fit_input_and_variables(X)

        # initialize categorical encoder
        cat_encoder = OrdinalEncoder(
            encoding_method=self.encoding_method, variables=self.variables
        )

        # initialize decision tree discretiser
        tree_discretiser = DecisionTreeDiscretiser(
            cv=self.cv,
            scoring=self.scoring,
            variables=self.variables,
            param_grid=self.param_grid,
            regression=self.regression,
            random_state=self.random_state,
        )

        # pipeline for the encoder
        self.encoder_ = Pipeline(
            [
                ("categorical_encoder", cat_encoder),
                ("tree_discretiser", tree_discretiser),
            ]
        )

        self.encoder_.fit(X, y)

        self.input_shape_ = X.shape

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace categorical variable by the predictions of the decision tree.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Raises
        ------
        TypeError
            If the input is not a Pandas DataFrame
        ValueError
            - If the variable(s) contain null values
            - If dataframe is not of same size as that used in fit()
        Warning
            If after encoding, NAN were introduced.

        Returns
        -------
        X : pandas dataframe of shape = [n_samples, n_features].
            Dataframe with variables encoded with decision tree predictions.
        """

        X = self._check_transform_input_and_state(X)

        X = self.encoder_.transform(X)

        return X

    def inverse_transform(self, X: pd.DataFrame):
        """inverse_transform is not implemented for this transformer yet."""
        return self
