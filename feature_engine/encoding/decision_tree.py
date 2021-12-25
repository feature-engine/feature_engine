# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from feature_engine.discretisation import DecisionTreeDiscretiser
from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.encoding.ordinal import OrdinalEncoder


class DecisionTreeEncoder(BaseCategoricalTransformer):
    """
    The DecisionTreeEncoder() encodes categorical variables with predictions
    of a decision tree.

    The encoder first fits a decision tree using a single feature and the target (fit),
    and then replaces the values of the original feature by the predictions of the
    tree (transform). The transformer will train a decision tree per every feature to
    encode.

    The DecisionTreeEncoder() will encode only categorical variables by default
    (type 'object' or 'categorical'). You can pass a list of variables to encode or the
    encoder will find and encode all categorical variables.

    With `ignore_format=True` you have the option to encode numerical variables as
    well. In this case, you can either enter the list of variables to encode, or the
    transformer will automatically select all variables.

    More details in the :ref:`User Guide <decisiontree_encoder>`.

    Parameters
    ----------
    encoding_method: str, default='arbitrary'
        The method used to encode the categories to numerical values before fitting the
        decision tree.

        **'ordered'**: the categories are numbered in ascending order according to
        the target mean value per category.

        **'arbitrary'** : categories are numbered arbitrarily.

    cv: int, default=3
        Desired cross-validation fold to fit the decision tree.

    scoring: str, default='neg_mean_squared_error'
        Desired metric to optimise the performance for the decision tree. Comes from
        sklearn.metrics. See the DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    param_grid: dictionary, default=None
        The hyperparameters for the decision tree to test with a grid search. The
        `param_grid` can contain any of the permitted hyperparameters for Scikit-learn's
        DecisionTreeRegressor() or DecisionTreeClassifier(). If None, then
        `param_grid = {'max_depth': [1, 2, 3, 4]}`.

    regression: boolean, default=True
        Indicates whether the encoder should train a regression or a classification
        decision tree.

    random_state: int, default=None
        The random_state to initialise the training of the decision tree. It is one
        of the parameters of the Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier(). For reproducibility it is recommended to set
        the random_state to an integer.

    variables: list, default=None
        The list of categorical variables that will be encoded. If None, the
        encoder will find and transform all variables of type object or categorical by
        default. You can also make the transformer accept numerical variables, see the
        next parameter.

    ignore_format: bool, default=False
        Whether the format in which the categorical variables are cast should be
        ignored. If False, the encoder will automatically select variables of type
        object or categorical, or check that the variables entered by the user are of
        type object or categorical. If True, the encoder will select all variables or
        accept all variables entered by the user, including those cast as numeric.

    Attributes
    ----------
    encoder_:
        sklearn Pipeline containing the ordinal encoder and the decision tree.

    variables_:
        The group of variables that will be transformed.

    n_features_in_:
        The number of features in the train set used in fit.

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
    The authors designed this method originally to work with numerical variables. We
    can replace numerical variables by the predictions of a decision tree utilising the
    DecisionTreeDiscretiser(). Here we extend this functionality to work also with
    categorical variables.

    NAN are introduced when encoding categories that were not present in the training
    dataset. If this happens, try grouping infrequent categories using the
    RareLabelEncoder().

    See Also
    --------
    sklearn.ensemble.DecisionTreeRegressor
    sklearn.ensemble.DecisionTreeClassifier
    feature_engine.discretisation.DecisionTreeDiscretiser
    feature_engine.encoding.RareLabelEncoder
    feature_engine.encoding.OrdinalEncoder

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
        ignore_format: bool = False,
    ) -> None:

        super().__init__(variables, ignore_format)
        self.encoding_method = encoding_method
        self.cv = cv
        self.scoring = scoring
        self.regression = regression
        self.param_grid = param_grid
        self.random_state = random_state

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
        """

        # check input dataframe
        X = self._check_fit_input_and_variables(X)

        if self.param_grid:
            param_grid = self.param_grid
        else:
            param_grid = {"max_depth": [1, 2, 3, 4]}

        # initialize categorical encoder
        cat_encoder = OrdinalEncoder(
            encoding_method=self.encoding_method,
            variables=self.variables_,
            ignore_format=self.ignore_format,
            errors="raise",
        )

        # initialize decision tree discretiser
        tree_discretiser = DecisionTreeDiscretiser(
            cv=self.cv,
            scoring=self.scoring,
            variables=self.variables_,
            param_grid=param_grid,
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

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace categorical variables by the predictions of the decision tree.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_new : pandas dataframe of shape = [n_samples, n_features].
            Dataframe with variables encoded with decision tree predictions.
        """

        X = self._check_transform_input_and_state(X)

        X = self.encoder_.transform(X)

        return X

    def inverse_transform(self, X: pd.DataFrame):
        """inverse_transform is not implemented for this transformer."""
        return self
