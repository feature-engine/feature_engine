# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import List, Optional, Union

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import check_classification_targets, type_of_target

from feature_engine.discretisation import DecisionTreeDiscretiser
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.fit_attributes import (
    _variables_attribute_docstring,
    _feature_names_in_docstring,
    _n_features_in_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.encoding._docstrings import (
    _ignore_format_docstring,
    _variables_docstring,
)
from feature_engine.encoding.base_encoder import BaseCategoricalTransformer
from feature_engine.encoding.ordinal import OrdinalEncoder
from feature_engine.tags import _return_tags


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
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

    cv: int, cross-validation generator or an iterable, default=3
        Determines the cross-validation splitting strategy. Possible inputs for cv are:

            - None, to use cross_validate's default 5-fold cross validation

            - int, to specify the number of folds in a (Stratified)KFold,

            - CV splitter
                - (https://scikit-learn.org/stable/glossary.html#term-CV-splitter)

            - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and y is either binary or
        multiclass, StratifiedKFold is used. In all other cases, KFold is used. These
        splitters are instantiated with `shuffle=False` so the splits will be the same
        across calls. For more details check Scikit-learn's `cross_validate`'s
        documentation.

    scoring: str, default='neg_mean_squared_error'
        Desired metric to optimise the performance for the decision tree. Comes from
        sklearn.metrics. See the DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    param_grid: dictionary, default=None
        The hyperparameters for the decision tree to test with a grid search. The
        `param_grid` can contain any of the permitted hyperparameters for Scikit-learn's
        DecisionTreeRegressor() or DecisionTreeClassifier(). If None, then param_grid
        will optimise the 'max_depth' over `[1, 2, 3, 4]`.

    regression: boolean, default=True
        Indicates whether the encoder should train a regression or a classification
        decision tree.

    random_state: int, default=None
        The random_state to initialise the training of the decision tree. It is one
        of the parameters of the Scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier(). For reproducibility it is recommended to set
        the random_state to an integer.

    {variables}

    {ignore_format}

    Attributes
    ----------
    encoder_:
        sklearn Pipeline containing the ordinal encoder and the decision tree.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Fit a decision tree per variable.

    {fit_transform}

    transform:
        Replace categorical variable by the predictions of the decision tree.

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
        cv=3,
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

    def fit(self, X: pd.DataFrame, y: pd.Series):
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
        X, y = self._check_X_y(X, y)

        # confirm model type and target variables are compatible.
        if self.regression is True:
            if type_of_target(y) == "binary":
                raise ValueError(
                    "Trying to fit a regression to a binary target is not "
                    "allowed by this transformer. Check the target values "
                    "or set regression to False."
                )

        else:
            check_classification_targets(y)

        self._check_or_select_variables(X)
        self._get_feature_names_in(X)

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

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "categorical"
        tags_dict["requires_y"] = True
        # the below test will fail because sklearn requires to check for inf, but
        # you can't check inf of categorical data, numpy returns and error.
        # so we need to leave without this test
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict
