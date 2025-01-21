# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import check_classification_targets, type_of_target

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.tags import _return_tags


@Substitution(
    variables=_variables_numerical_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
)
class DecisionTreeDiscretiser(BaseNumericalTransformer):
    """
    The DecisionTreeDiscretiser() replaces numerical variables by discrete, i.e.,
    finite variables, whose values are the predictions of a decision tree, the  bin
    number, or the bin limits.

    The method is inspired by the following article from the winners of the KDD
    2009 competition:
    http://www.mtome.com/Publications/CiML/CiML-v3-book.pdf

    The DecisionTreeDiscretiser() trains a decision tree per variable. Then it finds
    the boundaries of each bin. Finally, it replaces the variable values with
    the predictions of the decision tree, the bin number, or the bin limits.

    The DecisionTreeDiscretiser() works only with numerical variables. You can pass a
    list with the variables you wish to transform. Alternatively, the discretiser will
    automatically select all numerical variables.

    More details in the :ref:`User Guide <decisiontree_discretiser>`.

    Parameters
    ----------
    {variables}

    bin_output: str, default = "prediction"
        Whether to return the predictions of the tree, the bin number, or the interval
        boundaries. Takes values "prediction", "bin_number" and "boundaries",
        respectively.

    precision: int, default=None
        The precision at which to store and display the bins labels. In other words,
        the number of decimals after the comma. Only used when `bin_output` is
        "prediction" or "boundaries". If `bin_output="boundaries"` then precision
        cannot be None.

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
        Desired metric to optimise the performance of the tree. Comes from
        sklearn.metrics. See the DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    param_grid: dictionary, default=None
        The hyperparameters for the decision tree to test with a grid search. The
        `param_grid` can contain any of the permitted hyperparameters for Scikit-learn's
        DecisionTreeRegressor() or DecisionTreeClassifier(). If None, then param_grid
        will optimise the 'max_depth' over `[1, 2, 3, 4]`.

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
         Dictionary with the interval limits per variable or the fitted tree per
         variable, depending on how `bin_output` was set up.

    scores_dict_:
        Dictionary with the score of the best decision tree per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Fit a decision tree per variable and find the interval limits.

    {fit_transform}

    transform:
        Sort continuous variables into intervals or replace them with the predictions.

    See Also
    --------
    sklearn.tree.DecisionTreeClassifier
    sklearn.tree.DecisionTreeRegressor

    References
    ----------
    .. [1] Niculescu-Mizil, et al. "Winning the KDD Cup Orange Challenge with Ensemble
        Selection". JMLR: Workshop and Conference Proceedings 7: 23-34. KDD 2009
        http://proceedings.mlr.press/v7/niculescu09/niculescu09.pdf

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from feature_engine.discretisation import DecisionTreeDiscretiser
    >>> np.random.seed(42)
    >>> X = pd.DataFrame(dict(x= np.random.randint(1,100, 100)))
    >>> y_reg = pd.Series(np.random.randn(100))
    >>> dtd = DecisionTreeDiscretiser(random_state=42)
    >>> dtd.fit(X, y_reg)
    >>> dtd.transform(X)["x"].value_counts()
    -0.090091    90
    0.479454    10
    Name: x, dtype: int64

    You can also apply this for classification problems adjusting the scoring metric.

    >>> y_clf = pd.Series(np.random.randint(0,2,100))
    >>> dtd = DecisionTreeDiscretiser(regression=False, scoring="f1", random_state=42)
    >>> dtd.fit(X, y_clf)
    >>> dtd.transform(X)["x"].value_counts()
    0.480769    52
    0.687500    48
    Name: x, dtype: int64
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        bin_output: str = "prediction",
        precision: Union[int, None] = None,
        cv=3,
        scoring: str = "neg_mean_squared_error",
        param_grid: Optional[Dict[str, Union[str, int, float, List[int]]]] = None,
        regression: bool = True,
        random_state: Optional[int] = None,
    ) -> None:

        if bin_output not in ["prediction", "bin_number", "boundaries"]:
            raise ValueError(
                "bin_output takes values  'prediction', 'bin_number' or 'boundaries'. "
                f"Got {bin_output} instead."
            )

        if precision is not None and (not isinstance(precision, int) or precision < 1):
            raise ValueError(
                "precision must be None or a positive integer. "
                f"Got {precision} instead."
            )

        if bin_output == "boundaries" and precision is None:
            raise ValueError(
                "When `bin_output == 'boundaries', `precision` cannot be None. "
                "Change precision's value to a positive integer."
            )
        if not isinstance(regression, bool):
            raise ValueError(
                f"regression can only take True or False. Got {regression} instead."
            )

        self.bin_output = bin_output
        self.precision = precision
        self.cv = cv
        self.scoring = scoring
        self.regression = regression
        self.variables = _check_variables_input_value(variables)
        self.param_grid = param_grid
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):  # type: ignore
        """
        Fit one decision tree per variable to discretize with cross-validation and
        grid-search for hyperparameters.

        Parameters
        ----------

        X: pandas dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y: pandas series.
            Target variable. Required to train the decision tree.
        """
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

        # check input dataframe
        X = super().fit(X)

        if self.param_grid:
            param_grid = self.param_grid
        else:
            param_grid = {"max_depth": [1, 2, 3, 4]}

        binner_dict_ = {}
        scores_dict_ = {}

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

            binner_dict_[var] = tree_model
            scores_dict_[var] = tree_model.score(X[var].to_frame(), y)

        if self.bin_output != "prediction":
            for var in self.variables_:
                clf = binner_dict_[var].best_estimator_
                threshold = clf.tree_.threshold
                feature = clf.tree_.feature
                feature_threshold = threshold[feature == 0]
                thresholds = sorted(feature_threshold)
                thresholds = [-np.inf] + thresholds + [np.inf]
                binner_dict_[var] = thresholds

        self.binner_dict_ = binner_dict_
        self.scores_dict_ = scores_dict_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces original variable values with the predictions of the tree. The
        decision tree predictions are finite, aka, discrete.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The dataframe with transformed variables.
        """

        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        if self.bin_output == "prediction":
            for feature in self.variables_:
                if self.regression:
                    preds = self.binner_dict_[feature].predict(X[feature].to_frame())
                    if self.precision is None:
                        X[feature] = preds
                    else:
                        X[feature] = np.round(preds, self.precision)
                else:
                    tmp = self.binner_dict_[feature].predict_proba(
                        X[feature].to_frame()
                    )
                    preds = tmp[:, 1]
                    if self.precision is None:
                        X[feature] = preds
                    else:
                        X[feature] = np.round(preds, self.precision)

        elif self.bin_output == "boundaries":
            for feature in self.variables_:
                X[feature] = pd.cut(
                    X[feature],
                    self.binner_dict_[feature],
                    precision=self.precision,
                    include_lowest=True,
                )
            X[self.variables_] = X[self.variables_].astype(str)

        else:
            for feature in self.variables_:
                X[feature] = pd.cut(
                    X[feature],
                    self.binner_dict_[feature],
                    labels=False,
                    include_lowest=True,
                )

        return X

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
