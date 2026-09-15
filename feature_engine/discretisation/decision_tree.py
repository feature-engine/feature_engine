# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause

from typing import Dict, List, Optional, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from joblib import Parallel, delayed
from narwhals.typing import IntoDataFrame, IntoSeries
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import check_classification_targets, type_of_target

from feature_engine._base_transformers.base_numerical import BaseNumericalTransformer
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_return_empty_is_bool,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _return_empty_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.methods import _fit_transform_docstring
from feature_engine._docstrings.substitute import Substitution
from feature_engine.tags import _return_tags


def _round_bin_edge(x: float, precision: int) -> float:
    """Round a bin edge the way pandas.cut historically formatted Interval labels:
    -inf/inf/0 pass through unrounded, and numbers with magnitude < 1 get extra
    decimals so that `precision` significant digits survive past the leading
    zeros (e.g. -0.0942 at precision=3 keeps 4 decimals, not 3, since
    round(-0.0942, 3) == -0.094 would only keep 2 significant digits).
    """
    if not np.isfinite(x) or x == 0:
        return x
    frac, whole = np.modf(x)
    if whole == 0:
        digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
    else:
        digits = precision
    return round(x, digits)


def _infer_bin_precision(thresholds: List[float], precision: int) -> int:
    """Find the smallest precision >= `precision` at which every rounded
    threshold is still distinct, mirroring pandas.cut's behaviour of bumping
    precision (for every edge, not just the colliding pair) when the requested
    precision would make two adjacent bin edges collide."""
    for prec in range(precision, 20):
        rounded = [_round_bin_edge(t, prec) for t in thresholds]
        if len(set(rounded)) == len(thresholds):
            return prec
    return precision


def _format_bin_edge(x: float, precision: int) -> str:
    if x == -np.inf:
        return "-inf"
    if x == np.inf:
        return "inf"
    return str(_round_bin_edge(x, precision))


def _bin_labels(thresholds: List[float], precision: int) -> List[str]:
    """Build the `(left, right]` interval label for every bin delimited by
    `thresholds`, which starts with -inf and ends with inf."""
    precision = _infer_bin_precision(thresholds, precision)
    edges = [_format_bin_edge(t, precision) for t in thresholds]
    return [f"({edges[i]}, {edges[i + 1]}]" for i in range(len(edges) - 1)]


def _bin_index(values: np.ndarray, thresholds: List[float]) -> np.ndarray:
    """Map each value to the 0-indexed bin delimited by `thresholds` (which
    starts with -inf and ends with inf), bins being closed on the right.
    """
    return np.digitize(values, thresholds[1:-1], right=True)


@Substitution(
    variables=_variables_numerical_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    return_empty=_return_empty_docstring,
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

    {return_empty}

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
        across calls. For more details check scikit-learn's `cross_validate`'s
        documentation.

    scoring: str, default='neg_mean_squared_error'
        Desired metric to optimise the performance of the tree. Comes from
        sklearn.metrics. See the DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    param_grid: dictionary, default=None
        The hyperparameters for the decision tree to test with a grid search. The
        `param_grid` can contain any of the permitted hyperparameters for scikit-learn's
        DecisionTreeRegressor() or DecisionTreeClassifier(). If None, then param_grid
        will optimise the 'max_depth' over `[1, 2, 3, 4]`.

    regression: boolean, default=True
        Indicates whether the discretiser should train a regression or a classification
        decision tree.

    random_state : int, default=None
        The random_state to initialise the training of the decision tree. It is one
        of the parameters of scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier(). For reproducibility it is recommended to set
        the random_state to an integer.

    n_jobs: int, default=None
        The number of jobs to run in parallel when training the decision trees
        across variables. Trees are fit using threads rather than processes,
        since fitting a decision tree releases the GIL for the bulk of its
        computation, which avoids the overhead of copying the entire dataframe
        to separate worker processes. `None` means 1, i.e. sequential training
        (this transformer's original behaviour); `-1` means using all available
        processors.

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
    x
    -0.090091    90
     0.479454    10
    Name: count, dtype: int64

    You can also apply this for classification problems adjusting the scoring metric.

    >>> y_clf = pd.Series(np.random.randint(0,2,100))
    >>> dtd = DecisionTreeDiscretiser(regression=False, scoring="f1", random_state=42)
    >>> dtd.fit(X, y_clf)
    >>> dtd.transform(X)["x"].value_counts()
    x
    0.480769    52
    0.687500    48
    Name: count, dtype: int64

    With polars:

    >>> import polars as pl
    >>> X = pl.DataFrame({"x": X["x"].to_list()})
    >>> dtd = DecisionTreeDiscretiser(random_state=42)
    >>> dtd.fit(X, y_reg)
    >>> dtd.transform(X)["x"].value_counts()
    shape: (2, 2)
    ┌───────────┬───────┐
    │ x         ┆ count │
    │ ---       ┆ ---   │
    │ f64       ┆ u32   │
    ╞═══════════╪═══════╡
    │ -0.090091 ┆ 90    │
    │ 0.479454  ┆ 10    │
    └───────────┴───────┘
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        return_empty: bool = False,
        bin_output: str = "prediction",
        precision: Union[int, None] = None,
        cv=3,
        scoring: str = "neg_mean_squared_error",
        param_grid: Optional[Dict[str, Union[str, int, float, List[int]]]] = None,
        regression: bool = True,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
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

        _check_return_empty_is_bool(return_empty)

        self.bin_output = bin_output
        self.precision = precision
        self.cv = cv
        self.scoring = scoring
        self.regression = regression
        self.variables = _check_variables_input_value(variables)
        self.param_grid = param_grid
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.return_empty = return_empty

    def fit(self, X: IntoDataFrame, y: IntoSeries):
        """
        Fit one decision tree per variable to discretise with cross-validation and
        grid-search for hyperparameters.

        Parameters
        ----------

        X: dataframe of shape = [n_samples, n_features]
            The training dataset. Can be the entire dataframe, not just the
            variables to be transformed.

        y: Series.
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
        X, variables_ = self._fit_setup(X)

        if self.param_grid:
            param_grid = self.param_grid
        else:
            param_grid = {"max_depth": [1, 2, 3, 4]}

        nw_X = nw.from_native(X, eager_only=True)
        X_subs = [nw_X.get_column(var).to_frame().to_native() for var in variables_]

        fitted = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._fit_one_tree)(X_sub, y, param_grid) for X_sub in X_subs
        )

        binner_dict_ = dict(zip(variables_, fitted))
        scores_dict_ = {
            var: tree_model.score(X_sub, y)
            for var, X_sub, tree_model in zip(variables_, X_subs, fitted)
        }

        if self.bin_output != "prediction":
            for var in variables_:
                clf = binner_dict_[var].best_estimator_
                threshold = clf.tree_.threshold
                feature = clf.tree_.feature
                feature_threshold = threshold[feature == 0]
                thresholds = sorted(feature_threshold)
                thresholds = [-np.inf] + thresholds + [np.inf]
                binner_dict_[var] = thresholds

        self.binner_dict_ = binner_dict_
        self.scores_dict_ = scores_dict_
        self.variables_ = variables_
        self._get_feature_names_in(X)

        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Replaces original variable values with the predictions of the tree. The
        decision tree predictions are finite, aka, discrete.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features]
            The dataframe with transformed variables.
        """
        # check input dataframe and if class was fitted
        X = self._check_transform_input_and_state(X)

        is_pandas = nwd.is_pandas_dataframe(X)
        nw_X = nw.from_native(X, eager_only=True)

        # build every replacement column before touching X, so pandas gets a
        # single non-mutating `.assign()` and polars a single `.with_columns()`
        # instead of one column swap per variable (avoids fragmentation and,
        # since check_X no longer copies pandas input, avoids mutating the
        # dataframe the caller passed in).
        new_columns: Dict[str, np.ndarray] = {}

        if self.bin_output == "prediction":
            for feature in self.variables_:
                X_sub = nw_X.get_column(feature).to_frame().to_native()
                if self.regression is True:
                    preds = self.binner_dict_[feature].predict(X_sub)
                else:
                    preds = self.binner_dict_[feature].predict_proba(X_sub)[:, 1]
                if self.precision is not None:
                    preds = np.round(preds, self.precision)
                new_columns[feature] = preds

        elif self.bin_output == "boundaries":
            # __init__ already guarantees precision is set when bin_output is
            # "boundaries"; assert narrows the type for mypy.
            assert self.precision is not None
            for feature in self.variables_:
                thresholds = self.binner_dict_[feature]
                labels = _bin_labels(thresholds, self.precision)
                values = nw_X.get_column(feature).to_numpy()
                bin_idx = _bin_index(values, thresholds)
                new_columns[feature] = np.array(labels)[bin_idx]

        else:
            for feature in self.variables_:
                thresholds = self.binner_dict_[feature]
                values = nw_X.get_column(feature).to_numpy()
                new_columns[feature] = _bin_index(values, thresholds)

        if is_pandas is True:
            X = X.assign(**new_columns)
        else:
            new_series = [
                nw.new_series(name, values, backend=nw_X.implementation)
                for name, values in new_columns.items()
            ]
            X = nw_X.with_columns(*new_series).to_native()

        return X

    def _fit_one_tree(self, X_sub: IntoDataFrame, y: IntoSeries, param_grid: Dict):
        """Instantiate and fit one decision tree on one variable."""
        if self.regression is True:
            model = DecisionTreeRegressor(random_state=self.random_state)
        else:
            model = DecisionTreeClassifier(random_state=self.random_state)

        tree_model = GridSearchCV(
            model, cv=self.cv, scoring=self.scoring, param_grid=param_grid
        )
        tree_model.fit(X_sub, y)
        return tree_model

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "numerical"
        tags_dict["requires_y"] = True
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
