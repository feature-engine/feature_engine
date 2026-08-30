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

from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_transformers import (
    _return_empty_docstring,
    _variables_categorical_docstring,
)
from feature_engine._docstrings.init_parameters.encoders import (
    _ignore_format_docstring,
    _unseen_docstring,
)
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _inverse_transform_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import _check_contains_na, check_X_y
from feature_engine.encoding._helper_functions import check_parameter_unseen
from feature_engine.encoding.base_encoder import (
    CategoricalInitMixin,
    CategoricalMethodsMixin,
)
from feature_engine.tags import _return_tags

_unseen_docstring = (
    _unseen_docstring
    + """ If `'encode'` unseen categories will be encoded as `fill_value`."""
)


@Substitution(
    ignore_format=_ignore_format_docstring,
    variables=_variables_categorical_docstring,
    return_empty=_return_empty_docstring,
    variables_=_variables_attribute_docstring,
    unseen=_unseen_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    fit_transform=_fit_transform_docstring,
    inverse_transform=_inverse_transform_docstring,
)
class DecisionTreeEncoder(CategoricalMethodsMixin, CategoricalInitMixin):
    """
    The DecisionTreeEncoder() encodes categorical variables with the predictions
    of a decision tree.

    The encoder fits a single feature decision tree to predict the target, and
    with that, it creates mappings from category to prediction value. Then, it uses
    these mappings to replace the categories of the feature. The encoder trains a
    decision tree per feature to encode.

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
        across calls. For more details check scikit-learn's `cross_validate`'s
        documentation.

    scoring: str, default='neg_mean_squared_error'
        Desired metric to optimise the performance for the decision tree. Comes from
        sklearn.metrics. See the DecisionTreeRegressor or DecisionTreeClassifier
        model evaluation documentation for more options:
        https://scikit-learn.org/stable/modules/model_evaluation.html

    param_grid: dictionary, default=None
        The hyperparameters for the decision tree to test with a grid search. The
        `param_grid` can contain any of the permitted hyperparameters for scikit-learn's
        DecisionTreeRegressor() or DecisionTreeClassifier(). If None, then param_grid
        will optimise the 'max_depth' over `[1, 2, 3, 4]`.

    regression: boolean, default=True
        Indicates whether the encoder should train a regression or a classification
        decision tree.

    random_state: int, default=None
        The random_state to initialise the training of the decision tree. It is one
        of the parameters of scikit-learn's DecisionTreeRegressor() or
        DecisionTreeClassifier(). For reproducibility it is recommended to set
        the random_state to an integer.

    {variables}

    {return_empty}

    {ignore_format}

    precision: int, default=None
        The precision at which to store and display the category mappings. In other
        words, the number of decimals after the comma for the tree predictions.

    {unseen}

    fill_value: float, default=None
        The value used to encode unseen categories. Only used when `unseen='encode'`.

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
    encoder_dict_:
        Dictionary with the prediction per category, per variable.

    {variables_}

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Fit a decision tree per variable.

    {fit_transform}

    {inverse_transform}

    transform:
        Replace categorical variable by the predictions of the decision tree.

    Notes
    -----
    The authors designed this method originally to work with numerical variables. We
    can replace numerical variables by the predictions of a decision tree utilising the
    DecisionTreeDiscretiser(). Here, we extend this functionality to work also with
    categorical variables.

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

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.encoding import DecisionTreeEncoder
    >>> X = pd.DataFrame(dict(x1 = [1,2,3,4,5], x2 = ["b", "b", "b", "a", "a"]))
    >>> y = pd.Series([2.2,4, 1.5, 3.2, 1.1])
    >>> dte = DecisionTreeEncoder(cv=2)
    >>> dte.fit(X, y)
    >>> dte.transform(X)
       x1        x2
    0   1  2.566667
    1   2  2.566667
    2   3  2.566667
    3   4  2.150000
    4   5  2.150000

    You can also use it for classification by using `regression=False`.

    >>> y = pd.Series([0,1,1,1,0])
    >>> dte = DecisionTreeEncoder(regression=False, cv=2)
    >>> dte.fit(X, y)
    >>> dte.transform(X)
       x1        x2
    0   1  0.666667
    1   2  0.666667
    2   3  0.666667
    3   4  0.500000
    4   5  0.500000

    With polars:

    >>> import polars as pl
    >>> X = pl.DataFrame(dict(x1 = [1,2,3,4,5], x2 = ["b", "b", "b", "a", "a"]))
    >>> y = [0, 1, 1, 1, 0]
    >>> dte = DecisionTreeEncoder(regression=False, cv=2)
    >>> dte.fit(X, y)
    >>> dte.transform(X)
    shape: (5, 2)
    ┌─────┬──────────┐
    │ x1  ┆ x2       │
    │ --- ┆ ---      │
    │ i64 ┆ f64      │
    ╞═════╪══════════╡
    │ 1   ┆ 0.666667 │
    │ 2   ┆ 0.666667 │
    │ 3   ┆ 0.666667 │
    │ 4   ┆ 0.5      │
    │ 5   ┆ 0.5      │
    └─────┴──────────┘
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
        return_empty: bool = False,
        ignore_format: bool = False,
        precision: Optional[int] = None,
        unseen: str = "ignore",
        fill_value: Optional[float] = None,
        n_jobs: Optional[int] = None,
    ) -> None:

        if encoding_method not in ["ordered", "arbitrary"]:
            raise ValueError(
                "`encoding_method` takes only values 'ordered' and 'arbitrary'."
                f" Got {encoding_method} instead."
            )

        if unseen == "encode" and (
            fill_value is None or not isinstance(fill_value, (int, float))
        ):
            raise ValueError(
                "When `unseen='encode'` you need to pass a number to `fill_value`. "
                f"Got {fill_value} instead."
            )

        if precision is not None and (not isinstance(precision, int) or precision < 0):
            raise ValueError(
                "Parameter `precision` takes integers or None. "
                f"Got {precision} instead."
            )

        check_parameter_unseen(unseen, ["ignore", "raise", "encode"])
        super().__init__(variables, return_empty, ignore_format)
        self.encoding_method = encoding_method
        self.cv = cv
        self.scoring = scoring
        self.regression = regression
        self.param_grid = param_grid
        self.random_state = random_state
        self.precision = precision
        self.unseen = unseen
        self.fill_value = fill_value
        self.n_jobs = n_jobs

    def fit(self, X: IntoDataFrame, y: IntoSeries):
        """
        Fit a decision tree per variable.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just the
            categorical variables.

        y: Series.
            The target variable. Required to train the decision tree and for
            ordered ordinal encoding.
        """
        nw_X, y = check_X_y(X, y)

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

        variables_ = self._check_or_select_variables(X)
        _check_contains_na(X, variables_)

        param_grid = self._assign_param_grid()

        # if the list of variables to transform is empty, we
        # stop the logic.
        if isinstance(variables_, list) and len(variables_) == 0:
            self.encoder_dict_ = {}
            self.variables_ = variables_
            self._get_feature_names_in(X)
            return self

        # only needed for "ordered": pairs the target with X once so every
        # variable's group_by below can reuse it, instead of rebuilding it
        # per variable.
        nw_Xy = None
        target_name = "__feature_engine_decision_tree_target__"
        if self.encoding_method == "ordered":
            if nwd.is_into_series(y):
                y_nw = nw.from_native(y, series_only=True).alias(target_name)
            else:
                y_nw = nw.new_series(
                    name=target_name, values=y, backend=nw_X.implementation
                )
            nw_Xy = nw_X.with_columns(y_nw)

        mappings = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._fit_one_variable)(
                nw_X, nw_Xy, var, y, target_name, param_grid
            )
            for var in variables_
        )

        if self.unseen == "encode":
            self._unseen = self.fill_value

        self.encoder_dict_ = dict(zip(variables_, mappings))
        self.variables_ = variables_
        self._get_feature_names_in(X)
        return self

    def transform(self, X: IntoDataFrame) -> IntoDataFrame:
        """
        Replace categorical variables by the predictions of the decision tree.

        Parameters
        ----------
        X: dataframe of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_new: dataframe of shape = [n_samples, n_features].
            Dataframe with variables encoded with decision tree predictions.
        """
        nw_X = self._check_transform_input_and_state(X)
        _check_contains_na(X, self.variables_)
        X = self._encode(nw_X)

        return X

    def _fit_one_variable(
        self, nw_X, nw_Xy, var, y: IntoSeries, target_name: str, param_grid: Dict
    ) -> Dict:
        """Learn the category-to-prediction mapping for one variable: encode its
        categories to ordinal integers, fit a decision tree on those integers, and
        predict on each unique category to get its final mapped value."""
        if self.encoding_method == "ordered":
            # sort by (mean, category): group_by's own order isn't guaranteed
            # across backends, and this tie-break on the category itself
            # reproduces pandas' groupby(sort=True) + stable sort_values
            # behavior for categories with equal target means.
            categories = (
                nw_Xy.group_by(var, drop_null_keys=True)
                .agg(nw.col(target_name).mean())
                .sort([target_name, var])
                .get_column(var)
                .to_list()
            )
        else:
            categories = nw_X.get_column(var).unique(maintain_order=True).to_list()

        ordinal_map = {k: i for i, k in enumerate(categories, 0)}

        X_sub = nw_X.get_column(var).replace_strict(ordinal_map).to_frame().to_native()
        estimator = self._fit_one_tree(X_sub, y, param_grid)

        # predict directly on the (few) unique ordinal codes instead of the
        # full column: the tree's prediction for a category depends only on
        # its ordinal code, so this gives identical results far cheaper.
        X_pred = nw.new_series(
            var, list(range(len(categories))), backend=nw_X.implementation
        ).to_frame().to_native()

        if self.regression is True:
            preds = estimator.predict(X_pred)
        else:
            preds = estimator.predict_proba(X_pred)[:, 1]

        if self.precision is not None:
            preds = np.round(preds, self.precision)

        return dict(zip(categories, preds))

    def _fit_one_tree(self, X_sub: IntoDataFrame, y: IntoSeries, param_grid: Dict):
        """Instantiate and fit one decision tree on one variable's ordinal-encoded
        values."""
        if self.regression is True:
            model = DecisionTreeRegressor(random_state=self.random_state)
        else:
            model = DecisionTreeClassifier(random_state=self.random_state)

        tree_model = GridSearchCV(
            model, cv=self.cv, scoring=self.scoring, param_grid=param_grid
        )
        tree_model.fit(X_sub, y)
        return tree_model

    def _assign_param_grid(self):
        if self.param_grid:
            param_grid = self.param_grid
        else:
            param_grid = {"max_depth": [1, 2, 3, 4]}
        return param_grid

    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["variables"] = "categorical"
        tags_dict["requires_y"] = True
        # the below test will fail because sklearn requires to check for inf, but
        # you can't check inf of categorical data, numpy returns and error.
        # so we need to leave without this test
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        return tags_dict
