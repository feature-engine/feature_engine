import itertools
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted

from feature_engine._base_transformers.mixins import GetFeatureNamesOutMixin
from feature_engine._check_init_parameters.check_init_input_params import (
    _check_param_drop_original,
    _check_param_missing_values,
)
from feature_engine._check_init_parameters.check_variables import (
    _check_variables_input_value,
)
from feature_engine._docstrings.fit_attributes import (
    _feature_names_in_docstring,
    _n_features_in_docstring,
    _variables_attribute_docstring,
)
from feature_engine._docstrings.init_parameters.all_trasnformers import (
    _drop_original_docstring,
    _missing_values_docstring,
    _variables_numerical_docstring,
)
from feature_engine._docstrings.init_parameters.creation import _features_to_combine
from feature_engine._docstrings.methods import (
    _fit_transform_docstring,
    _transform_creation_docstring,
)
from feature_engine._docstrings.substitute import Substitution
from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
    check_X_y,
)
from feature_engine.tags import _return_tags
from feature_engine.variable_handling import (
    check_numerical_variables,
    find_numerical_variables,
)


@Substitution(
    variables=_variables_numerical_docstring,
    features_to_combine=_features_to_combine,
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    transform=_transform_creation_docstring,
    fit_transform=_fit_transform_docstring,
)
class DecisionTreeFeatures(TransformerMixin, BaseEstimator, GetFeatureNamesOutMixin):
    """
    `DecisionTreeFeatures()` adds new variables to the data that result of the output of
    decision trees trained with one or more features.

    Features that result from the predictions of decision trees are likely monotonic
    with the target and therefore could improve the performance of linear models.
    Features that result from decision trees trained on various features can help
    capture feature interactions that could otherwise be missed by simpler models.

    `DecisionTreeFeatures()` works only with numerical variables. You can pass a list of
    variables to use as inputs of the decision trees. Alternatively, the transformer
    will automatically select and combine all numerical variables.

    Missing data should be imputed before using this transformer.

    More details in the :ref:`User Guide <dtree_features>`.

    Parameters
    ----------
    {variables}

    {features_to_combine}

    precision: int, default=None
        The precision of the predictions. In other words, the number of decimals after
        the comma for the new feature values.

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

    {missing_values}

    {drop_original}

    Attributes
    ----------
    {variables_}

    input_features_ = list
        List containing all the feature combinations that are used to create new
        features.

    estimators_: List
        The decision trees trained on the feature combinations.

    {feature_names_in_}

    {n_features_in_}

    Methods
    -------
    fit:
        Trains the decision trees.

    {fit_transform}

    {transform}

    See Also
    --------
    feature_engine.discretisation.DecisionTreeDiscretiser
    feature_engine.encoding.DecisionTreeEncoder
    sklearn.tree.DecisionTreeClassifier
    sklearn.tree.DecisionTreeRegressor

    References
    ----------
    .. [1] Niculescu-Mizil, et al. "Winning the KDD Cup Orange Challenge with Ensemble
        Selection". JMLR: Workshop and Conference Proceedings 7: 23-34. KDD 2009
        http://proceedings.mlr.press/v7/niculescu09/niculescu09.pdf

    Examples
    --------

    >>> import pandas as pd
    >>> from feature_engine.creation import DecisionTreeFeatures
    >>> X = dict()
    >>> X["Name"] = ["tom", "nick", "krish", "megan", "peter",
    >>>              "jordan", "fred", "sam", "alexa", "brittany"]
    >>> X["Age"] = [20, 44, 19, 33, 51, 40, 41, 37, 30, 54]
    >>> X["Height"] = [164, 150, 178, 158, 188, 190, 168, 174, 176, 171]
    >>> X["Marks"] = [1.0, 0.8, 0.6, 0.1, 0.3, 0.4, 0.8, 0.6, 0.5, 0.2]
    >>> X = pd.DataFrame(X)
    >>> y = pd.Series([4.1, 5.8, 3.9, 6.2, 4.3, 4.5, 7.2, 4.4, 4.1, 6.7])
    >>> dtf = DecisionTreeFeatures(features_to_combine=2)
    >>> dtf.fit(X, y)
    >>> dtf.transform(X)
               Name  Age  Height  ...  tree(['Age', 'Height'])  tree(['Age', 'Marks'])
    0       tom   20     164  ...                    4.100                     4.2
    1      nick   44     150  ...                    6.475                     5.6
    2     krish   19     178  ...                    4.000                     4.2
    3     megan   33     158  ...                    6.475                     6.2
    4     peter   51     188  ...                    4.400                     5.6
    5    jordan   40     190  ...                    4.400                     4.2
    6      fred   41     168  ...                    6.475                     7.2
    7       sam   37     174  ...                    4.400                     4.2
    8     alexa   30     176  ...                    4.000                     4.2
    9  brittany   54     171  ...                    6.475                     5.6
       tree(['Height', 'Marks'])
    0             6.00
    1             6.00
    2             4.24
    3             6.00
    4             4.24
    5             4.24
    6             6.00
    7             4.24
    8             4.24
    9             6.00
    """

    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        features_to_combine: Optional[Union[Iterable[Any], int]] = None,
        precision: Union[int, None] = None,
        cv=3,
        scoring: str = "neg_mean_squared_error",
        param_grid: Optional[Dict[str, Union[str, int, float, List[int]]]] = None,
        regression: bool = True,
        random_state: int = 0,
        missing_values: str = "raise",
        drop_original: bool = False,
    ) -> None:

        if precision is not None and (not isinstance(precision, int) or precision < 1):
            raise ValueError(
                "precision must be None or a positive integer. "
                f"Got {precision} instead."
            )

        if not isinstance(regression, bool):
            raise ValueError(
                f"regression must be a boolean value. Got {regression} instead."
            )

        _check_param_missing_values(missing_values)
        _check_param_drop_original(drop_original)

        self.variables = _check_variables_input_value(variables)
        self.features_to_combine = features_to_combine
        self.precision = precision
        self.cv = cv
        self.scoring = scoring
        self.param_grid = param_grid
        self.regression = regression
        self.random_state = random_state
        self.missing_values = missing_values
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits decision trees based on the input variable combinations with
        cross-validation and grid-search for hyperparameters.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training input samples. Can be the entire dataframe, not just
            the variables to transform.

        y: pandas Series or np.array = [n_samples,]
            The target variable that is used to train the decision tree.
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
            self._is_binary = type_of_target(y)

        X, y = check_X_y(X, y)

        # find or check for numerical variables
        if self.variables is None:
            variables_ = find_numerical_variables(X)
        else:
            variables_ = check_numerical_variables(X, self.variables)

        # check if dataset contains na or inf
        _check_contains_na(X, variables_)
        _check_contains_inf(X, variables_)

        if self.param_grid is not None:
            param_grid = self.param_grid
        else:
            param_grid = {"max_depth": [1, 2, 3, 4]}

        # get the sets of variables that will be used to create new features
        input_features = self._create_variable_combinations(
            how_to_combine=self.features_to_combine, variables=variables_
        )

        estimators_ = []
        for features in input_features:
            estimator = self._make_decision_tree(param_grid=param_grid)

            # single feature models
            if isinstance(features, str):
                estimator.fit(X[features].to_frame(), y)
            # multi feature models
            else:
                estimator.fit(X[features], y)

            estimators_.append(estimator)

        self.variables_ = variables_
        self.input_features_ = input_features
        self.estimators_ = estimators_
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create and add new variables.

        Parameters
        ----------
        X: Pandas DataFrame of shame = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: Pandas dataframe.
            Either the original dataframe plus the new features or
            a dataframe of only the new features.
        """
        # Check method fit has been called
        check_is_fitted(self)

        # check that input is a dataframe
        X = check_X(X)

        # Check if input data contains same number of columns as dataframe used to fit.
        _check_X_matches_training_df(X, self.n_features_in_)

        # check if dataset contains na or inf
        _check_contains_na(X, self.variables_)
        _check_contains_inf(X, self.variables_)

        # reorder variables to match train set
        X = X[self.feature_names_in_]

        # create new features and add them to the original dataframe
        # if regression or multiclass, we return the output of predict()
        if self.regression is True:
            for features, estimator in zip(self.input_features_, self.estimators_):
                if isinstance(features, str):
                    preds = estimator.predict(X[features].to_frame())
                    if self.precision is not None:
                        preds = np.round(preds, self.precision)
                    X[f"tree({features})"] = preds
                else:
                    preds = estimator.predict(X[features])
                    if self.precision is not None:
                        preds = np.round(preds, self.precision)
                    X[f"tree({features})"] = preds

        # if binary classification, we return the probability
        elif self._is_binary == "binary":
            for features, estimator in zip(self.input_features_, self.estimators_):
                if isinstance(features, str):
                    preds = estimator.predict_proba(X[features].to_frame())
                    if self.precision is not None:
                        preds = np.round(preds, self.precision)
                    X[f"tree({features})"] = preds[:, 1]
                else:
                    preds = estimator.predict_proba(X[features])
                    if self.precision is not None:
                        preds = np.round(preds, self.precision)
                    X[f"tree({features})"] = preds[:, 1]

        # if multiclass, we return the output of predict()
        else:
            for features, estimator in zip(self.input_features_, self.estimators_):
                if isinstance(features, str):
                    preds = estimator.predict(X[features].to_frame())
                    X[f"tree({features})"] = preds
                else:
                    preds = estimator.predict(X[features])
                    X[f"tree({features})"] = preds

        if self.drop_original:
            X.drop(columns=self.variables_, inplace=True)

        return X

    def _make_decision_tree(self, param_grid: Dict):
        """Instantiate decision tree."""
        if self.regression is True:
            est = DecisionTreeRegressor(random_state=self.random_state)
        else:
            est = DecisionTreeClassifier(random_state=self.random_state)

        tree_model = GridSearchCV(
            est,
            cv=self.cv,
            scoring=self.scoring,
            param_grid=param_grid,
        )

        return tree_model

    def _create_variable_combinations(
        self,
        variables: List,
        how_to_combine: Optional[Union[Iterable[Any], int]] = None,
    ) -> List[Any]:
        """
        Create a list with the combinations of variables based on the entered
        parameters.

        Parameters
        ----------
        variables: list
            The variables to combine.

        how_to_combine: int, list, tuple or None.
            How to combine the variables.

        Returns
        -------
        combos: list.
            The list of feature combinations that will be used to train the deicion
            trees.
        """
        combos = []
        if isinstance(how_to_combine, tuple):
            for feature in how_to_combine:
                if isinstance(feature, str):
                    combos.append([feature])
                else:
                    combos.append(list(feature))

        # if output_features is None, int or list.
        else:
            if how_to_combine is None:
                if len(variables) == 1:
                    combos = variables
                else:
                    for i in range(1, len(variables) + 1):
                        els = [list(x) for x in itertools.combinations(variables, i)]
                        combos += els

            elif isinstance(how_to_combine, int):
                for i in range(1, how_to_combine + 1):
                    els = [list(x) for x in itertools.combinations(variables, i)]
                    combos += els

            # output_feature is a list
            else:
                for i in how_to_combine:
                    els = [list(x) for x in itertools.combinations(variables, i)]
                    combos += els

        return [item[0] if len(item) == 1 else item for item in combos]

    def _get_new_features_name(self) -> List:
        """Return names of the created features."""
        feature_names = [f"tree({combo})" for combo in self.input_features_]
        return feature_names

    # for the check_estimator tests
    def _more_tags(self):
        tags_dict = _return_tags()
        tags_dict["requires_y"] = True
        tags_dict["variables"] = "numerical"
        return tags_dict

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
