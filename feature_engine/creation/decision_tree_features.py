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
    missing_values=_missing_values_docstring,
    drop_original=_drop_original_docstring,
    variables_=_variables_attribute_docstring,
    feature_names_in_=_feature_names_in_docstring,
    n_features_in_=_n_features_in_docstring,
    transform=_transform_creation_docstring,
    fit_transform=_fit_transform_docstring,
)
class DecisionTreeFeatures(BaseEstimator, TransformerMixin, GetFeatureNamesOutMixin):
    """
    DecisionTreeFeatures() creates new variables by transforming a variable, or
    combining 2 or more variables with decision trees.

    Parameters
    ----------
    {variables}

    features_to_combine: integer, list or tuple, default=None
        Used to determine how the variables indicated in `variables` will be combined
        to obtain the new features by using decision trees. If integer, then the value
        corresponds to the largest size of combinations allowed between features. For
        example, if you want to combine three variables, ["var_A", "var_B", "var_C"],
        and:
            - features_to_combine = 1, the transformer returns new features based on the
                predictions of a decision tree trained on each individual variable,
                generating 3 new features.
            - features_to_combine = 2, the transformer returns the features from
                `features_to_combine=1`, plus features based on the predictions of a
                decision tree based on all possible combinations of 2 variables, i.e.,
                ("var_A", "var_B"), ("var_A", "var_C"), and ("var_B", "var_C"),
                resulting in a total of 6 new features.
            - features_to_combine = 3, the transformer returns the features from
                `features_to_combine=2`, plus one additional feature based on the
                predictions of a decision trained on the 3 variables,
                ["var_A", "var_B", "var_C"], resulting in a total of 7 new features.

        If list, the list must contain integers indicating the number of features that
        should be used as input of a decision tree. For example, if the data has 4
        variables, ["var_A", "var_B", "var_C", "var_D"] and and
        `features_to_combine = [2,3]`, then the following combinations will be used to
        create new features using decision trees: ("var_A", "var_B"),
        ("var_A", "var_C"), ("var_A", "var_D"), ("var_B", "var_C"), ("var_B", "var_D"),
        ("var_C", "var_D"), ("var_A", "var_B", "var_C"), ("var_A", "var_B", "var_D"),
        ("var_A", "var_C", "var_D"), and ("var_B", "var_C", "var_D").

        If tuple, the tuple must contain strings and/or tuples that indicate how to
        combine the variables to create the new features. For example, if
        `features_to_combine=("var_C", ("var_A", "var_C"), "var_C", ("var_B", "var_D")`,
        then, the transformer will train a decision tree based of each value within the
        tuple, resulting in 4 new features.

        If None, then the transformer will create all possible combinations of 1 or
        more features, and use those as inputs to decision trees. This is equivalent to
        passing an integer that is equal to the number of variables to combine.

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

    References
    ----------
    .. [1] Niculescu-Mizil, et al. "Winning the KDD Cup Orange Challenge with Ensemble
        Selection". JMLR: Workshop and Conference Proceedings 7: 23-34. KDD 2009
        http://proceedings.mlr.press/v7/niculescu09/niculescu09.pdf

    Examples
    --------
    TBS
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
        tags_dict["variables"] = "numerical"
        tags_dict["allow_nan"] = True
        tags_dict["variables"] = "skip"
        # Tests that are OK to fail:
        tags_dict["_xfail_checks"][
            "check_parameters_default_constructible"
        ] = "transformer has 1 mandatory parameter"
        tags_dict["_xfail_checks"][
            "check_fit2d_1feature"
        ] = "this transformer works with datasets that contain at least 2 variables. \
        Otherwise, there is nothing to combine"
        return tags_dict
