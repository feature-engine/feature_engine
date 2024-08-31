from typing import List, Union
from types import GeneratorType

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from feature_engine.variable_handling import (
    check_all_variables,
    check_numerical_variables,
    find_all_variables,
    find_numerical_variables,
    retain_variables_if_in_df,
)

Variables = Union[int, str, List[Union[str, int]], None]


def get_feature_importances(estimator):
    """Retrieve feature importance from a fitted estimator"""

    importances = getattr(estimator, "feature_importances_", None)

    coef_ = getattr(estimator, "coef_", None)

    if coef_ is not None:

        if estimator.coef_.ndim == 1:
            importances = np.abs(coef_)

        else:
            importances = np.linalg.norm(coef_, axis=0, ord=len(estimator.coef_))

        importances = list(importances)

    return importances


def _select_all_variables(
    X: pd.DataFrame,
    variables: Variables,
    confirm_variables: bool,
    exclude_datetime: bool = False,
):
    """
    Selects the variables over which the selector will operate.

    If variables is None, it will select all variables except datetime.
    If variables is a list and confirm_variables is True, it will retain those
    variables that are present in X. If confirm_variables is False, it will use all
    variables in the list.
    """
    if variables is None:
        variables_ = find_all_variables(X, exclude_datetime)
    else:
        if confirm_variables is True:
            variables_ = retain_variables_if_in_df(X, variables)
            variables_ = check_all_variables(X, variables_)
        else:
            variables_ = check_all_variables(X, variables)
    return variables_


def _select_numerical_variables(
    X: pd.DataFrame,
    variables: Variables,
    confirm_variables: bool,
):
    """
    Selects the numerical variables over which the selector will operate.

    If variables is None, it will select all numerical variables.
    If variables is a list and confirm_variables is True, it will retain those
    numerical variables that are present in X. If confirm_variables is False, it will
    use all numerical variables in the list.
    """
    if variables is None:
        variables_ = find_numerical_variables(X)
    else:
        if confirm_variables is True:
            variables_ = retain_variables_if_in_df(X, variables)
            variables_ = check_numerical_variables(X, variables_)
        else:
            variables_ = check_numerical_variables(X, variables)
    return variables_


def find_correlated_features(
    X: pd.DataFrame,
    variables: list[Union[str, int]],
    method: str,
    threshold: float,
):
    """
    Find groups of correlated variables.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The training dataset.

    variables : list
        The variables to examine.

    method: string or callable, default='pearson'
        Can take 'pearson', 'spearman', 'kendall' or callable. It refers to the
        correlation method to be used to identify the correlated features.

        - 'pearson': standard correlation coefficient
        - 'kendall': Kendall Tau correlation coefficient
        - 'spearman': Spearman rank correlation
        - callable: callable with input two 1d ndarrays and returning a float.

        For more details on this parameter visit the  `pandas.corr()` documentation.

    threshold: float, default=0.8
        The correlation threshold above which a feature will be deemed correlated with
        another one and removed from the dataset.

    Returns
    -------

    correlated_feature_groups: set
        Sets of correlated feature groups.

    features_to_drop: list
        The list of features that have been found to be correlated to at least one
        other feature.

    correlated_feature_dict: dict
        Dictionary containing the correlated feature groups. The key is the feature
        against which all other features were evaluated. The values are the features
        correlated with the key. The key + the values should be the same as the set
        found in `correlated_feature_groups`.
    """
    # the correlation matrix
    correlated_matrix = X[variables].corr(method=method).to_numpy()

    # the correlated pairs
    correlated_mask = np.triu(np.abs(correlated_matrix), 1) > threshold

    examined = set()
    correlated_groups = list()
    features_to_drop = list()
    correlated_dict = {}
    for i, f_i in enumerate(variables):
        if f_i not in examined:
            examined.add(f_i)
            temp_set = set([f_i])
            for j, f_j in enumerate(variables):
                if f_j not in examined:
                    if correlated_mask[i, j] == 1:
                        examined.add(f_j)
                        features_to_drop.append(f_j)
                        temp_set.add(f_j)
            if len(temp_set) > 1:
                correlated_groups.append(temp_set)
                correlated_dict[f_i] = temp_set.difference({f_i})

    return correlated_groups, features_to_drop, correlated_dict


def single_feature_performance(
    X: pd.DataFrame,
    y: pd.Series,
    variables: List[Union[str, int]],
    estimator,
    cv,
    scoring,
    groups=None,
):
    """
    Trains one estimator per feature and determines the performance of that estimator.

    Parameters
    ----------
    X: pandas dataframe of shape = [n_samples, n_features]
       The input dataframe

    y: array-like of shape (n_samples)
       Target variable. Required to train the estimator.

    variables: list
        The variables to examine.

    estimator:
        Any Scikit-learn estimator.

    cv:
        Cross-validation scheme. Any supported by the Scikit-learn estimator.

    scoring:
        The performance metric. Any supported by the Scikit-learn estimator.

    groups: Array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting
        the dataset into train/test set. Only used in conjunction with a
        “Group” cv instance (e.g., GroupKFold).

    Returns
    -------
    feature_performance: dict
        A dictionary with the feature name as key and the performance of the model
        trained with that feature as value.

    feature_performance_std: dict
        A dictionary with the feature name as key and the standard deviation of the
        performance of a model trained with that feature as value.
    """
    feature_performance = {}
    feature_performance_std = {}

    cv = list(cv) if isinstance(cv, GeneratorType) else cv

    # train a model for every feature and store the performance
    for feature in variables:
        model = cross_validate(
            estimator,
            X[feature].to_frame(),
            y,
            cv=cv,
            groups=groups,
            return_estimator=False,
            scoring=scoring,
        )

        feature_performance[feature] = model["test_score"].mean()
        feature_performance_std[feature] = model["test_score"].std()
    return feature_performance, feature_performance_std


def find_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    cv,
    scoring,
    groups=None,
):
    """
    Trains an estimator using cross-validation and derives feature importance from it.
    The estimator needs to have the attributes `coef_` or `feature_importances_` after
    fitting. The importance is given by the coefficients of linear models or the purity
    gain obtained from tree-based models.

    Parameters
    ----------
    X: pandas dataframe of shape = [n_samples, n_features]
       The input dataframe

    y: array-like of shape (n_samples)
       Target variable. Required to train the estimator.

    estimator:
        A Scikit-learn estimator with parameters `coef_` or `feature_importances_`
        after fitting.

    cv:
        Cross-validation scheme. Any supported by the Scikit-learn estimator.

    scoring:
        The performance metric. Any supported by the Scikit-learn estimator.

    groups: Array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting
        the dataset into train/test set. Only used in conjunction with a
        “Group” cv instance (e.g., GroupKFold).

    Returns
    -------
    feature_importance: pd.Series
        A pandas Series with the feature name as index and its importance as value. The
        importance is given by the coefficients of linear models or the impurity gain
        from tree-based models.

    feature_importance_std: pd.Series
        A pandas Series with the feature name as key and the standard deviation of the
        feature importance as value.
    """
    cv = list(cv) if isinstance(cv, GeneratorType) else cv

    model = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring=scoring,
        return_estimator=True,
    )

    # dataframe to store the feature importance for each cv fold
    feature_importances_cv = pd.DataFrame()

    # Populate dataframe with columns containing the feature importance values
    # for each cv fold. There are as many columns as folds.
    for i in range(len(model["estimator"])):
        m = model["estimator"][i]
        feature_importances_cv[i] = get_feature_importances(m)

    # add the variables as the index to feature_importances_cv
    feature_importances_cv.index = X.columns

    # aggregate the feature importance returned in each fold
    feature_importances_ = feature_importances_cv.mean(axis=1)
    feature_importances_std_ = feature_importances_cv.std(axis=1)
    return feature_importances_, feature_importances_std_
