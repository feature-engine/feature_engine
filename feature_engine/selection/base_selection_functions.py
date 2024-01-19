from typing import List, Union

import numpy as np
import pandas as pd

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
    threshold: int,
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

    features_to_drop:
        The list of features that have been found to be correlated to at least another
        feature.

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
    features_to_drop = set()
    correlated_dict = {}
    for i, f_i in enumerate(variables):
        if f_i not in examined:
            examined.add(f_i)
            temp_set = set([f_i])
            for j, f_j in enumerate(variables):
                if f_j not in examined:
                    if correlated_mask[i, j] == 1:
                        examined.add(f_j)
                        features_to_drop.add(f_j)
                        temp_set.add(f_j)
            if len(temp_set) > 1:
                correlated_groups.append(temp_set)
                correlated_dict[f_i] = temp_set.difference({f_i})

    return correlated_groups, features_to_drop, correlated_dict
