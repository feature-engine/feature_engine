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
        exclude_datetime: bool=False,
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
