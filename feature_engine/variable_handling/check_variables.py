"""Functions to check that the variables in a list are of a certain type."""

from typing import List, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype as is_numeric

from feature_engine.variable_handling._variable_type_checks import (
    _is_categorical_and_is_datetime,
)
from feature_engine.variable_handling.dtypes import DATETIME_TYPES

Variables = Union[int, str, List[Union[str, int]]]


def check_numerical_variables(
    X: pd.DataFrame, variables: Variables
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are of type numerical.

    More details in the :ref:`User Guide <check_num_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset.

    variables : List
        The list with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the numerical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_numerical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_ = check_numerical_variables(X, variables=["var_num"])
    >>> var_
    ['var_num']
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    if len(X[variables].select_dtypes(exclude="number").columns) > 0:
        raise TypeError(
            "Some of the variables are not numerical. Please cast them as "
            "numerical before using this transformer."
        )

    return variables


def check_categorical_variables(
    X: pd.DataFrame, variables: Variables
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are of type object or categorical.

    More details in the :ref:`User Guide <check_cat_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list
        The list with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the categorical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_categorical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_ = check_categorical_variables(X, "var_cat")
    >>> var_
    ['var_cat']
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    if len(X[variables].select_dtypes(exclude=["O", "category"]).columns) > 0:
        raise TypeError(
            "Some of the variables are not categorical. Please cast them as "
            "object or categorical before using this transformer."
        )

    return variables


def check_datetime_variables(
    X: pd.DataFrame,
    variables: Variables,
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are or can be parsed as datetime and or
    datetimetz.

    More details in the :ref:`User Guide <check_datetime_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list
        The list with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the datetime variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_datetime_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_date = check_datetime_variables(X, "var_date")
    >>> var_date
    ['var_date']
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    # find non datetime variables, if any:
    non_datetime_vars = []
    for column in X[variables].select_dtypes(exclude=DATETIME_TYPES):
        if is_numeric(X[column]) or not _is_categorical_and_is_datetime(X[column]):
            non_datetime_vars.append(column)

    if len(non_datetime_vars) > 0:
        raise TypeError(
            "Some of the variables are not or cannot be parsed as datetime."
        )

    return variables


def check_all_variables(
    X: pd.DataFrame,
    variables: Variables,
) -> List[Union[str, int]]:
    """
    Checks that the variables in the list are in the dataframe.

    More details in the :ref:`User Guide <check_all_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list
        The list with the names of the variables to check.

    Returns
    -------
    variables: List
        The names of the variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import check_all_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> vars_all = check_all_variables(X, ['var_num', 'var_cat', 'var_date'])
    >>> vars_all
    ['var_num', 'var_cat', 'var_date']
    """
    if isinstance(variables, (str, int)):
        if variables not in X.columns.to_list():
            raise KeyError(f"The variable {variables} is not in the dataframe.")
        variables_ = [variables]

    else:
        if not set(variables).issubset(set(X.columns)):
            raise KeyError("Some of the variables are not in the dataframe.")

        variables_ = variables

    return variables_
