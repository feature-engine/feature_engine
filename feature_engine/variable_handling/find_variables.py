from typing import List, Union

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from feature_engine.variable_handling._variable_type_checks import (
    _is_categorical_and_is_datetime,
    _is_categorical_and_is_not_datetime,
)


def find_numerical_variables(X: pd.DataFrame) -> List[Union[str, int]]:
    """
    Returns a list with the names of all the numerical variables in a dataframe.

    More details in the :ref:`User Guide <find_num_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    Returns
    -------
    variables: List
        The names of the numerical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_numerical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_ = find_numerical_variables(X)
    >>> var_
    ['var_num']
    """
    variables = list(X.select_dtypes(include="number").columns)
    if len(variables) == 0:
        raise TypeError(
            "No numerical variables found in this dataframe. Please check "
            "variable format with pandas dtypes."
        )
    return variables


def find_categorical_variables(X: pd.DataFrame) -> List[Union[str, int]]:
    """
    Returns a list with the names of all the categorical variables in a dataframe.
    Note that variables cast as object that can be parsed to datetime will be
    excluded.

    More details in the :ref:`User Guide <find_cat_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    Returns
    -------
    variables: List
        The names of the categorical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_categorical_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_ = find_categorical_variables(X)
    >>> var_
    ['var_cat']
    """
    variables = [
        column
        for column in X.select_dtypes(include=["O", "category"]).columns
        if _is_categorical_and_is_not_datetime(X[column])
    ]
    if len(variables) == 0:
        raise TypeError(
            "No categorical variables found in this dataframe. Please check "
            "variable format with pandas dtypes."
        )
    return variables


def find_datetime_variables(X: pd.DataFrame) -> List[Union[str, int]]:
    """
    Returns a list with the names of the variables that are or can be parsed as
    datetime.

    Note that this function will select variables cast as object if they can be cast as
    datetime as well.

    More details in the :ref:`User Guide <find_datetime_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset.

    Returns
    -------
    variables: List
        The names of the datetime variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_datetime_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_date = find_datetime_variables(X)
    >>> var_date
    ['var_date']
    """

    variables = [
        column
        for column in X.select_dtypes(exclude="number").columns
        if is_datetime(X[column]) or _is_categorical_and_is_datetime(X[column])
    ]

    if len(variables) == 0:
        raise ValueError("No datetime variables found in this dataframe.")

    return variables
