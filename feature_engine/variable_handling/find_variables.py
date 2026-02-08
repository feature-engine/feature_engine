"""Functions to select certain types of variables."""

import warnings
from typing import List, Tuple, Union

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.core.dtypes.common import is_numeric_dtype as is_numeric

from feature_engine.variable_handling._variable_type_checks import (
    _is_categorical_and_is_datetime,
    _is_categorical_and_is_not_datetime,
    is_object,
)


def find_numerical_variables(
    X: pd.DataFrame,
    allow_empty: bool = False,
) -> List[Union[str, int]]:
    """
    Returns a list with the names of all the numerical variables in a dataframe.

    More details in the :ref:`User Guide <find_num_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset.

    allow_empty : bool, default=False
        Whether to allow the function to return an empty list when no numerical
        variables are found. If False, the function raises an error.

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
        if allow_empty is False:
            raise TypeError(
                "No numerical variables found in this dataframe. Check "
                "variable dtypes or set allow_empty to True "
                "to return an empty list instead."
            )
        else:
            warnings.warn(
                "No numerical variables found in this dataframe. "
                "Returning an empty list.",
                UserWarning,
            )
    return variables


def find_categorical_variables(
    X: pd.DataFrame,
    allow_empty: bool = False,
) -> List[Union[str, int]]:
    """
    Returns a list with the names of all the categorical variables in a dataframe.
    Note that variables cast as object that can be parsed to datetime will be
    excluded.

    More details in the :ref:`User Guide <find_cat_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset.

    allow_empty : bool, default=False
        Whether to allow the function to return an empty list when no categorical
        variables are found. If False, the function raises an error.

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
        for column in X.select_dtypes(include=["O", "category", "string"]).columns
        if _is_categorical_and_is_not_datetime(X[column])
    ]
    if len(variables) == 0:
        if allow_empty is False:
            raise TypeError(
                "No categorical variables found in this dataframe. Check variable "
                "dtypes or set allow_empty to True to return an "
                "empty list instead."
            )
        else:
            warnings.warn(
                "No categorical variables found in this dataframe. "
                "Returning an empty list.",
                UserWarning,
            )
    return variables


def find_datetime_variables(
    X: pd.DataFrame,
    allow_empty: bool = False,
) -> List[Union[str, int]]:
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

    allow_empty : bool, default=False
        Whether to allow the function to return an empty list when no datetime
        variables are found. If False, the function raises an error.

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
        if allow_empty is False:
            raise ValueError(
                "No datetime variables found in this dataframe. Set allow_empty to "
                "True to return an empty list instead."
            )
        else:
            warnings.warn(
                "No datetime variables found in this dataframe. "
                "Returning an empty list.",
                UserWarning,
            )
    return variables


def find_all_variables(
    X: pd.DataFrame,
    exclude_datetime: bool = False,
    allow_empty: bool = False,
) -> List[Union[str, int]]:
    """
    Returns a list with the names of all the variables in the dataframe.
    Optionally, it exlcudes variables that can be parsed as datetime or datetimetz.

    More details in the :ref:`User Guide <find_all_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset.

    exclude_datetime: bool, default=False
        Whether to exclude datetime variables.

    allow_empty : bool, default=False
        Whether to allow the function to return an empty list when no variables
        are found. If False, the function raises an error.

    Returns
    -------
    variables: List
        The names of the variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import find_all_variables
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> vars_all = find_all_variables(X)
    >>> vars_all
    ['var_num', 'var_cat', 'var_date']
    """
    if exclude_datetime is True:
        DATETIME_TYPES = ("datetimetz", "datetime")
        variables = X.select_dtypes(exclude=DATETIME_TYPES).columns.to_list()
        variables = [
            var
            for var in variables
            if is_numeric(X[var]) or not _is_categorical_and_is_datetime(X[var])
        ]
    else:
        variables = X.columns.to_list()

    if len(variables) == 0:
        if allow_empty is False:
            raise ValueError(
                "No variables found in this dataframe. Set allow_empty to "
                "True to return an empty list instead."
            )
        else:
            warnings.warn(
                "No variables found in this dataframe. "
                "Returning an empty list.",
                UserWarning,
            )
    return variables


def find_categorical_and_numerical_variables(
    X: pd.DataFrame,
    variables: Union[None, int, str, List[Union[str, int]]] = None,
    allow_empty: bool = False,
) -> Tuple[List[Union[str, int]], List[Union[str, int]]]:
    """
    Find numerical and categorical variables in a dataframe or from a list.

    The function returns two lists: the first with categorical variables and
    the second with numerical variables.

    More details in the :ref:`User Guide <find_cat_and_num_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset.

    variables : list, default=None
        If `None`, the function will find all categorical and numerical variables in X.
        Alternatively, it will find categorical and numerical variables in X, selecting
        from the given list.

    allow_empty : bool, default=False
        Whether to allow the function to return empty lists when no variables
        are found. If False, the function raises an error.

    Returns
    -------
    variables: tuple
        Tuple containing a list with the categorical variables and a list with the
        numerical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import (
    >>>   find_categorical_and_numerical_variables
    >>>)    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_cat, var_num = find_categorical_and_numerical_variables(X)
    >>> var_cat, var_num
    (['var_cat'], ['var_num'])
    """
    if isinstance(variables, (str, int)):
        if X[variables].dtype.name == "category" or is_object(X[variables]):
            variables_cat = [variables]
            variables_num = []
        elif is_numeric(X[variables]):
            variables_num = [variables]
            variables_cat = []
        else:
            if allow_empty is False:
                raise TypeError(
                    "The variable entered is neither numerical nor categorical. "
                    "Set allow_empty to True to return an empty list instead."
                )
            else:
                warnings.warn(
                    "The variable entered is neither numerical nor "
                    "categorical. Returning empty lists.",
                    UserWarning,
                )
                variables_cat = []
                variables_num = []

    elif variables is None:
        variables_cat = [
            column
            for column in X.select_dtypes(include=["O", "category", "string"]).columns
            if _is_categorical_and_is_not_datetime(X[column])
        ]
        variables_num = list(X.select_dtypes(include="number").columns)
        if len(variables_num) == 0 and len(variables_cat) == 0:
            if not allow_empty:
                raise TypeError(
                    "There are no numerical or categorical variables in the "
                    "dataframe. Set allow_empty to True to return if you want "
                    "to return an empty list instead."
                )
            else:
                warnings.warn(
                    "There are no numerical or categorical variables in the "
                    "dataframe. Returning empty lists.",
                    UserWarning,
                )
                variables_cat = []
                variables_num = []
    else:
        if len(variables) == 0:
            if allow_empty is False:
                raise ValueError(
                    "The list of variables is empty. Set allow_empty to True if you "
                    "want to return an empty list instead.")
            else:
                warnings.warn(
                    "The list of variables is empty. Returning empty lists.",
                    UserWarning,
                )
                variables_cat = []
                variables_num = []

        else:
            variables_cat = list(
                X[variables].select_dtypes(include=["O", "category", "string"]).columns
            )
            variables_num = list(X[variables].select_dtypes(include="number").columns)
            if any(v for v in variables if v not in variables_cat + variables_num):
                raise TypeError(
                    "Some of the variables are neither numerical nor categorical."
                )

    return variables_cat, variables_num
