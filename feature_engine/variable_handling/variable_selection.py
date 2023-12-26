"""Functions to select certain types of variables."""

from typing import List, Tuple, Union

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_numeric_dtype as is_numeric
from pandas.api.types import is_object_dtype as is_object

from feature_engine.variable_handling._variable_type_checks import (
    _is_categorical_and_is_not_datetime,
)

Variables = Union[None, int, str, List[Union[str, int]]]


def find_all_variables(
    X: pd.DataFrame,
    variables: Variables = None,
    exclude_datetime: bool = False,
) -> List[Union[str, int]]:
    """
    Returns the names of all the variables in the dataframe in a list. Alternatively,
    it will check that the variables indicated by the user are in the dataframe.

    More details in the :ref:`User Guide <find_all_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list, default=None
        If `None`, the function will return the names of all the variables in X.
        Alternatively, it checks that the variables in the list are present in the
        dataframe.

    exclude_datetime: bool, default=False
        Whether to exclude datetime variables.

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
    # find all variables in dataset
    if variables is None:
        if exclude_datetime is True:
            variables_ = X.select_dtypes(exclude="datetime").columns.to_list()
        else:
            variables_ = X.columns.to_list()

    elif isinstance(variables, (str, int)):
        if variables not in X.columns.to_list():
            raise ValueError(f"The variable {variables} is not in the dataframe.")
        variables_ = [variables]

    else:
        if len(variables) == 0:
            raise ValueError("The list of variables is empty.")

        if any(f for f in variables if f not in X.columns):
            raise KeyError("Some of the variables are not in the dataframe.")

        variables_ = variables

    return variables_


def _filter_out_variables_not_in_dataframe(X, variables):
    """Filter out variables that are not present in the dataframe.

    Function removes variables that the user defines in the argument `variables`
    but that are not present in the input dataframe.

    Useful when using several feature selection procedures in a row. The dataframe
    input to the first selection algorithm likely contains more variables than the
    input dataframe to subsequent selection algorithms, and it is not possible a
    priori, to say which variable will be dropped.

    Parameters
    ----------
    X:  pandas DataFrame
    variables: string, int or list of (strings or int).

    Returns
    -------
    filtered_variables: List of variables present in `variables` and in the
        input dataframe.
    """
    # When variables is not defined, keep it like this and return None.
    if variables is None:
        return None

    # If an integer or a string is provided, convert to a list.
    if not isinstance(variables, list):
        variables = [variables]

    # Filter out elements of variables that are not in the dataframe.
    filtered_variables = [var for var in variables if var in X.columns]

    # Raise an error if no column is left to work with.
    if len(filtered_variables) == 0:
        raise ValueError(
            "After filtering no variable remaining. At least 1 is required."
        )

    return filtered_variables


def find_categorical_and_numerical_variables(
    X: pd.DataFrame, variables: Variables = None
) -> Tuple[List[Union[str, int]], List[Union[str, int]]]:
    """
    Find numerical and categorical variables.

    The function returns two lists; the first one with the names of the variables of
    type object or categorical and the second list with the names of the numerical
    variables.

    More details in the :ref:`User Guide <find_cat_and_num_vars>`.

    Parameters
    ----------
    X : pandas dataframe of shape = [n_samples, n_features]
        The dataset

    variables : list, default=None
        If `None`, the function will find categorical and numerical variables in X.
        Alternatively, it will find categorical and numerical variables in the given
        list.

    Returns
    -------
    variables: tuple
        List of numerical and list of categorical variables.

    Examples
    --------
    >>> import pandas as pd
    >>> from feature_engine.variable_handling import (
    >>>   find_categorical_and_numerical_variables
    >>>)
    >>> X = pd.DataFrame({
    >>>     "var_num": [1, 2, 3],
    >>>     "var_cat": ["A", "B", "C"],
    >>>     "var_date": pd.date_range("2020-02-24", periods=3, freq="T")
    >>> })
    >>> var_cat, var_num = find_categorical_and_numerical_variables(X)
    >>> var_cat, var_num
    (['var_cat'], ['var_num'])
    """

    # If the user passes just 1 variable outside a list.
    if isinstance(variables, (str, int)):

        if X[variables].dtype.name == "category" or is_object(X[variables]):
            variables_cat = [variables]
            variables_num = []
        elif is_numeric(X[variables]):
            variables_num = [variables]
            variables_cat = []
        else:
            raise TypeError(
                "The variable entered is neither numerical nor categorical."
            )

    # If user leaves default None parameter.
    elif variables is None:
        # find categorical variables
        if variables is None:
            variables_cat = [
                column
                for column in X.select_dtypes(include=["O", "category"]).columns
                if _is_categorical_and_is_not_datetime(X[column])
            ]
        # find numerical variables in dataset
        variables_num = list(X.select_dtypes(include="number").columns)

        if len(variables_num) == 0 and len(variables_cat) == 0:
            raise TypeError(
                "There are no numerical or categorical variables in the dataframe"
            )

    # If user passes variable list.
    else:
        if len(variables) == 0:
            raise ValueError("The list of variables is empty.")

        # find categorical variables
        variables_cat = [
            var for var in X[variables].select_dtypes(include=["O", "category"]).columns
        ]

        # find numerical variables
        variables_num = list(X[variables].select_dtypes(include="number").columns)

        if any([v for v in variables if v not in variables_cat + variables_num]):
            raise TypeError(
                "Some of the variables are neither numerical nor categorical."
            )

    return variables_cat, variables_num
