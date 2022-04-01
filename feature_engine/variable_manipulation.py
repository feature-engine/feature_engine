"""Functions to select certain types of variables."""

from typing import Any, List, Tuple, Union

import pandas as pd
from pandas.api.types import is_categorical_dtype as is_categorical
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_numeric_dtype as is_numeric
from pandas.api.types import is_object_dtype as is_object

Variables = Union[None, int, str, List[Union[str, int]]]


# set return value typehint to Any to avoid issues with the base transformer fit method
def _check_input_parameter_variables(variables: Variables) -> Any:
    """
    Checks that the input is of the correct type. Allowed  values are None, int, str or
    list of strings and ints.

    Parameters
    ----------
    variables : string, int, list of strings, list of integers. Default=None

    Returns
    -------
    variables: same as input
    """

    msg = "variables should be a string, an int or a list of strings or integers."

    if variables:
        if isinstance(variables, list):
            if not all(isinstance(i, (str, int)) for i in variables):
                raise ValueError(msg)
        else:
            if not isinstance(variables, (str, int)):
                raise ValueError(msg)

    return variables


def _find_or_check_numerical_variables(
    X: pd.DataFrame, variables: Variables = None
) -> List[Union[str, int]]:
    """
    Checks that variables provided by the user are of type numerical. If None, finds
    all the numerical variables in the DataFrame.

    Parameters
    ----------
    X : Pandas DataFrame.
    variables : variable or list of variables. Defaults to None.

    Raises
    ------
    ValueError
        If there are no numerical variables in the df or the df is empty.
    TypeError
        If any of the user provided variables are not numerical.

    Returns
    -------
    variables: List of numerical variables.
    """

    if variables is None:
        # find numerical variables in dataset
        variables = list(X.select_dtypes(include="number").columns)
        if len(variables) == 0:
            raise ValueError(
                "No numerical variables found in this dataframe. Please check "
                "variable format with pandas dtypes."
            )

    elif isinstance(variables, (str, int)):
        if is_numeric(X[variables]):
            variables = [variables]
        else:
            raise TypeError("The variable entered is not numeric.")

    else:
        if len(variables) == 0:
            raise ValueError("The list of variables is empty.")

        # check that user entered variables are of type numerical
        else:
            if len(X[variables].select_dtypes(exclude="number").columns) > 0:
                raise TypeError(
                    "Some of the variables are not numerical. Please cast them as "
                    "numerical before using this transformer."
                )

    return variables


def _is_convertible_to_num(column: pd.Series) -> bool:
    return is_numeric(pd.to_numeric(column, errors="ignore"))


def _is_convertible_to_dt(column: pd.Series) -> bool:
    return is_datetime(pd.to_datetime(column, errors="ignore", utc=True))


def _is_categories_num(column: pd.Series) -> bool:
    return is_numeric(column.dtype.categories)


def _is_categorical_and_is_not_datetime(column: pd.Series) -> bool:

    # check for datetime only if object cannot be cast as numeric because
    # if it could pd.to_datetime would convert it to datetime regardless
    if is_object(column):
        is_cat = _is_convertible_to_num(column) or not _is_convertible_to_dt(column)

    # check for datetime only if the type of the categories is not numeric
    # because pd.to_datetime throws an error when it is an integer
    elif is_categorical(column):
        is_cat = _is_categories_num(column) or not _is_convertible_to_dt(column)

    return is_cat


def _find_or_check_categorical_variables(
    X: pd.DataFrame, variables: Variables = None
) -> List[Union[str, int]]:
    """
    Checks that variables provided by the user are of type object or categorical.
    If None, finds all the categorical and object type variables in the DataFrame.

    Parameters
    ----------
    X : pandas DataFrame.
    variables : variable or list of variables. Defaults to None.

    Raises
    ------
    ValueError
        If there are no categorical variables in df or df is empty.
    TypeError
        If any of the user provided variables are not categorical.

    Returns
    -------
    variables : List of categorical variables.
    """

    if variables is None:
        # find categorical variables in dataset
        variables = [
            column
            for column in X.select_dtypes(include=["O", "category"]).columns
            if _is_categorical_and_is_not_datetime(X[column])
        ]
        if len(variables) == 0:
            raise ValueError(
                "No categorical variables found in this dataframe. Please check "
                "variable format with pandas dtypes."
            )

    elif isinstance(variables, (str, int)):
        if is_categorical(X[variables]) or is_object(X[variables]):
            variables = [variables]
        else:
            raise TypeError("The variable entered is not categorical.")

    else:
        if len(variables) == 0:
            raise ValueError("The list of variables is empty.")

        # check that user entered variables are of type categorical
        else:
            if len(X[variables].select_dtypes(exclude=["O", "category"]).columns) > 0:
                raise TypeError(
                    "Some of the variables are not categorical. Please cast them as "
                    "categorical or object before using this transformer."
                )

    return variables


def _is_categorical_and_is_datetime(column: pd.Series) -> bool:

    # check for datetime only if object cannot be cast as numeric because
    # if it could pd.to_datetime would convert it to datetime regardless
    if is_object(column):
        is_dt = not _is_convertible_to_num(column) and _is_convertible_to_dt(column)

    # check for datetime only if the type of the categories is not numeric
    # because pd.to_datetime throws an error when it is an integer
    elif is_categorical(column):
        is_dt = not _is_categories_num(column) and _is_convertible_to_dt(column)

    return is_dt


def _find_or_check_datetime_variables(
    X: pd.DataFrame, variables: Variables = None
) -> List[Union[str, int]]:
    """
    Checks that variables provided by the user are of type datetime.
    If None, finds all datetime variables in the DataFrame.

    Parameters
    ----------
    X : pandas DataFrame
    variables : variable or list of variables. Defaults to None.

    Returns
    -------
    variables : List of datetime variables.
    """

    if variables is None:
        variables = [
            column
            for column in X.select_dtypes(exclude="number").columns
            if is_datetime(X[column]) or _is_categorical_and_is_datetime(X[column])
        ]

        if len(variables) == 0:
            raise ValueError("No datetime variables found in this dataframe.")

    elif isinstance(variables, (str, int)):

        if is_datetime(X[variables]) or (
            not is_numeric(X[variables])
            and _is_categorical_and_is_datetime(X[variables])
        ):
            variables = [variables]
        else:
            raise TypeError("The variable entered is not datetime.")

    else:
        if len(variables) == 0:
            raise ValueError("The indicated list of variables is empty.")

        # check that the variables entered by the user are datetime
        else:
            vars_non_dt = [
                column
                for column in X[variables].select_dtypes(exclude="datetime")
                if is_numeric(X[column])
                or not _is_categorical_and_is_datetime(X[column])
            ]

            if len(vars_non_dt) > 0:
                raise TypeError("Some of the variables are not datetime.")

    return variables


def _find_all_variables(
    X: pd.DataFrame, variables: Variables = None
) -> List[Union[str, int]]:
    """
    If variables are None, captures all variables in the dataframe in a list.
    If user enters variable names in list, it returns the list.

    Parameters
    ----------
    X :  pandas DataFrame
    variables : List of variables. Defaults to None.

    Raises
    ------
    TypeError
        If the variable list provided by the user contains variables not present in the
        dataframe.

    Returns
    -------
    variables : List of numerical variables
    """

    if variables is None:
        # find all variables in dataset
        variables_ = X.columns.to_list()

    elif isinstance(variables, (str, int)):
        if variables not in X.columns.to_list():
            raise ValueError("The variable is not in the dataframe.")
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

    Useful when ussing several feature selection procedures in a row. The dataframe
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


def _find_categorical_and_numerical_variables(
    X: pd.DataFrame, variables: Variables = None
) -> Tuple[List[Union[str, int]], List[Union[str, int]]]:
    """
    Find numerical and categorical variables.

    Parameters
    ----------
    X :  pandas DataFrame

    variables : List of variables. Defaults to None.

    Returns
    -------
    variables : Tuple with List of numerical and list of categorical variables.
    """

    # If the user passes just 1 variable outside a list.
    if isinstance(variables, (str, int)):

        if is_categorical(X[variables]) or is_object(X[variables]):
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
