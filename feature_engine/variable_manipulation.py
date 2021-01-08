"""Functions to select certain types of variables."""

from typing import List, Union, Any

import pandas as pd

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
    X : Pandas DataFrame
    variables : variable or list of variables. Defaults to None.

    Raises
    ------
    ValueError
        If there are no numerical variables in the df or the df is empty
    TypeError
        If any of the user provided variables are not numerical

    Returns
    -------
    variables: List of numerical variables
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    elif not variables:
        # find numerical variables in dataset
        variables = list(X.select_dtypes(include="number").columns)
        if len(variables) == 0:
            raise ValueError(
                "No numerical variables in this dataframe. Please check variable"
                "format with pandas dtypes"
            )

    else:
        # check that user entered variables are of type numerical
        if any(X[variables].select_dtypes(exclude="number").columns):
            raise TypeError(
                "Some of the variables are not numerical. Please cast them as "
                "numerical before using this transformer"
            )

    return variables


def _find_or_check_categorical_variables(
    X: pd.DataFrame, variables: Variables = None
) -> List[Union[str, int]]:
    """
    Checks that variables provided by the user are of type object. If None, finds all
    the categorical (object type) variables in the DataFrame.

    Parameters
    ----------
    X : pandas DataFrame
    variables : variable or list of variables. Defaults to None.

    Raises
    ------
    ValueError
        If there are no categorical variables in df or df is empty
    TypeError
        If any of the user provided variables are not categorical

    Returns
    -------
    variables : List of numerical variables
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    elif not variables:
        variables = list(X.select_dtypes(include="O").columns)
        if len(variables) == 0:
            raise ValueError(
                "No categorical variables in this dataframe. Please check variable "
                "format with pandas dtypes"
            )

    else:
        if any(X[variables].select_dtypes(exclude="O").columns):
            raise TypeError(
                "Some of the variables are not categorical. Please cast them as object "
                "before calling this transformer"
            )

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

    if isinstance(variables, (str, int)):
        variables = [variables]

    elif not variables:
        variables = list(X.columns)

    else:
        # call pandas to test if variables entered by user are in df
        X[variables]

    return variables
