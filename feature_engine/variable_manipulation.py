# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause
# functions shared across transformers

from typing import List, Optional, Union

import pandas as pd


def _define_variables(
    variables: Union[str, Optional[List[str]]]
) -> Optional[List[str]]:
    """
    Takes string or list of strings and checks if argument is list of strings.
    Can take None as argument.

    Args:
        variables: string or list of strings

    Returns:
        List of strings
    """

    if not variables or (
        isinstance(variables, list) and all(isinstance(i, str) for i in variables)
    ):
        variables = variables

    else:
        if isinstance(variables, str):
            variables = [variables]

        else:
            raise ValueError("Variables should be string or list of strings")

    return variables


def _find_numerical_variables(
    X: pd.DataFrame, variables: Optional[List[str]] = None
) -> List[str]:
    """
    Takes Pandas DataFrame and checks if user provided variables
    are numerical type. If no variables are provided by the user,
    it captures all the numerical variables presented in DataFrame.

    Args:
        X: DataFrame to perform the check against
        variables: List of variables. Defaults to None.

    Raises:
        ValueError: If all variables are non-numerical type or DataFrame is empty
        TypeError: If user provided variables are non-numerical

    Returns:
        List of variables
    """

    if not variables:
        variables = list(X.select_dtypes(include="number").columns)
        if len(variables) == 0:
            raise ValueError(
                "No numerical variables in this dataframe. Please check variable"
                "format with pandas dtypes"
            )

    else:
        if any(X[variables].select_dtypes(exclude="number").columns):
            raise TypeError(
                "Some of the variables are not numerical. Please cast them as "
                "numerical before calling this transformer"
            )

    return variables


def _find_categorical_variables(
    X: pd.DataFrame, variables: Optional[List[str]] = None
) -> List[str]:
    """
    Takes Pandas DataFrame and finds all categorical variables if not provided.
    If variables are provided, checks if they are indeed categorical.

    Args:
        X: DataFrame to perform the check against
        variables: List of variables. Defaults to None.

    Raises:
        ValueError: If all variables are non-categorical type or DataFrame is empty
        TypeError: If some of the variables are non-categorical

    Returns:
        List of variables
    """

    if not variables:
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
    X: pd.DataFrame, variables: Optional[List[str]] = None
) -> List[str]:
    """
    If variables are None, captures all variables in the dataframe in a list.
    If user enters variable names list, it returns the list.

    Args:
        X:  DataFrame to perform the check against
        variables: List of variables. Defaults to None.

    Raises:
        TypeError: If variable list provided by user contains
                    variables not present in the dataframe

    Returns:
        List of variables
    """

    if not variables:
        variables = list(X.columns)

    else:
        # variables indicated by user
        if any(set(variables).difference(X.columns)):
            raise TypeError(
                "Some variables are not present in the dataset. Please check your "
                "variable list."
            )

    return variables
