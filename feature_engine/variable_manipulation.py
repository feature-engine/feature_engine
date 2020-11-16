# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause
# functions shared across transformers

from typing import List, Optional, Union, Any

import pandas as pd

Variables = Union[None, int, str, List[Union[str, int]]]

# set return value typehint to Any here to avoid issues with the base transformer fit methods
def _check_input_parameter_variables(variables: Variables) -> Any:
    """
    Checks that the input is of the correct type

    Args:
        variables: string, int, list of strings, list of integers. Default=None

    Returns:
        Returns the same input
    """
    if variables:
        if isinstance(variables, list):
            if not all(isinstance(i, (str, int)) for i in variables):
                raise ValueError(
                    "Variables should be string, int, list of strings, list of integers"
                )
        else:
            if not isinstance(variables, (str, int)):
                raise ValueError(
                    "Variables should be string, int, list of strings, list of integers"
                )

    return variables


def _find_or_check_numerical_variables(
    X: pd.DataFrame, variables: Union[None, int, str, List[Union[str, int]]] = None
) -> List[Union[str, int]]:
    """
    Checks that variables provided by the user are of type numerical. If None was
    entered, finds all the numerical variables in the DataFrame.

    Args:
        X: DataFrame
        variables: variable or list of variables. Defaults to None.

    Raises:
        ValueError: If all variables are non-numerical type or DataFrame is empty
        TypeError: If any user provided variables are non-numerical

    Returns:
        List of variables
    """
    if isinstance(variables, (str, int)):
        variables = [variables]

    if not variables:
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
                "numerical before calling this transformer"
            )

    return variables


def _find_or_check_categorical_variables(
    X: pd.DataFrame, variables: Union[None, int, str, List[Union[str, int]]] = None
) -> List[Union[str, int]]:
    """
    Checks that variables provided by the user are of type object. If None was
    entered, finds all the categorical (object type) variables in the DataFrame.

    Args:
        X: DataFrame
        variables: variable or list of variables. Defaults to None.

    Raises:
        ValueError: If all variables are non-categorical type or DataFrame is empty
        TypeError: If some of the variables are non-categorical

    Returns:
        List of variables
    """
    if isinstance(variables, (str, int)):
        variables = [variables]

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
