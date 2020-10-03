# Authors: Soledad Galli <solegalli@protonmail.com>
# License: BSD 3 clause
# functions shared across transformers

from typing import List, Optional, Union

import pandas as pd


def _define_variables(variables: Union[Optional[str], List[str]]) -> Union[Optional[str], List[str]]:
    """
    Takes string or list of strings and checks if argument is list of strings.
    Can take None as argument.

    Args:
        variables: string or list of strings

    Returns:
        List of strings
    """

    #! Can take Tuple of variables and return list of tuple. Is it accaptable?
    if not variables or isinstance(variables, list):
        variables = variables
    else:
        variables = [variables]

    return variables


def _define_numerical_dict(the_dict: Union[Optional[str], dict]) -> Union[Optional[str], dict]:
    """
    Takes dictionary and checks if all values in dictionary are integers and floats.
    Can take None as argument.

    Args:
        the_dict: Dict to perform check against

    Raises:
        ValueError: If all values of dict are not int or float
        TypeError: When argument type is not dict

    Returns:
        None or the dict
    """

    if not the_dict:
        the_dict = the_dict
    elif isinstance(the_dict, dict):
        if not all([isinstance(x, (float, int)) for x in the_dict.values()]):
            raise ValueError('All values in the dictionary must be integer or float')
    else:
        raise TypeError('The parameter can only take a dictionary or None')

    return the_dict


def _find_numerical_variables(X: pd.DataFrame, variables: List[str]=None) -> List[str]:
    """
    Takes Pandas DataFrame along with list of colum names and
    checks if these columns are numerical. If not provided columns,
    checks if all of them are numerical.

    Args:
        X: DataFrame to perform the check against
        variables: List of variables. Defaults to None.

    Raises:
        ValueError: If all variables are non-numerical type or DataFrame is empty
        TypeError: If some of the variables are non-numerical

    Returns:
        List of variables
    """

    if not variables:
        variables = list(X.select_dtypes(include='number').columns)
        if len(variables) == 0:
            raise ValueError('No numerical variables in this dataframe. Please check variable format with dtypes')
    else:
        if len(X[variables].select_dtypes(exclude='number').columns) != 0:
            raise TypeError("Some of the variables are not numerical. Please cast them as numerical "
                            "before calling this transformer")

    return variables


def _find_categorical_variables(X: pd.DataFrame, variables: List[str]=None) -> List[str]:
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
        variables = list(X.select_dtypes(include='O').columns)
        if len(variables) == 0:
            raise ValueError('No categorical variables in this dataframe. Please check variable format with dtypes')
    else:
        # variables indicated by user
        if len(X[variables].select_dtypes(exclude='O').columns) != 0:
            raise TypeError("Some of the variables are not categorical. Please cast them as object "
                            "before calling this transformer")

    return variables


def _find_all_variables(X: pd.DataFrame, variables: List[str]=None) -> List[str]:
    """
    Takes Pandas DataFrame and extracts all variables if not provided corresponding list.
    If variables are provided, returns them back.

    Args:
        X:  DataFrame to perform the check against
        variables: List of variables. Defaults to None.

    Raises:
        TypeError: If user provided list of variables include variable(s),
                    which does not exist in DataFrame

    Returns:
        List of variables
    """
    # Find all variables in a data set
    if not variables:
        variables = list(X.columns)
    else:
        # variables indicated by user
        if len(set(variables).difference(X.columns)) != 0:
            raise TypeError("Some variables are not present in the dataset. Please check your variable"
                            " list")

    return variables
