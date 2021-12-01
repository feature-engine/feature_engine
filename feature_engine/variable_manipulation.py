"""Functions to select certain types of variables."""

from typing import Any, List, Union

import numpy as np
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
    Checks that variables provided by the user are of type object or categorical.
    If None, finds all the categorical and object type variables in the DataFrame.

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
    variables : List of categorical variables
    """

    if isinstance(variables, (str, int)):
        variables = [variables]

    elif not variables:
        variables = list(
            _convert_variables_to_datetime(X)
            .select_dtypes(include=["O", "category"])
            .columns
        )
        if len(variables) == 0:
            raise ValueError(
                "No categorical variables in this dataframe. Please check the "
                "variables format with pandas dtypes"
            )

    else:
        if any(
            _convert_variables_to_datetime(X, variables)[variables]
            .select_dtypes(exclude=["O", "category"])
            .columns
        ):
            raise TypeError(
                "Some of the variables are not categorical. Please cast them as object "
                "or category before calling this transformer"
            )

    return variables


def _convert_variable_to_datetime(X: pd.Series, **kwargs) -> pd.Series:
    """
    Take a series and attempt to convert its dtype to datetime unless
    its type is numeric. In that case, return the series as is.
    If the series type is object or category, it is processed as follows:
        1 if it could be cast as number, return the series as is
        2 otherwise, attempt to convert it to datetime; if pd.to_datetime
          fails the conversion, return the series as is

    Arguments
    ---------
    X: pd.Series that may or may not contain values convertible to datetime
    kwargs: see pd.to_datetime()

    **Notes**
    if the series type is category, it is the dtype of the categories
    that is converted, not the series itself This means that the primary type
    is still going to be category whereas the pd.Series.dtype.categories
    property is going to have its type converted.

    Missing values in str/cat objects are going to be replaced by 'NaT'
    in the converted datetime columns
    **
    """

    if hasattr(X, "dt"):
        return X

    if isinstance(X.dtype, (pd.CategoricalDtype, object)):
        try:
            X.astype("float64")
            return X
        except ValueError:
            return X.apply(pd.to_datetime, errors="ignore", **kwargs)

    elif np.issubdtype(X.dtype, np.number):
        return X

    return X.apply(pd.to_datetime, errors="ignore", **kwargs)


def _convert_variables_to_datetime(
    X: pd.DataFrame, variables: Variables = None, **kwargs
) -> pd.DataFrame:
    """
    Takes a dataframe and returns it with object/category features
    that represent a datetime converted into datetime variables.
    It does not manipulate numerical features or object/category features
    that could be casted into numerical type (see _convert_variable_to_datetime)
    If variables == None, process all the columns in the dataframe.
    """

    variables = X.columns if not variables else variables
    X_dt = X.copy()
    for var in X_dt[variables].select_dtypes(exclude="number"):
        X_dt[var] = _convert_variable_to_datetime(X[var], **kwargs)

    return X_dt


def _find_or_check_datetime_variables(
    X: pd.DataFrame, variables: Variables = None
) -> List[Union[str, int]]:
    """
    Checks that variables provided by the user are of type datetime,
    and transform date/times given in str/obj format into datetimes.
    If None, finds all the datetime type variables in the DataFrame.

    Parameters
    ----------
    X : pandas DataFrame
    variables : variable or list of variables. Defaults to None.

    Raises
    ------
    ValueError
        If there are no datetime variables in df or df is empty
    TypeError
        If any of the user provided variables are not datetime or
        [failed to convert str/obj to datetime]?

    Returns
    -------
    variables : List of datetime variables
    """

    if not variables:
        X_conv = _convert_variables_to_datetime(X)
        variables = list(
            X_conv.columns[[hasattr(X_conv[s], "dt") for s in X_conv.columns]]
        )
        print(variables)
        if len(variables) == 0:
            raise ValueError(
                "No datetime variables in this dataframe. "
                "Note: purely numeric variables representing dates or times "
                "will not be treated as datetime by this transformer."
            )

    if isinstance(variables, (str, int)):
        variables = [variables]

    X_conv = _convert_variables_to_datetime(X, variables)[variables]
    if not all(hasattr(X_conv[s], "dt") for s in X_conv.columns):
        raise TypeError(
            "Some of the variables are not or could not be converted to datetime. "
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
